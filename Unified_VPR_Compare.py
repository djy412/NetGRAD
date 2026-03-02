# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 22:36:04 2025

@author: djy41
"""
import os, time, math, random, re, argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw

# =========================
# Your existing dataloaders (unchanged)
# =========================
from dataloading import (
    get_Pittsburg30k_Train_dataloaders,
    get_Pittsburg30k_Triplet_Train_dataloaders,
    get_Pittsburgh30k_Eval_db_dataloaders_5m,
    get_Pittsburgh30k_Eval_query_dataloaders_5m,
    get_Tokyo_Query_dataloaders,
    get_Tokyo_Eval_dataloaders,
    get_Tokyo_Triplet_Train_dataloaders,
    get_Tokyo_Train_dataloaders,
    get_RobotCar_Train_dataloaders,
    get_RobotCar_Triplet_Train_dataloaders,
    get_RobotCar_Eval_db_dataloaders,
    get_RobotCar_Eval_query_dataloaders,
)

# =========================
# Global flags (kept for GR-VLAD compatibility)
# =========================
Pretrain = False
Global_Train = True

# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Method selector ----
    method: str = "netvlad"             # one of: {"netvlad", "apgem", "grvlad"}

    # ---- Common training ----
    batch_size: int = 32
    num_workers: int = 8
    recall_ks: Tuple[int, ...] = (1,2,3,4,5,10,20,25)

    # ---- NetVLAD settings ----
    vlad_clusters: int = 64
    vlad_dim: int = 512
    vlad_alpha: float = 100.0
    vlad_normalize_input: bool = True
    vlad_epochs_ft: int = 5
    vlad_lr_backbone: float = 1e-5
    vlad_lr_head: float = 1e-3
    vlad_margin: float = 0.1
    vlad_kmeans_samples: int = 100000

    # ---- AP-GeM settings ----
    gem_p_init: float = 3.0
    gem_learn_p: bool = True
    apgem_proj_dim: int = 2048       # set 0 to disable
    apgem_normalize_input: bool = True
    apgem_epochs_ft: int = 5
    apgem_lr_backbone: float = 1e-5
    apgem_lr_head: float = 1e-3
    apgem_margin: float = 0.1
    apgem_use_fastap: bool = False
    apgem_fastap_bins: int = 10

    # ---- Global Residual VLAD settings (your model) ----
    latent_dim: int = 512
    K: int = 256
    tau_init: float = 0.35
    concat: bool = True
    proj_dim: int = 1024

    epochs_warmup: int = 50
    epochs_joint: int = 20
    lr: float = 1e-3
    lr_centers: float = 5e-3
    triplet_margin: float = 0.2

    w_rec: float = 1.0
    w_trip: float = 1.0
    w_div: float = 0.01
    w_cos: float = 0.5
    w_ent: float = 0.5

    # KMeans for GR-VLAD z-space
    kmeans_samples: int = 10000
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 3
    kmeans_max_iter: int = 300

cfg = CFG()


def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)

# =========================
# Label extraction helper (works with your loaders)
# =========================
@torch.no_grad()
def _extract_labels(y_or_meta, batch_size: int) -> torch.Tensor:
    import numpy as np, math
    if torch.is_tensor(y_or_meta):
        return y_or_meta.long()
    if isinstance(y_or_meta, dict):
        lab = y_or_meta.get("label", None)
        if lab is None:
            return torch.full((batch_size,), -1, dtype=torch.long)
        if torch.is_tensor(lab):
            return lab.long()
        if isinstance(lab, (list, tuple)):
            vals = []
            for v in lab:
                if v is None: vals.append(-1); continue
                try:
                    if isinstance(v, float) and math.isnan(v): vals.append(-1)
                    else: vals.append(int(v))
                except Exception:
                    vals.append(-1)
            return torch.tensor(vals, dtype=torch.long)
        if isinstance(lab, np.ndarray):
            return torch.from_numpy(lab.astype(np.int64))
        try:
            return torch.tensor([int(lab)]*batch_size, dtype=torch.long)
        except Exception:
            return torch.full((batch_size,), -1, dtype=torch.long)
    if isinstance(y_or_meta, (list, tuple)):
        vals = []
        for v in y_or_meta:
            try: vals.append(-1 if v is None else int(v))
            except Exception: vals.append(-1)
        return torch.tensor(vals, dtype=torch.long)
    return torch.full((batch_size,), -1, dtype=torch.long)


def safe_build_paths_from_dataset(dataset):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base, "data"):
        import os
        df = base.data; root = getattr(base, "root_dir", "")
        return [os.path.join(root, str(x)) for x in df.iloc[:, 0].astype(str).tolist()]
    if hasattr(base, "annotations"):
        import os
        df = base.annotations; root = getattr(base, "root_dir", "")
        return [os.path.join(root, str(x)) for x in df.iloc[:, 0].astype(str).tolist()]
    if hasattr(base, "imgs") and isinstance(base.imgs, list) and base.imgs and isinstance(base.imgs[0], (tuple, list)):
        return [p for (p, _) in base.imgs]
    if hasattr(base, "samples") and isinstance(base.samples, list) and base.samples and isinstance(base.samples[0], (tuple, list)):
        return [p for (p, _) in base.samples]
    raise AttributeError("Cannot infer file paths for this dataset. Expose `.data`/`.annotations` or `.imgs`/`.samples`.")

# =========================
# Viz helpers (Recall, t-SNE, galleries)
# =========================

def plot_recall_curve(recall_dict, save_path=None, title="Recall@K"):
    def _parse_k(k):
        if isinstance(k, (int, float)): return int(k)
        m = re.search(r"@(\d+)", str(k))
        return int(m.group(1)) if m else int(k)
    ks = sorted(_parse_k(k) for k in recall_dict.keys())
    ys = [100.0 * (recall_dict.get(f"R@{k}", recall_dict.get(k))) for k in ks]
    plt.figure(figsize=(6,4))
    plt.plot(ks, ys, marker="o", linewidth=2)
    plt.xticks(ks)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.ylim(0, 100)
    plt.xlabel("K"); plt.ylabel("Recall")
    plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()


def plot_tsne(desc, labels, max_points=3000, save_path=None, title="t-SNE of descriptors"):
    n = min(len(desc), max_points)
    idx = np.random.choice(len(desc), n, replace=False)
    X = desc[idx].numpy(); y = labels[idx].numpy()
    X2 = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], s=6, c=y, cmap="tab20")
    plt.title(title); plt.xticks([]); plt.yticks([]); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()


def make_retrieval_gallery(q_idx_list, sims, q_paths, db_paths, q_labels, db_labels,
                           K=5, save_dir="runs/vis", name_prefix="gallery"):
    os.makedirs(save_dir, exist_ok=True)
    for qi in q_idx_list:
        topk = torch.topk(sims[qi], K).indices.tolist()
        rows = [q_paths[qi]] + [db_paths[j] for j in topk]
        labels = [int(q_labels[qi].item())] + [int(db_labels[j].item()) for j in topk]
        images = [Image.open(p).convert("RGB") for p in rows]
        H, W = 180, 240
        thumbs = [img.resize((W, H)) for img in images]
        draw_imgs = []
        q_lab = labels[0]
        for r, (thumb, lab) in enumerate(zip(thumbs, labels)):
            canvas = thumb.copy(); d = ImageDraw.Draw(canvas)
            txt = "Q" if r == 0 else f"#{r}"
            ok = True if r == 0 else (lab == q_lab)
            color = (0,200,0) if (r == 0 or ok) else (220,0,0)
            d.rectangle([5,5,50,28], fill=(0,0,0,140))
            d.text((8,8), txt, fill=color)
            draw_imgs.append(canvas)
        strip = Image.new("RGB", ((K+1)*W, H), (255,255,255))
        for i, im in enumerate(draw_imgs): strip.paste(im, (i*W, 0))
        out_path = os.path.join(save_dir, f"{name_prefix}_q{qi}_K{K}.jpg")
        strip.save(out_path)

# =========================
# Recall@K (labels)
# =========================
@torch.no_grad()
def recall_at_k_gpu(db_desc, db_labels, q_desc, q_labels, Ks, device=None, skip_unlabeled=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    db = F.normalize(db_desc.to(device), dim=1)
    q  = F.normalize(q_desc.to(device),  dim=1)
    db_labels = db_labels.to(device).long()
    q_labels  = q_labels.to(device).long()

    maxK = max(Ks)
    B = 4096
    hits = {K: 0 for K in Ks}
    valid = 0

    for i in range(0, q.size(0), B):
        qb = q[i:i+B]
        qlb = q_labels[i:i+B]
        if skip_unlabeled:
            mask = (qlb >= 0)
            if mask.sum() == 0:
                continue
            qb  = qb[mask]
            qlb = qlb[mask]
        sims = qb @ db.T
        topk = sims.topk(maxK, dim=1).indices
        retrieved = db_labels[topk]
        qlb = qlb.unsqueeze(1)
        for K in Ks:
            hits[K] += (retrieved[:, :K] == qlb).any(dim=1).sum().item()
        valid += qb.size(0)

    denom = max(1, valid)
    return {f"R@{K}": hits[K] / denom for K in Ks}

# =====================================================
#   METHOD 1: NetVLAD (VGG16 backbone)
# =====================================================
class NetVLAD(nn.Module):
    def __init__(self, num_clusters: int, dim: int, alpha: float = 100.0, normalize_input: bool = True):
        super().__init__()
        self.K = num_clusters
        self.D = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.clusters = nn.Parameter(torch.rand(self.K, self.D))
        self.conv = nn.Conv2d(self.D, self.K, kernel_size=1, bias=True)
        self.register_buffer('centroids_init', torch.zeros_like(self.clusters))
        self.inited = False
    @torch.no_grad()
    def init_params(self, centroids: torch.Tensor):
        assert centroids.shape == (self.K, self.D)
        self.clusters.copy_(F.normalize(centroids, dim=1))
        self.centroids_init.copy_(self.clusters)
        w = 2.0 * self.alpha * self.clusters
        b = - self.alpha * (self.clusters.pow(2).sum(dim=1))
        self.conv.weight.copy_(w.view(self.K, self.D, 1, 1))
        self.conv.bias.copy_(b)
        self.inited = True
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        soft = F.softmax(self.conv(x).view(B, self.K, -1), dim=2).view(B, self.K, H, W)
        x_flat = x.view(B, C, -1)
        soft_flat = soft.view(B, self.K, -1)
        vlad = torch.zeros([B, self.K, C], dtype=x.dtype, device=x.device)
        for k in range(self.K):
            soft_k = soft_flat[:, k, :].unsqueeze(1)
            x_res = x_flat - self.clusters[k:k+1].unsqueeze(-1)
            vlad[:, k, :] = (soft_k * x_res).sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(B, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad, soft

class VGG16_Backbone(nn.Module):
    def __init__(self, requires_grad: bool = False):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(weights=None)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:31])
        if not requires_grad:
            for p in self.backbone.parameters():
                p.requires_grad = False
    def forward(self, x):
        return self.backbone(x)

class VGG16_NetVLAD(nn.Module):
    def __init__(self, K: int, alpha: float, normalize_input: bool, train_backbone: bool = False):
        super().__init__()
        self.backbone = VGG16_Backbone(requires_grad=train_backbone)
        self.netvlad = NetVLAD(num_clusters=K, dim=512, alpha=alpha, normalize_input=normalize_input)
    def init_netvlad(self, centroids: torch.Tensor):
        self.netvlad.init_params(centroids)
    def forward(self, x):
        f = self.backbone(x)
        d, a = self.netvlad(f)
        return d, a

@torch.no_grad()
def collect_local_descriptors(backbone: nn.Module, loader: DataLoader, device: str, max_desc: int) -> np.ndarray:
    backbone.eval()
    pool = []
    total = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        fmap = backbone(imgs)
        fmap = F.normalize(fmap, p=2, dim=1)
        B,C,H,W = fmap.shape
        X = fmap.permute(0,2,3,1).contiguous().view(-1, C).detach().cpu().numpy()
        pool.append(X)
        total += X.shape[0]
        if total >= max_desc:
            break
    X = np.concatenate(pool, axis=0)
    if X.shape[0] > max_desc:
        idx = np.random.choice(X.shape[0], max_desc, replace=False)
        X = X[idx]
    return X.astype(np.float32)

def init_clusters_kmeans(backbone: nn.Module, train_loader: DataLoader, device: str, K: int, max_desc: int) -> torch.Tensor:
    X = collect_local_descriptors(backbone, train_loader, device, max_desc=max_desc)
    print(f"KMeans (NetVLAD) on {len(X)} descriptors ...")
    km = KMeans(n_clusters=K, init='k-means++', n_init=3, max_iter=100, random_state=cfg.seed, verbose=0)
    km.fit(X)
    C = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
    return F.normalize(C, dim=1)

def triplet_cosine(a, p, n, margin: float):
    di = 1.0 - (a * p).sum(dim=1)
    dn = 1.0 - (a * n).sum(dim=1)
    return F.relu(margin + di - dn).mean()

def train_triplet_netvlad(model: VGG16_NetVLAD, trip_loader: DataLoader, device: str):
    model.train()
    params = [
        {"params": [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg.vlad_lr_backbone},
        {"params": [p for n,p in model.named_parameters() if "netvlad" in n], "lr": cfg.vlad_lr_head},
    ]
    opt = torch.optim.Adam(params)
    for ep in range(cfg.vlad_epochs_ft):
        t0 = time.time(); losses = []
        for anchor, pos, neg, *_ in trip_loader:
            anchor = anchor.to(device, non_blocking=True)
            pos    = pos.to(device, non_blocking=True)
            neg    = neg.to(device, non_blocking=True)
            Da,_ = model(anchor); Dp,_ = model(pos); Dn,_ = model(neg)
            loss = triplet_cosine(Da, Dp, Dn, cfg.vlad_margin)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[NetVLAD ft {ep+1}/{cfg.vlad_epochs_ft}] loss={np.mean(losses):.4f} time={time.time()-t0:.1f}s")

@torch.no_grad()
def compute_descriptors_netvlad(model: VGG16_NetVLAD, loader: DataLoader, device: str):
    model.eval()
    descs, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, y_or_meta = batch[0], batch[1]
            y = _extract_labels(y_or_meta, batch_size=imgs.size(0))
        elif isinstance(batch, (list, tuple)):
            imgs = batch[0]; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        else:
            imgs = batch; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        imgs = imgs.to(device, non_blocking=True)
        d,_ = model(imgs)
        descs.append(F.normalize(d, dim=1).cpu()); labels.append(y.cpu())
    return torch.cat(descs,0), torch.cat(labels,0)

# =====================================================
#   METHOD 2: AP-GeM (VGG16 backbone)
# =====================================================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))) if learn_p else torch.tensor(float(p))
        self.eps = eps
    def forward(self, x):
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = x.clamp(min=self.eps)
        x = x.pow(p.view(1,1,1,1))
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / p.view(1,1,1,1))
        return x.squeeze(-1).squeeze(-1)

class APGeM(nn.Module):
    def __init__(self, gem_p_init=3.0, learn_p=True, proj_dim: int = 2048, train_backbone: bool = False, normalize_input: bool = True):
        super().__init__()
        self.backbone = VGG16_Backbone(requires_grad=train_backbone)
        self.normalize_input = normalize_input
        self.gem = GeM(p=gem_p_init, learn_p=learn_p)
        self.feat_dim = 512
        self.proj = nn.Linear(self.feat_dim, proj_dim) if proj_dim and proj_dim > 0 else None
        self.out_dim = proj_dim if self.proj is not None else self.feat_dim
    def forward(self, x):
        f = self.backbone(x)
        if self.normalize_input:
            f = F.normalize(f, p=2, dim=1)
        g = self.gem(f)
        if self.proj is not None:
            g = self.proj(g)
        g = F.normalize(g, dim=1)
        return g

def fastap_loss(emb: torch.Tensor, labels: torch.Tensor, bins: int = 10, eps: float = 1e-6):
    with torch.no_grad():
        centers = torch.linspace(0, 1, steps=bins, device=emb.device)
        widths = 1.0 / bins
    S = (emb @ emb.t()).clamp(0, 1)
    B = emb.size(0)
    loss = 0.0; valid = 0
    for i in range(B):
        yi = labels[i]
        if yi < 0: continue
        pos = (labels == yi); pos[i] = False
        if pos.sum() == 0: continue
        s_i = S[i]
        def soft_hist(s_vals):
            diff = s_vals.unsqueeze(1) - centers.view(1,-1)
            w = F.relu(1.0 - diff.abs() / widths)
            return w.sum(dim=0)
        h_pos = soft_hist(s_i[pos]) + eps
        h_all = soft_hist(s_i[labels>=0]) + eps
        c_pos = torch.cumsum(h_pos, dim=0)
        c_all = torch.cumsum(h_all, dim=0)
        precision = c_pos / c_all
        ap = (precision * h_pos / h_pos.sum()).sum()
        loss += (1.0 - ap); valid += 1
    if valid == 0:
        return emb.new_tensor(0.0, requires_grad=True)
    return loss / valid

def train_apgem(model: APGeM, trip_loader: DataLoader, cls_loader: Optional[DataLoader], device: str):
    model.train()
    params = [
        {"params": [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg.apgem_lr_backbone},
        {"params": [p for n,p in model.named_parameters() if "backbone" not in n], "lr": cfg.apgem_lr_head},
    ]
    opt = torch.optim.Adam(params)
    cls_iter = iter(cls_loader) if (cfg.apgem_use_fastap and cls_loader is not None) else None
    for ep in range(cfg.apgem_epochs_ft):
        t0 = time.time(); losses = []
        for anchor, pos, neg, *_ in trip_loader:
            anchor = anchor.to(device, non_blocking=True)
            pos    = pos.to(device, non_blocking=True)
            neg    = neg.to(device, non_blocking=True)
            Da = model(anchor); Dp = model(pos); Dn = model(neg)
            L = triplet_cosine(Da, Dp, Dn, cfg.apgem_margin)
            if cfg.apgem_use_fastap and cls_iter is not None:
                try:
                    imgs, y_or_meta = next(cls_iter)
                except StopIteration:
                    cls_iter = iter(cls_loader)
                    imgs, y_or_meta = next(cls_iter)
                imgs = imgs.to(device, non_blocking=True)
                labels = _extract_labels(y_or_meta, imgs.size(0)).to(device)
                E = model(imgs)
                L = L + fastap_loss(E, labels, bins=cfg.apgem_fastap_bins)
            opt.zero_grad(); L.backward(); opt.step()
            losses.append(float(L.item()))
        print(f"[AP-GeM ft {ep+1}/{cfg.apgem_epochs_ft}] loss={np.mean(losses):.4f} time={time.time()-t0:.1f}s")

@torch.no_grad()
def compute_descriptors_apgem(model: APGeM, loader: DataLoader, device: str):
    model.eval()
    descs, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, y_or_meta = batch[0], batch[1]
            y = _extract_labels(y_or_meta, batch_size=imgs.size(0))
        elif isinstance(batch, (list, tuple)):
            imgs = batch[0]; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        else:
            imgs = batch; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        imgs = imgs.to(device, non_blocking=True)
        d = model(imgs)
        descs.append(F.normalize(d, dim=1).cpu()); labels.append(y.cpu())
    return torch.cat(descs,0), torch.cat(labels,0)

# =====================================================
#   METHOD 3: Global Residual VLAD (your model)
# =====================================================
class GeM_small(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p)) if learn_p else torch.tensor(p)
        self.eps = eps
    def forward(self, x):
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p.view(1,1,1,1))
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / p.view(1,1,1,1))
        return x.squeeze(-1).squeeze(-1)

class Encoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        ch = [32, 64, 128, 256, 512]
        self.body = nn.Sequential(
            nn.Conv2d(3, ch[0], 7, stride=2, padding=3), nn.GroupNorm(8, ch[0]), nn.ReLU(True),
            nn.Conv2d(ch[0], ch[1], 3, stride=2, padding=1), nn.GroupNorm(8, ch[1]), nn.ReLU(True),
            nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1), nn.GroupNorm(16, ch[2]), nn.ReLU(True),
            nn.Conv2d(ch[2], ch[3], 3, stride=2, padding=1), nn.GroupNorm(16, ch[3]), nn.ReLU(True),
            nn.Conv2d(ch[3], ch[4], 3, stride=2, padding=1), nn.GroupNorm(32, ch[4]), nn.ReLU(True),
        )
        self.gem = GeM_small()
        self.fc = nn.Linear(ch[4], out_dim)
    def forward(self, x):
        f = self.body(x)
        g = self.gem(f)
        z = self.fc(g)
        z = F.normalize(z, dim=1)
        return z

class Decoder(nn.Module):
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 512*8*6), nn.ReLU(True))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 3, 3, padding=1),
        )
    def forward(self, z, target_hw: Tuple[int,int]):
        B = z.size(0)
        h = self.fc(z).view(B, 512, 8, 6)
        xhat = self.up(h)
        xhat = F.interpolate(xhat, size=target_hw, mode="bilinear", align_corners=False)
        return xhat

class GlobalResidualVLAD(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, D: int, K: int, tau_init: float, concat: bool, proj_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.C = nn.Parameter(torch.randn(K, D))
        nn.init.orthogonal_(self.C)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.concat = concat
        self.proj = nn.Linear((K*D) if concat else D, proj_dim)
    def descriptor(self, z):
        z2 = (z**2).sum(dim=1, keepdim=True)
        c2 = (self.C**2).sum(dim=1)
        logits = -(z2 + c2[None,:] - 2*z@self.C.T) / torch.clamp(self.tau, min=1e-3)
        a = logits.softmax(dim=1)
        res = z[:,None,:] - self.C[None,:,:]
        r = a[...,None] * res
        R = r.reshape(z.size(0), -1) if self.concat else r.sum(dim=1)
        dvec = F.normalize(self.proj(R), dim=1)
        return dvec, a
    def forward(self, x):
        z = self.encoder(x)
        dvec, a = self.descriptor(z)
        xhat = self.decoder(z, target_hw=(x.shape[-2], x.shape[-1]))
        return dvec, xhat, z, a

# ---- GR-VLAD losses ----

def reconstruction_loss(xhat, x):
    return (xhat - x).abs().mean()

def diversity_loss(centers: torch.Tensor, assign: torch.Tensor, w_cos=0.5, w_ent=0.5):
    Cn = F.normalize(centers, dim=1)
    G  = Cn @ Cn.T
    K  = G.size(0)
    off = G - torch.eye(K, device=G.device)
    L_cos = (off**2).sum() / (K*K - K + 1e-8)
    a = assign.clamp_min(1e-8)
    L_ent = -(a * a.log()).sum(dim=1).mean()
    return w_cos*L_cos + w_ent*L_ent

@torch.no_grad()
def collect_z_samples(encoder: Encoder, loader: DataLoader, max_n: int, device: str):
    encoder.eval(); chunks = []; total = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        z = encoder(imgs).cpu().numpy(); chunks.append(z); total += z.shape[0]
        if total >= max_n: break
    Z = np.concatenate(chunks, axis=0)
    return Z[:max_n]

def init_centers_kmeans_gr(encoder: Encoder, train_loader: DataLoader, D: int, K: int, device: str) -> torch.Tensor:
    Z = collect_z_samples(encoder, train_loader, cfg.kmeans_samples, device)
    km = KMeans(n_clusters=K, init=cfg.kmeans_init, n_init=cfg.kmeans_n_init, max_iter=cfg.kmeans_max_iter, random_state=cfg.seed, verbose=0)
    km.fit(Z)
    C = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
    return F.normalize(C, dim=1)

def train_warmup_ae(model: GlobalResidualVLAD, train_loader: DataLoader, device: str):
    if not Pretrain:
        print("[GR-VLAD] Skipping AE warmup (Pretrain=False)"); return
    model.train()
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    for ep in range(cfg.epochs_warmup):
        t0 = time.time(); losses = []
        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            _, xhat, _, a = model(imgs)
            Lrec = reconstruction_loss(xhat, imgs)
            Ldiv = diversity_loss(model.C, a, w_cos=cfg.w_cos, w_ent=cfg.w_ent) * 0.2
            loss = cfg.w_rec*Lrec + cfg.w_div*Ldiv
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"[GR-VLAD warmup {ep+1}/{cfg.epochs_warmup}] loss={np.mean(losses):.4f} time={time.time()-t0:.1f}s")

def train_joint_gr(model: GlobalResidualVLAD, trip_loader: DataLoader, recon_loader: DataLoader, device: str):
    if not Global_Train:
        print("[GR-VLAD] Skipping joint train (Global_Train=False)"); return
    model.train()
    param_groups = [
        {"params": list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.proj.parameters()), "lr": cfg.lr},
        {"params": [model.C, model.tau], "lr": cfg.lr_centers},
    ]
    opt = torch.optim.Adam(param_groups)
    recon_iter = iter(recon_loader)
    def tau_schedule(t, T, t0=cfg.tau_init, t1=0.20):
        cosv = 0.5*(1+math.cos(math.pi * t / max(T,1)))
        return t1 + (t0 - t1)*cosv
    for ep in range(cfg.epochs_joint):
        t0 = time.time(); losses = []
        with torch.no_grad():
            model.tau.data = torch.clamp(torch.tensor(tau_schedule(ep, cfg.epochs_joint), device=device), 1e-3)
        for anchor, pos, neg, *_ in trip_loader:
            anchor = anchor.to(device, non_blocking=True)
            pos    = pos.to(device, non_blocking=True)
            neg    = neg.to(device, non_blocking=True)
            Da, xhat_a, _, a = model(anchor)
            Dp, _,     _, _  = model(pos)
            Dn, _,     _, _  = model(neg)
            Ltrip = triplet_cosine(Da, Dp, Dn, cfg.triplet_margin)
            Lrec  = reconstruction_loss(xhat_a, anchor)
            Ldiv  = diversity_loss(model.C, a, w_cos=cfg.w_cos, w_ent=cfg.w_ent)
            loss = cfg.w_trip*Ltrip + cfg.w_rec*Lrec + cfg.w_div*Ldiv
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            try:
                imgs, _ = next(recon_iter)
            except StopIteration:
                recon_iter = iter(recon_loader); imgs, _ = next(recon_iter)
            imgs = imgs.to(device, non_blocking=True)
            _, xhat, _, a2 = model(imgs)
            l2 = cfg.w_rec*reconstruction_loss(xhat, imgs) + (cfg.w_div*0.2)*diversity_loss(model.C, a2, cfg.w_cos, cfg.w_ent)
            opt.zero_grad(); l2.backward(); opt.step()
        print(f"[GR-VLAD joint {ep+1}/{cfg.epochs_joint}] loss={np.mean(losses):.4f} tau={float(model.tau.data):.3f} time={time.time()-t0:.1f}s")

@torch.no_grad()
def compute_desc_labels_assign_gr(model: GlobalResidualVLAD, loader: DataLoader, device: str):
    model.eval()
    descs, labels, assigns = [], [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, y_or_meta = batch[0], batch[1]
            y = _extract_labels(y_or_meta, batch_size=imgs.size(0))
        elif isinstance(batch, (list, tuple)):
            imgs = batch[0]; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        else:
            imgs = batch; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        imgs = imgs.to(device, non_blocking=True)
        d, _, _, a = model(imgs)
        descs.append(F.normalize(d, dim=1).cpu()); labels.append(y.cpu()); assigns.append(a.cpu())
    return torch.cat(descs,0), torch.cat(labels,0), torch.cat(assigns,0)

# =========================
# Runner
# =========================

def main(method: str = "netvlad"):
    device = cfg.device
    print("Device:", device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ---- Choose a dataset (RobotCar default here) ----
    train_loader, _ = get_RobotCar_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    trip_loader,  _ = get_RobotCar_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    db_loader, db_ds = get_RobotCar_Eval_db_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    q_loader, q_ds   = get_RobotCar_Eval_query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    os.makedirs("runs/vis", exist_ok=True)

    if method == "netvlad":
        # Build
        model = VGG16_NetVLAD(K=cfg.vlad_clusters, alpha=cfg.vlad_alpha, normalize_input=cfg.vlad_normalize_input, train_backbone=False).to(device)
        # Init clusters
        print("==> KMeans init clusters (NetVLAD)")
        with torch.no_grad():
            C0 = init_clusters_kmeans(model.backbone, train_loader, device, K=cfg.vlad_clusters, max_desc=cfg.vlad_kmeans_samples)
            model.init_netvlad(C0)
        # Train
        print("==> Fine-tune NetVLAD on triplets")
        train_triplet_netvlad(model, trip_loader, device)
        # Eval
        db_desc, db_lbls = compute_descriptors_netvlad(model, db_loader, device)
        q_desc,  q_lbls  = compute_descriptors_netvlad(model, q_loader,  device)
        sims = F.normalize(q_desc, dim=1) @ F.normalize(db_desc, dim=1).T
        R = recall_at_k_gpu(db_desc, db_lbls, q_desc, q_lbls, Ks=cfg.recall_ks)
        print("==== Recall (NetVLAD, labels) ====")
        for k in cfg.recall_ks: print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")
        plot_recall_curve(R, save_path="runs/vis/netvlad_recall_curve_labels.png", title="VGG16+NetVLAD Recall@K")
        # t-SNE + gallery
        plot_tsne(db_desc, db_lbls, max_points=3000, save_path="runs/vis/netvlad_tsne_db.png", title="t-SNE (VGG16+NetVLAD DB)")
        q_paths = safe_build_paths_from_dataset(q_ds)
        db_paths = safe_build_paths_from_dataset(db_ds)
        picked = random.sample(range(len(q_paths)), k=min(8, len(q_paths)))
        make_retrieval_gallery(picked, sims, q_paths, db_paths, q_lbls, db_lbls, K=5, save_dir="runs/vis", name_prefix="netvlad_topk")

    elif method == "apgem":
        # Build
        model = APGeM(gem_p_init=cfg.gem_p_init, learn_p=cfg.gem_learn_p, proj_dim=cfg.apgem_proj_dim, train_backbone=False, normalize_input=cfg.apgem_normalize_input).to(device)
        # Train
        print("==> Fine-tune AP-GeM on triplets" + (" + FastAP" if cfg.apgem_use_fastap else ""))
        cls_loader = train_loader if cfg.apgem_use_fastap else None
        train_apgem(model, trip_loader, cls_loader, device)
        # Eval
        db_desc, db_lbls = compute_descriptors_apgem(model, db_loader, device)
        q_desc,  q_lbls  = compute_descriptors_apgem(model, q_loader,  device)
        sims = F.normalize(q_desc, dim=1) @ F.normalize(db_desc, dim=1).T
        R = recall_at_k_gpu(db_desc, db_lbls, q_desc, q_lbls, Ks=cfg.recall_ks)
        print("==== Recall (AP-GeM, labels) ====")
        for k in cfg.recall_ks: print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")
        plot_recall_curve(R, save_path="runs/vis/apgem_recall_curve_labels.png", title="VGG16+AP-GeM Recall@K")
        # t-SNE + gallery
        plot_tsne(db_desc, db_lbls, max_points=3000, save_path="runs/vis/apgem_tsne_db.png", title="t-SNE (VGG16+AP-GeM DB)")
        q_paths = safe_build_paths_from_dataset(q_ds)
        db_paths = safe_build_paths_from_dataset(db_ds)
        picked = random.sample(range(len(q_paths)), k=min(8, len(q_paths)))
        make_retrieval_gallery(picked, sims, q_paths, db_paths, q_lbls, db_lbls, K=5, save_dir="runs/vis", name_prefix="apgem_topk")

    elif method == "grvlad":
        # Build
        enc = Encoder(out_dim=cfg.latent_dim)
        dec = Decoder(in_dim=cfg.latent_dim)
        model = GlobalResidualVLAD(enc, dec, D=cfg.latent_dim, K=cfg.K, tau_init=cfg.tau_init, concat=cfg.concat, proj_dim=cfg.proj_dim).to(device)
        model = model.to(memory_format=torch.channels_last)
        # AE warmup (optional)
        print("==> GR-VLAD Warmup AE")
        train_warmup_ae(model, train_loader, device)
        # KMeans init in z-space
        print("==> GR-VLAD KMeans init centers (z-space)")
        with torch.no_grad():
            C0 = init_centers_kmeans_gr(model.encoder, train_loader, D=cfg.latent_dim, K=cfg.K, device=device)
            model.C.data.copy_(C0)
        # Joint training
        print("==> GR-VLAD Joint training (triplet + recon + diversity)")
        train_joint_gr(model, trip_loader, train_loader, device)
        # Eval
        db_desc, db_lbls, _ = compute_desc_labels_assign_gr(model, db_loader, device)
        q_desc,  q_lbls,  _ = compute_desc_labels_assign_gr(model, q_loader,  device)
        sims = F.normalize(q_desc, dim=1) @ F.normalize(db_desc, dim=1).T
        R = recall_at_k_gpu(db_desc, db_lbls, q_desc, q_lbls, Ks=cfg.recall_ks)
        print("==== Recall (GR-VLAD, labels) ====")
        for k in cfg.recall_ks: print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")
        plot_recall_curve(R, save_path="runs/vis/grvlad_recall_curve_labels.png", title="GlobalResidualVLAD Recall@K")
        # t-SNE + gallery
        plot_tsne(db_desc, db_lbls, max_points=3000, save_path="runs/vis/grvlad_tsne_db.png", title="t-SNE (GR-VLAD DB)")
        q_paths = safe_build_paths_from_dataset(q_ds)
        db_paths = safe_build_paths_from_dataset(db_ds)
        picked = random.sample(range(len(q_paths)), k=min(8, len(q_paths)))
        make_retrieval_gallery(picked, sims, q_paths, db_paths, q_lbls, db_lbls, K=5, save_dir="runs/vis", name_prefix="grvlad_topk")

    else:
        raise ValueError("Unknown method. Use one of: netvlad | apgem | grvlad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified VPR comparison: NetVLAD / AP-GeM / GlobalResidualVLAD")
    parser.add_argument('--method', type=str, default=cfg.method, choices=['netvlad','apgem','grvlad'], help='Which method to run')
    args = parser.parse_args()
    cfg.method = args.method
    main(method=cfg.method)

