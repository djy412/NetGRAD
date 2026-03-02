# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 19:49:19 2025
AP-GeM Implementation for Comparison
@author: djy41
"""
import os, time, math, random, re
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageDraw
from sklearn.neighbors import BallTree  
from typing import Tuple, Dict, List, Optional
import copy
import psutil
import collections
from collections import deque
from sklearn.metrics import precision_recall_curve, average_precision_score

# =========================
# Your existing dataloaders (unchanged)
# =========================
from dataloading import (
    get_Pittsburg30k_Train_dataloaders,
    get_Pittsburg30k_Triplet_Train_dataloaders,
    get_Pittsburg30k_Eval_db_dataloaders,
    get_Pittsburg30k_Eval_query_dataloaders,
    get_Pittsburgh30k_Eval_db_dataloaders_5m,
    get_Pittsburgh30k_Eval_query_dataloaders_5m,
    get_Tokyo_Query_dataloaders,
    get_Tokyo_Eval_dataloaders,
    get_Tokyo_Triplet_Test_dataloaders,
    get_Tokyo_Triplet_Train_dataloaders,
    get_Tokyo_Train_dataloaders,
    get_RobotCar_Train_dataloaders,
    get_RobotCar_Triplet_Train_dataloaders,
    get_RobotCar_Eval_db_dataloaders,
    get_RobotCar_Eval_query_dataloaders,
    get_RobotCar_Seq_Eval_db_dataloaders,
    get_RobotCar_25_Train_dataloaders,
    get_RobotCar_25_Triplet_Train_dataloaders,
    get_RobotCar_25_Eval_db_dataloaders,
    get_RobotCar_25_Eval_query_dataloaders)

Pretrain = False
Global_Train = False
EVAL_Live_Stream = True
# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # AP-GeM settings
    gem_p_init: float = 3.0       # initial GeM exponent
    learn_p: bool = True          # learn p during fine-tuning
    feat_dim: int = 512           # VGG-16 conv5_3 channels
    proj_dim: int = 2048          # final embedding dim (after FC); set None to use raw GeM (512)
    normalize_input: bool = True

    # Training
    batch_size: int = 18
    num_workers: int = 8
    epochs_ft: int = 8
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    triplet_margin: float = 0.1

    # Optional listwise ranking loss (FastAP-style)
    use_fastap: bool = False      # set True if your loader yields (imgs, labels) classification-style batches
    fastap_bins: int = 10         # histogram bins for FastAP

    # Eval
    recall_ks: Tuple[int, ...] = (1,2,3,4,5,10,15,20,25)

cfg = CFG()

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)

# =========================
# Helpers reused (labels, viz, etc.)
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
# GeM pooling & AP-GeM head
# =========================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))) if learn_p else torch.tensor(float(p))
        self.eps = eps
    def forward(self, x):
        # x: [B,C,H,W]
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = x.clamp(min=self.eps)
        x = x.pow(p.view(1,1,1,1))
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / p.view(1,1,1,1))
        return x.squeeze(-1).squeeze(-1)  # [B,C]

# class VGG16_Backbone(nn.Module):
#     def __init__(self, requires_grad: bool = False):
#         super().__init__()
#         try:
#             from torchvision.models import vgg16, VGG16_Weights
#             vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
#         except Exception:
#             from torchvision.models import vgg16
#             vgg = vgg16(weights=None)
#         features = list(vgg.features.children())
#         self.backbone = nn.Sequential(*features[:31])  # up to conv5_3
#         if not requires_grad:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#     def forward(self, x):
#         return self.backbone(x)  # [B,512,h,w]
# replace your VGG16_Backbone with this version
class VGG16_Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, requires_grad: bool = True):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        else:
            vgg = vgg16(weights=None)
            # Kaiming init for convs when training from scratch
            for m in vgg.features.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:31])  # up to conv5_3
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad
    def forward(self, x):
        return self.backbone(x)


class APGeM(nn.Module):
    """VGG16 + GeM + (optional) projection head -> L2-normalized descriptor.
    Training: triplet by default; optional FastAP listwise if cfg.use_fastap=True and labels available.
    """
    def __init__(self, gem_p_init=3.0, learn_p=True, proj_dim=2048,
                 train_backbone=False, normalize_input=True, pretrained_backbone=True):
        super().__init__()
        self.backbone = VGG16_Backbone(pretrained=pretrained_backbone,
                                       requires_grad=train_backbone)
        self.normalize_input = normalize_input
        self.gem = GeM(p=gem_p_init, learn_p=learn_p)
        self.feat_dim = 512
        self.proj = None
        if proj_dim and proj_dim > 0:
            self.proj = nn.Linear(self.feat_dim, proj_dim)
            self.out_dim = proj_dim
        else:
            self.out_dim = self.feat_dim

    def forward(self, x):
        feat = self.backbone(x)                 # [B,512,h,w]
        if self.normalize_input:
            feat = F.normalize(feat, p=2, dim=1)
        g = self.gem(feat)                      # [B,512]
        if self.proj is not None:
            g = self.proj(g)
        g = F.normalize(g, dim=1)               # final descriptor
        return g


# =========================
# Losses: Triplet and optional FastAP (approx AP)
# =========================

def triplet_cosine(a, p, n, margin: float):
    di = 1.0 - (a * p).sum(dim=1)
    dn = 1.0 - (a * n).sum(dim=1)
    return F.relu(margin + di - dn).mean()


def fastap_loss(emb: torch.Tensor, labels: torch.Tensor, bins: int = 10, eps: float = 1e-6):
    """A simple FastAP-style listwise loss.
    - emb: [B,D] L2 descriptors
    - labels: [B] int labels (>=0 defines positives)
    Reference idea: Cakir et al., "FastAP" (CVPR'19). This is a compact implementation
    sufficient for fine-tuning; for heavy use, consider a dedicated, numerically-stable impl.
    """
    with torch.no_grad():
        # Similarity histogram bin centers in [0, 1]
        centers = torch.linspace(0, 1, steps=bins, device=emb.device)
        widths = 1.0 / bins
    S = (emb @ emb.t()).clamp(0, 1)  # cosine sim (since L2-normalized)
    B = emb.size(0)
    loss = 0.0; valid = 0
    for i in range(B):
        yi = labels[i]
        if yi < 0:
            continue
        pos = (labels == yi)
        neg = (labels != yi) & (labels >= 0)
        pos[i] = False
        P = pos.nonzero(as_tuple=False).view(-1)
        N = neg.nonzero(as_tuple=False).view(-1)
        if P.numel() == 0:
            continue
        s_i = S[i]  # [B]
        # soft histograms
        def soft_hist(s_vals):
            # triangular kernel to centers
            # weight_ij = relu(1 - |s- c_j|/w)
            diff = s_vals.unsqueeze(1) - centers.view(1, -1)
            w = F.relu(1.0 - diff.abs() / widths)
            return w.sum(dim=0)  # [bins]
        h_pos = soft_hist(s_i[P]) + eps
        h_all = soft_hist(s_i[labels>=0]) + eps
        # cumulative sums across similarity (low->high)
        c_pos = torch.cumsum(h_pos, dim=0)
        c_all = torch.cumsum(h_all, dim=0)
        precision = c_pos / c_all
        # AP as sum over bins of precision * pos density at that bin
        ap = (precision * h_pos / h_pos.sum()).sum()
        loss += (1.0 - ap)
        valid += 1
    if valid == 0:
        return emb.new_tensor(0.0, requires_grad=True)
    return loss / valid

# =========================
# Training
# =========================

def train_apgem(model: APGeM, trip_loader: DataLoader, cls_loader: DataLoader, device: str):
    model.train()
    params = [
        {"params": [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg.lr_backbone},
        {"params": [p for n,p in model.named_parameters() if "backbone" not in n], "lr": cfg.lr_head},
    ]
    opt = torch.optim.Adam(params)

    if cfg.use_fastap and cls_loader is None:
        raise ValueError("cfg.use_fastap=True requires a classification-style loader providing (imgs, labels)")

    cls_iter = iter(cls_loader) if cfg.use_fastap else None

    for ep in range(cfg.epochs_ft):
        t0 = time.time(); losses = []
        for anchor, pos, neg, *_ in trip_loader:
            anchor = anchor.to(device, non_blocking=True)
            pos    = pos.to(device, non_blocking=True)
            neg    = neg.to(device, non_blocking=True)

            Da = model(anchor)
            Dp = model(pos)
            Dn = model(neg)

            L = triplet_cosine(Da, Dp, Dn, cfg.triplet_margin)

            # Optionally mix in a listwise step (FastAP) using a labeled batch
            if cfg.use_fastap:
                try:
                    imgs, y_or_meta = next(cls_iter)
                except StopIteration:
                    cls_iter = iter(cls_loader)
                    imgs, y_or_meta = next(cls_iter)
                imgs = imgs.to(device, non_blocking=True)
                labels = _extract_labels(y_or_meta, imgs.size(0)).to(device)
                E = model(imgs)
                L = L + fastap_loss(E, labels, bins=cfg.fastap_bins)

            opt.zero_grad(); L.backward(); opt.step()
            losses.append(float(L.item()))
        print(f"[AP-GeM fine-tune {ep+1}/{cfg.epochs_ft}] loss={np.mean(losses):.4f} time={time.time()-t0:.1f}s")
    torch.save(model.state_dict(), "weights/AP-GeM_Model.pth")
    print("Model saved ✅")
# =========================
# Inference & evaluation
# =========================
@torch.no_grad()
def compute_descriptors(model: APGeM, loader: DataLoader, device: str):
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
        descs.append(F.normalize(d, dim=1).cpu())
        labels.append(y.cpu())
    return torch.cat(descs, 0), torch.cat(labels, 0)

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

# ---------- Viz helpers ----------
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


EARTH_R = 6371008.8  # meters

def build_positive_sets_by_radius(db_ll_deg: np.ndarray,
                                  q_ll_deg: np.ndarray,
                                  radius_m: float = 25.0) -> List[set]:
    """
    Returns list of sets; pos_sets[i] contains DB indices that are within radius_m of query i.
    """
    assert db_ll_deg.shape[1] == 2 and q_ll_deg.shape[1] == 2
    db_rad = np.radians(db_ll_deg)
    q_rad  = np.radians(q_ll_deg)
    tree = BallTree(db_rad, metric='haversine')
    neigh = tree.query_radius(q_rad, r=radius_m / EARTH_R)   # list of arrays
    return [set(arr.tolist()) for arr in neigh]

@torch.no_grad()
def recall_at_k_radius(sims: torch.Tensor,
                       pos_sets: List[set],
                       Ks: Tuple[int, ...],
                       skip_no_pos: bool = True) -> Dict[str, float]:
    """
    sims: [Nq, Nd] cosine similarity matrix (higher is better)
    pos_sets: list of sets of DB indices that are positives for each query
    """
    maxK = max(Ks)
    topk = torch.topk(sims, k=maxK, dim=1).indices.cpu().numpy()  # [Nq, maxK]
    hits = np.zeros(len(Ks), dtype=np.int64)
    valid = 0
    for i, gt in enumerate(pos_sets):
        if skip_no_pos and len(gt) == 0:
            continue
        valid += 1
        for t, K in enumerate(Ks):
            if any(j in gt for j in topk[i, :K]):
                hits[t] += 1
    denom = max(1, valid)  # avoid div/0
    return {f"R@{k}": hits[t] / denom for t, k in enumerate(Ks)}



def to_numpy_image(t):
    t = t.detach().cpu()
    if t.dim() == 3 and t.size(0) in (1,3):
        t = t.permute(1,2,0)
    t = t.clamp(0,1)
    return (t*255.0).round().to(torch.uint8).numpy()
#----------------------------------------------------------------------------

# --- make memory accept optional idx (works for viz + mining) ---
class OnlinePlaceMemory:
    def __init__(self, dim, tau_assign=0.7, ema=0.1, max_per_cluster=None):
        self.dim = dim
        self.tau = tau_assign
        self.ema = ema
        self.max_per_cluster = max_per_cluster  # None = unbounded
        self.centers = []   # list[Tensor[D]] unit-norm, CPU
        self.items = []     # list[dict]: {'emb': Tensor[D], 'cid': int, 'idx': int}

    @torch.no_grad()
    def nearest_center(self, z_cpu: torch.Tensor):
        """Return (center_id, similarity)."""
        if not self.centers:
            return -1, 0.0
        C = torch.stack(self.centers)      # [C, D]
        sims = C @ z_cpu                   # [C]
        s, j = sims.max(dim=0)
        return int(j.item()), float(s.item())

    @torch.no_grad()
    def update_center(self, cid: int, z_cpu: torch.Tensor, lr: float | None = None):
        """Exponetial Moving Average update of a center without storing a new item."""
        z_cpu = F.normalize(z_cpu.detach().cpu(), p=2, dim=0)
        alpha = self.ema if lr is None else lr
        c = self.centers[cid]
        self.centers[cid] = F.normalize((1 - alpha) * c + alpha * z_cpu, p=2, dim=0)

    @torch.no_grad()
    def add(self, z_cpu: torch.Tensor, idx: int | None = None,
            force_new: bool = False, update_center: bool = True):
        """
        Add an embedding; returns (cid, is_new, s_center).
        - force_new=True    → always start a new cluster (used for red NEW PLACE).
        - update_center=False → attach without EMA update (rare, but sometimes handy).
        """
        z_cpu = F.normalize(z_cpu.detach().cpu(), p=2, dim=0)
        j, s = self.nearest_center(z_cpu)

        is_new = force_new or (j < 0) or (s < self.tau)
        if is_new:
            cid = len(self.centers)
            self.centers.append(z_cpu.clone())
        else:
            cid = j
            if update_center:
                c = self.centers[j]
                c = F.normalize((1 - self.ema) * c + self.ema * z_cpu, p=2, dim=0)
                self.centers[j] = c

        self.items.append({'emb': z_cpu, 'cid': cid, 'idx': idx})

        # optional per-cluster cap (FIFO)
        if self.max_per_cluster is not None:
            count = 0
            first_idx = None
            for k, it in enumerate(self.items):
                if it['cid'] == cid:
                    count += 1
                    if first_idx is None:
                        first_idx = k
            if count > self.max_per_cluster and first_idx is not None:
                del self.items[first_idx]

        return cid, is_new, s

    @torch.no_grad()
    def members(self, cid: int):
        """Return list of items (dicts) in cluster cid."""
        return [it for it in self.items if it['cid'] == cid]

    @torch.no_grad()
    def best_member(self, z_cpu: torch.Tensor, cid: int, top_m: int = 20):
        """
        Among newest top_m members of cluster cid, return:
          (best_item_dict, s_best, s_second)
        If cluster is empty, returns (None, 0.0, 0.0).
        """
        ms = self.members(cid)
        if not ms:
            return None, 0.0, 0.0
        ms_slice = ms[-top_m:]
        Zm = torch.stack([it['emb'] for it in ms_slice])    # [M, D]
        sims = Zm @ z_cpu                                   # [M]
        s_sorted, order = torch.sort(sims, descending=True)
        s_best = float(s_sorted[0])
        s_second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0
        best_item = ms_slice[int(order[0].item())]
        return best_item, s_best, s_second

# --- your pick_triplet from earlier (adapt to this memory) ---
@torch.no_grad()
def pick_triplet(z_t, i, mem: OnlinePlaceMemory, prev_z=None, Kneg=12, tau_new=0.6):
    if len(mem.items) == 0:
        return None
    j, s = mem.nearest_center(z_t)
    is_new = (s < tau_new or j < 0)

    # positive
    if is_new:
        if prev_z is not None:
            z_pos = prev_z
            pos_cids = []    # unknown cluster yet
        else:
            # fallback: nearest in memory
            Z = torch.stack([it['emb'] for it in mem.items])
            z_pos = Z[torch.argmax(Z @ z_t)]
            pos_cids = []
    else:
        members = [it for it in mem.items if it['cid'] == j]
        if members:
            Zm = torch.stack([it['emb'] for it in members])
            z_pos = Zm[torch.argmax(Zm @ z_t)]
            pos_cids = [j]
        else:
            Z = torch.stack([it['emb'] for it in mem.items])
            z_pos = Z[torch.argmax(Z @ z_t)]
            pos_cids = []

    # negatives
    pool = [it for it in mem.items if it['cid'] not in set(pos_cids)]
    if not pool:
        return None
    sel = torch.randperm(len(pool))[:min(Kneg, len(pool))].tolist()
    Zneg = torch.stack([pool[k]['emb'] for k in sel])
    return z_pos, Zneg
#----------------------------------------------------------------------------

def show_pair(current_np, right_np=None, left_title="", right_title=""):
    """Show current frame (left) and either revisit (right) or a blank panel."""
    h, w = current_np.shape[:2]
    blank = np.full((h, w, 3), 255, dtype=np.uint8)  # white
    right = right_np if right_np is not None else blank

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(current_np); axes[0].axis('off'); axes[0].set_title(left_title)
    axes[1].imshow(right);      axes[1].axis('off'); axes[1].set_title(right_title or ("REVISIT" if right_np is not None else "—"))
    plt.tight_layout(); plt.show()
#----------------------------------------------------------------------------

def get_image_by_index(dataset, idx):
    """Fetch raw image (post-transform tensor) from the same dataset used by the loader."""
    img, _, _ = dataset[idx]
    return to_numpy_image(img)  # uses your existing helper
#----------------------------------------------------------------------------
#************************************************************************
#--- Define Triplet Loss
#************************************************************************
def triplet_loss(anchor, positive, negatives):
    # anchor: [D], positive: [D], negatives: [K,D]; all unit-norm
    # cosine distance: d = 1 - cos
    ap = 1.0 - (anchor @ positive)          # scalar
    an = 1.0 - (anchor @ negatives.t())     # [K]
    # semi-hard: only negatives closer than margin boundary
    losses = torch.relu(ap - an + cfg.triplet_margin)   # [K]
    
    return losses.mean()
#----------------------------------------------------------------------------

# keep this for inference/memory
def embed_cpu(model, x):
    with torch.no_grad():
        dvec = model(x)  # (B, K*D)
        dvec = F.normalize(dvec, p=2, dim=1)
    return dvec.squeeze(0).detach().cpu()  # [K*D] on CPU

# new: for training updates (no no_grad, stays on device)
def embed_train(model, x):
    dvec = model(x)  # (B, K*D)
    dvec = F.normalize(dvec, p=2, dim=1)
    return dvec.squeeze(0)  # [K*D] on device (requires_grad=True)


# --- helper(s) --------------------------------------------------------------  # NEW
def get_place_id_by_index(dataset, idx):
    """
    Return integer place_id for a dataset index. Assumes dataset[idx] -> (img, place_id, revisit).
    Safely unwrap Subset and handle tensors/ints.
    """
    try:
        base = getattr(dataset, "dataset", dataset)  # unwrap torch.utils.data.Subset if present
        sample = base[idx]
        pid = sample[1]  # place_id
        # handle tensor, numpy, or plain int
        if hasattr(pid, "item"):
            pid = pid.item()
        elif isinstance(pid, (list, tuple)) and pid:
            pid = pid[0]
        return int(pid)
    except Exception:
        return None

def majority_place_id_from_members(dataset, members_slice):
    """
    Majority vote of place_id over a list of memory members (each has an 'idx').
    """
    counts = {}
    for it in members_slice or []:
        idx_m = it.get('idx', None)
        if idx_m is None:
            continue
        pid_m = get_place_id_by_index(dataset, idx_m)
        if pid_m is None:
            continue
        counts[pid_m] = counts.get(pid_m, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)



# =========================
# Main
# =========================
if __name__ == "__main__":
    device = cfg.device
    print("Device:", device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    train_loader, _ = get_RobotCar_25_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    trip_loader,  _ = get_RobotCar_25_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    db_loader, db_ds = get_RobotCar_25_Eval_db_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    q_loader, q_ds   = get_RobotCar_25_Eval_query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)    

    # train_loader, _ = get_RobotCar_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # trip_loader,  _ = get_RobotCar_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_RobotCar_Eval_db_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # q_loader, q_ds   = get_RobotCar_Eval_query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # train_loader, _ = get_Tokyo_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # trip_loader,  _ = get_Tokyo_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_Tokyo_Eval_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # q_loader, q_ds   = get_Tokyo_Query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # train_loader, _ = get_Pittsburg30k_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # trip_loader,  _ = get_Pittsburg30k_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_Pittsburgh30k_Eval_db_dataloaders_5m(batch_size=cfg.batch_size, num_workers=cfg.num_workers,return_meta=False)
    # q_loader, q_ds = get_Pittsburgh30k_Eval_query_dataloaders_5m(batch_size=cfg.batch_size, num_workers=cfg.num_workers,return_meta=False)
    # # Grab coords as NumPy for BallTree positives (distance-based GT)
    # db_ll = db_ds.coords_deg_np()   # shape [N_db, 2] or None
    # q_ll  = q_ds.coords_deg_np()    # shape [N_q, 2] or None
    # assert db_ll is not None and q_ll is not None, "CSV must include lat/lon columns"
    # # ---- Build model ----
    # model = APGeM(gem_p_init=cfg.gem_p_init, learn_p=cfg.learn_p, proj_dim=cfg.proj_dim,
    #               train_backbone=False, normalize_input=cfg.normalize_input).to(device)
    
    model = APGeM(
        gem_p_init=cfg.gem_p_init, learn_p=cfg.learn_p, proj_dim=cfg.proj_dim,
        train_backbone=True, normalize_input=cfg.normalize_input,
        pretrained_backbone=False
    ).to(device)
    cfg.apgem_lr_backbone = 1e-3
    cfg.apgem_lr_head     = 1e-3
    cfg.apgem_epochs_ft   = 30

    # ---- Fine-tune ----
    if Global_Train:
        print("==> Fine-tune AP-GeM on triplets" + (" + FastAP" if cfg.use_fastap else ""))
        cls_loader = train_loader if cfg.use_fastap else None
        train_apgem(model, trip_loader, cls_loader, device)
    else: #--- Load models weights
        print("Loading Model wieghts from Previous Training")  
        load_model_path = 'C:/Users/djy41/Desktop/PhD Work/Code/C_2) Visual Place Recognition/weights/AP-GeM_Model.pth'    
        model.load_state_dict(torch.load(load_model_path)) 

    # ---- Evaluation ----
    # print("==> Compute descriptors (DB & Query)")
    # db_desc, db_lbls = compute_descriptors(model, db_loader, device)
    # q_desc,  q_lbls  = compute_descriptors(model, q_loader,  device)

    # db_norm = F.normalize(db_desc, dim=1)
    # q_norm  = F.normalize(q_desc,  dim=1)
    # sims = q_norm @ db_norm.T

    # R = recall_at_k_gpu(db_desc, db_lbls, q_desc, q_lbls, Ks=cfg.recall_ks)
    # print("==== Recall (labels) ====")
    # for k in cfg.recall_ks:
    #     print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")
    # os.makedirs("runs/vis", exist_ok=True)
    
    # # # --- Distance-based recall ---    
    # # # Build positives by 25 m radius (Option A protocol)
    # # POS_RADIUS_M = 25.0
    # # pos_sets = build_positive_sets_by_radius(db_ll, q_ll, radius_m=POS_RADIUS_M)
    # # R = recall_at_k_radius(sims, pos_sets, cfg.recall_ks, skip_no_pos=True)
    # # print(f"Distance-based GT: {POS_RADIUS_M} m | Queries with ≥1 positive: "
    # #       f"{sum(1 for s in pos_sets if len(s)>0)}/{len(pos_sets)}")
    # # print("==== Recall (25 m) ====")
    # # for k in cfg.recall_ks:
    # #     print(f"R@{k}: {R[f'R@{k}']*100:.2f}%")   
    
    # # 1) Recall@K plot
    # plot_recall_curve(R, save_path="runs/vis/apgem_recall_curve_labels.png", title="VGG16+AP-GeM Recall@K")

    # # t-SNE & galleries (optional)
    # plot_tsne(db_desc, db_lbls, max_points=3000, save_path="runs/vis/apgem_tsne_db.png",
    #           title="t-SNE (VGG16+AP-GeM DB descriptors)")

    # q_paths = safe_build_paths_from_dataset(q_ds)
    # db_paths = safe_build_paths_from_dataset(db_ds)

    # picked = random.sample(range(len(q_paths)), k=min(8, len(q_paths)))
    # make_retrieval_gallery(picked, sims, q_paths, db_paths, q_lbls, db_lbls,
    #                        K=5, save_dir="runs/vis", name_prefix="apgem_topk")
    # print("Saved AP-GeM galleries and plots to runs/vis/")






    if EVAL_Live_Stream:
        teacher = copy.deepcopy(model).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        
        """---------------------------------------------------------------------------------------"""
        """---    Run the Frozen/Unfrozen model on a "live stream" of images and detect loop closure   ----"""               
        # Create process handle once outside loop
        process = psutil.Process(os.getpid())
        memory_usage = []   # store memory in MB
        memory_len = []     # store number of memory items
        
        #--- Thresholds 
        #Typical range 
        TAU_SAME = 0.91      #--- If last frame & current frame sim is greater than this, then in same place 
        #Typical range 0.75-89
        TAU_CENTER = 0.88    #--- How close the current frame needs to be to another cluster center to be considered for revisit
        TAU_MEMBER = 0.9     #--- Min similarity the current frame needs to be to be a revisit 
        RATIO_MIN = 1.05     #--- The ratio that the best similarity has to be better than the second best similarity
        TOP_M_CHECK = 30   
        EXCLUDE_LAST = 10 
        FRAME_RATE = 20
        UPDATE_EVERY = 10        # update 1 out of N sampled frames
        MIN_POS = 0.85           # require a strong positive
        MIN_NEG_SIM = 0.45       # ensure at least one negative is reasonably similar
        SHOW_EVERY = 10          # visualize every 10th sampled frame
        
        # exclude last N clusters from "revisit" eligibility 
        model.eval() 
        # memory init: use proj_dim
        memory = OnlinePlaceMemory(dim=cfg.proj_dim, tau_assign=TAU_CENTER, ema=0.1)
        
        params = list(model.proj.parameters())  # online adapt only head
        optimizer = torch.optim.Adam(params, lr=2e-5)  # << smaller LR
        CLIP_NORM = 0.75
    
        Kneg = 30 
        tau_new = 0.75  
        recent_cids = collections.deque(maxlen=EXCLUDE_LAST) 
        prev_emb = None 
        MIN_MEM_FOR_MINING = max(2*Kneg, 30)
        
        plt.figure("TAU Settings")
        plt.title("Tau Settings") 
        plt.text(0.1, 0.8, 'Tau Same: %.3f' %(TAU_SAME))
        plt.text(0.1, 0.7, 'Tau Center: %.3f' %(TAU_CENTER))
        plt.text(0.1, 0.6, 'Tau Member: %.3f' %(TAU_MEMBER))
        plt.text(0.1, 0.5, 'Min Ratio 1st, 2nd: %.3f' %(RATIO_MIN))
        plt.text(0.1, 0.4, 'Number Checked: %.1f' %(TOP_M_CHECK))
        plt.text(0.1, 0.3, 'Number last frames not to look at: %.1f' %(EXCLUDE_LAST))
        plt.text(0.1, 0.2, 'Triplet Margin: %.3f' %(cfg.triplet_margin))
        plt.text(0.1, 0.1, 'Frame Rate: %.1f' %(FRAME_RATE))
        plt.axis('off')
        
        #Seq_eval_loader, dataset = get_Newer_College_Seq_Eval_dataloaders(batch_size=1, num_workers=cfg.num_workers) 
        Seq_eval_loader, dataset = get_RobotCar_Seq_Eval_db_dataloaders(batch_size=1, num_workers=cfg.num_workers) 
           
        TP = FP = TN = FN = 0
        all_scores = []
        all_labels = []
        #--- Accounting for identifying a revisit and same place logic
        revisit_active = False          # Are we currently in a revisit episode?
        revisit_active_cid = None       # Which place (cid) is being revisited?
        # Optional: time-based latch (helps with brief drops)
        REVISIT_LATCH_FRAMES = int(1.5 * FRAME_RATE)  # e.g., 1.5 seconds
        latch_until = -1

        # ---- Episode state (keep it simple) ----
        revisit_active = False
        revisit_active_cid = None
        last_cid = None

        members_slice = None  # <-- set a default so it's always defined
            
        for i, (img, place_id, revisit) in enumerate(Seq_eval_loader):
            if i % FRAME_RATE != 0:
                continue
        
            sim_prev = 0.0           # so we can always use it in scoring
            cid = None               # will be set below per state
            best_member = None        # <-- reset each loop
            s_best = 0.0
            j_candidate = None        # track candidate cid for revisit UI text
            state = "new_place"
            j = None
            s_center = -1.0
    
            img = img.to(device)
            # ---- descriptors for mining/state must come from the FROZEN teacher ----
            with torch.no_grad():
                z_t_teacher = embed_cpu(teacher, img)  # CPU 1D
            # use z_t_teacher for memory.nearest_center, best member, state machine, prev_emb, scoring, etc.
               
    
            #--- Stage 0: Check if current frame is same-as-previous
            if prev_emb is not None:
                sim_prev = float((z_t_teacher @ prev_emb).item()) #--- dot prod/cosine similarity
                if sim_prev >= TAU_SAME:
                    state = "same_spot"
        
            # Stage 1+2: Check if we are in a new place or are revisiting
            if state != "same_spot":
                j, s_center = memory.nearest_center(z_t_teacher)
                #---similarity to centers and z_t has to be greater than tau_center
                if j >= 0 and (j not in recent_cids) and s_center >= TAU_CENTER: 
                    members = [it for it in memory.items if it['cid'] == j]
                    if members:
                        members_slice = members[-TOP_M_CHECK:]
                        Zm = torch.stack([it['emb'] for it in members_slice])
                        sims = (Zm @ z_t_teacher) #--- Check similiarity between current frame and past centers
                        s_sorted, order = torch.sort(sims, descending=True)
                        s_best = float(s_sorted[0]) #--- Store the best similarity
                        s_second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0
                        ratio_ok = (len(s_sorted) == 1) or (s_best >= RATIO_MIN * max(s_second, 1e-6))
                        #ratio_ok = (s_second <= 1e-6) or (s_best / max(s_second, 1e-6) >= RATIO_MIN)
                        member_ok = s_best >= TAU_MEMBER
                        if member_ok and ratio_ok:
                            state = "revisit"
                            j_candidate = j
                            best_local = int(order[0].item())
                            best_member = members_slice[best_local]
        
            # --- infer predicted place_id for this detection ------------------------  # NEW
            gt_place_id = int(place_id.item()) if hasattr(place_id, "item") else int(place_id)
            pred_place_id = None
        
            if state == "revisit":
                # 1) Prefer the matched best member's place_id (strongest signal)
                if (best_member is not None) and ('idx' in best_member) and (best_member['idx'] is not None):
                    pred_place_id = get_place_id_by_index(dataset, best_member['idx'])
                # 2) Fallback: majority place_id among the candidate cluster's recent members
                if (pred_place_id is None) and ('members_slice' in locals()):
                    pred_place_id = majority_place_id_from_members(dataset, members_slice)
        
            place_id_match = (pred_place_id is not None) and (pred_place_id == gt_place_id)
        
            # ---------- Prediction rule (binary) ----------
            pred_positive = (state == "revisit") or (revisit_active and state == "same_spot")
                            
            # --------- Triplet mining + (optional) fine-tune ---------- 
            mined = pick_triplet(z_t_teacher, i, memory, prev_z=prev_emb, Kneg=Kneg, tau_new=tau_new)
            # --------- CONDITIONAL, RARE, SAFE ONLINE UPDATE ----------
            do_update = (i % UPDATE_EVERY == 0) and (mined is not None)
            if state == "same_spot":
                do_update = False
    
            if len(memory.items) < MIN_MEM_FOR_MINING:
                do_update = False
    
            if do_update:
                z_pos_cpu, Zneg_cpu = mined
                s_pos = float(z_t_teacher @ z_pos_cpu)                   # strength of positive
                s_neg_max = float((Zneg_cpu @ z_t_teacher).max()) if Zneg_cpu.numel() else -1.0
            
                # require a confident pos and at least one reasonably similar negative
                if (s_pos >= MIN_POS) and (s_neg_max >= MIN_NEG_SIM):
                    model.train()
                    z_anchor_student = embed_train(model, img)                          # on device, with grad
                    pos  = z_pos_cpu.to(device, non_blocking=True).float().view(-1)     # [D]
                    negs = Zneg_cpu.to(device, non_blocking=True).float()               # [K, D]
                    loss = triplet_loss(z_anchor_student, pos, negs)
            
                    if torch.isfinite(loss) and (loss.item() > 0):   # semi-hard guard
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
                        optimizer.step()
                    model.eval()
            
                    # (optional) very slow EMA teacher update so it follows over time:
                    with torch.no_grad():
                        m = 0.995
                        for ps, pt in zip(model.parameters(), teacher.parameters()):
                            pt.data.mul_(m).add_(ps.data, alpha=(1-m))
            
            if state == "new_place":
                cid, is_new, s = memory.add(z_t_teacher, idx=i, force_new=True, update_center=False)  # stores one new item
                last_cid = cid
            elif state == "revisit":
                cid = j_candidate if j_candidate is not None else last_cid
                if cid is not None:
                    memory.update_center(cid, z_t_teacher)  # EMA only, no append
                    last_cid = cid
            elif state == "same_spot":
                cid = last_cid
                if cid is not None:
                    memory.update_center(cid, z_t_teacher)  # or skip to freeze centers
 
            current_np = to_numpy_image(img[0])
            # --- choose what to display on the right ---
            right_np, right_title = None, ""   
 
            # Prefer the within-cluster "best member" if we already computed it (state == revisit)
            if state == "revisit" and (best_member is not None) and (best_member.get('idx') is not None):
                try:
                    right_np = get_image_by_index(dataset, best_member['idx'])
                    right_title = f"Cluster best (cid={j_candidate}, s={s_best:.2f}, idx={best_member['idx']})"
                except Exception:
                    right_np = None
            
            if state == "revisit" and (cid is not None):
                # start/refresh a revisit episode on the detected revisit
                revisit_active = True
                revisit_active_cid = cid
        
            elif state == "same_spot":
                # stay in the episode if we were already in one; do nothing else
                # (revisit_active remains whatever it was)
                pass
        
            elif state == "new_place":
                # as soon as a new place is detected, end the episode
                revisit_active = False
                revisit_active_cid = None
            # ----------------------------------------------------------
 
            # Left title from your state
            if state == "same_spot":
                left_title = f"t={i} SAME SPOT"
            elif state == "revisit":
                s_txt = f"{s_best:.2f}" if s_best is not None else "?"
                cid_txt = f"{j_candidate}" if j_candidate is not None else "?"
                left_title = f"t={i} REVISIT (cid={cid_txt}, best={s_txt})"
            else:
                left_title = f"t={i} NEW PLACE"
            
            # Throttle visualization so training isn’t spammy
            #SHOW_EVERY = 20  # show every N processed frames
            #if (i % (SHOW_EVERY*10) == 0):  # note you already subsample by 10; adjust as you like
            show_pair(current_np, right_np, left_title=left_title, right_title=right_title)
            prev_img_np = current_np    
    
 
            recent_cids.append(cid)
            # use the teacher embedding as "prev_emb"
            prev_emb = z_t_teacher
               
            # ---------- Prediction rule ----------
            # Positive if:
            #   (1) we just triggered a "revisit", OR
            #   (2) we're in an active episode and current frame is "same_spot"
            pred_positive = (state == "revisit") or (revisit_active and state == "same_spot")
            # (No cid checks, per your requirement: continuity is strictly via "same_spot".)
        
            # ---------- Metrics with place-id constraint on TP ----------------------  # NEW
            gt_loop = int(revisit.item()) if hasattr(revisit, "item") else int(revisit)
        
            # A "true positive" now requires: GT loop, predicted positive AND a detected revisit with matching place_id
            is_tp = (gt_loop == 1) and pred_positive and (state == "revisit") and place_id_match
        
            if is_tp:
                TP += 1
            elif gt_loop == 1:
                # Missed the correct loop (either we didn't fire, or wrong place_id) → FN
                FN += 1
            elif gt_loop == 0 and pred_positive:
                # Fired on a non-loop → FP
                FP += 1
            else:
                TN += 1
    
            # --- compute a unified continuous score for PR ---
            score_t = -1.0  # lowest possible start (cosine in [-1,1])
            
            # if you had a candidate cluster j with center similarity s_center:
            # keep s_center from `memory.nearest_center(z_t)` earlier:
            # j, s_center = memory.nearest_center(z_t)
            
            # prefer member-match similarity if available
            if state == "revisit" and s_best is not None:
                score_t = float(s_best)
            elif j is not None and j >= 0:
                score_t = float(s_center)     # how close to nearest center
            elif prev_emb is not None:
                score_t = float(sim_prev)
            
            all_scores.append(score_t)
            all_labels.append(int(revisit.item()))  # <-- Python int
    
    
            mem_mb = process.memory_info().rss / (1024 * 1024)
            memory_usage.append(mem_mb)
            memory_len.append(len(memory.items))
            #memory_len.append(len(memory.centers))   # Track memory of places instead of total memory 
    
        print("Total True positives: ", TP)
        print("Total True negatives: ", TN)
        print("Total False positives: ", FP)
        print("Total False negatives: ", FN)
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Recall: {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F1 Score: {f1:.3f}")
      
        plt.figure("Run")
        plt.title("Scoring") 
        plt.text(0.1, 0.8, 'True Positives: %.1f' %(TP))
        plt.text(0.1, 0.7, 'True Negatives: %.1f' %(TN))
        plt.text(0.1, 0.6, 'False Positives: %.1f' %(FP))
        plt.text(0.1, 0.5, 'False Negatives: %.1f' %(FN))
        plt.text(0.1, 0.4, 'Recall : %.3f' %(recall))
        plt.text(0.1, 0.3, 'Precision: %.3f' %(precision))
        plt.text(0.1, 0.2, 'F1 Score: %.3f' %(f1))
        plt.axis('off')    
    
        y_true  = np.asarray(all_labels, dtype=np.int32).ravel()   # ensure 1D
        y_score = np.asarray(all_scores, dtype=np.float32).ravel()
        
        # Sanity checks
        assert np.isfinite(y_score).all(), "y_score contains NaN/inf"
        if len(np.unique(y_true)) < 2:
            print("Warning: y_true contains a single class; PR curve is degenerate.")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)  # higher score = more likely loop
        ap = average_precision_score(y_true, y_score)
        
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
        plt.grid(True)
        plt.xlim(-0.01, 1.01)      # tiny padding on both sides
        plt.ylim(0.0, 1.02)        # or: plt.ylim(top=1.02)
        plt.show()    
    
        plt.figure("Memory Usage")
        plt.plot(memory_usage, label="Memory (MB)")
        plt.plot(memory_len, label="Place Memory Size", linestyle="--")
        plt.xlabel("Frame Index (sampled)")
        plt.ylabel("Memory Usage / Items")
        plt.title("Memory & Place Memory Growth")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # memory_usage should be MiB samples like: rss_mib = process.memory_info().rss / (1024**2
        final_mib = float(memory_usage[-1])
        peak_mib  = float(np.max(memory_usage))
        avg_mib   = float(np.mean(memory_usage))
        print(f"Final memory: {final_mib:.1f} MiB ({final_mib/1024:.2f} GiB)")
        print(f"Peak memory:  {peak_mib:.1f} MiB ({peak_mib/1024:.2f} GiB)")
        print(f"Average:      {avg_mib:.1f} MiB over {len(memory_usage)} samples")