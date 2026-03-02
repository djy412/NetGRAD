# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 06:22:23 2025

@author: djy41
"""
# VGG-16 conv local features + DBoW2 (HKMeans + TF-IDF) for VPR
import os, time, math, random, re
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
import copy
import psutil
import collections
from collections import deque
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.neighbors import BallTree  # used for radius-based recall

# =========================
# Your existing dataloaders
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
    get_RobotCar_25_Eval_query_dataloaders
)
EVAL_Live_Stream = False
# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Backbone / features
    pretrained_backbone: bool = True
    train_backbone: bool = False       # DBoW2 is non-parametric; usually freeze VGG
    normalize_input: bool = True       # L2 over channels per (h,w) location

    # DBoW2 vocabulary (Hierarchical K-Means)
    bow_k: int = 8                     # branching factor
    bow_L: int = 3                     # tree depth  → vocab size = k^L (e.g., 8^3=512)
    bow_max_desc: int = 150_000        # local descriptors sampled for vocab build
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 3
    kmeans_max_iter: int = 100

    # BoW scoring
    use_hist_intersection: bool = False  # if True: intersection (slower, small sets). Else cosine on TF-IDF.

    # Dataloading
    batch_size: int = 32
    num_workers: int = 8

    # Eval
    recall_ks: Tuple[int, ...] = (1,2,3,4,5,10,15,20,25)
    pos_radius_m: float = 25.0          # for distance-based GT
    
    # Used only when EVAL_Live_Stream=True
    triplet_margin: float = 0.2
    proj_dim: int = 512

cfg = CFG()

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
set_seed(cfg.seed)

# =========================
# VGG-16 up to conv5_3 (same as your other scripts)
# =========================
class VGG16_Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, requires_grad: bool = False):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(weights=None)
        if not pretrained:
            for m in vgg.features.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:31])  # conv5_3
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    def forward(self, x):
        return self.backbone(x)  # [B,512,H',W']

# =========================
# Helpers (same style as yours)
# =========================
@torch.no_grad()
def _extract_labels(y_or_meta, batch_size: int) -> torch.Tensor:
    import numpy as np, math
    if torch.is_tensor(y_or_meta): return y_or_meta.long()
    if isinstance(y_or_meta, dict):
        lab = y_or_meta.get("label", None)
        if lab is None: return torch.full((batch_size,), -1, dtype=torch.long)
        if torch.is_tensor(lab): return lab.long()
        if isinstance(lab, (list, tuple)):
            out=[]; 
            for v in lab:
                if v is None: out.append(-1); continue
                try:
                    if isinstance(v, float) and math.isnan(v): out.append(-1)
                    else: out.append(int(v))
                except: out.append(-1)
            return torch.tensor(out, dtype=torch.long)
        if isinstance(lab, np.ndarray): return torch.from_numpy(lab.astype(np.int64))
        try: return torch.tensor([int(lab)]*batch_size, dtype=torch.long)
        except: return torch.full((batch_size,), -1, dtype=torch.long)
    if isinstance(y_or_meta, (list, tuple)):
        out=[]
        for v in y_or_meta:
            try: out.append(-1 if v is None else int(v))
            except: out.append(-1)
        return torch.tensor(out, dtype=torch.long)
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
            txt = "Q" if r == 0 else f"#{r}"; ok = True if r == 0 else (lab == q_lab)
            color = (0,200,0) if (r == 0 or ok) else (220,0,0)
            d.rectangle([5,5,50,28], fill=(0,0,0,140))
            d.text((8,8), txt, fill=color); draw_imgs.append(canvas)
        strip = Image.new("RGB", ((K+1)*W, H), (255,255,255))
        for i, im in enumerate(draw_imgs): strip.paste(im, (i*W, 0))
        out_path = os.path.join(save_dir, f"{name_prefix}_q{qi}_K{K}.jpg")
        strip.save(out_path)

def plot_recall_curve(recall_dict, save_path=None, title="Recall@K"):
    def _parse_k(k):
        if isinstance(k, (int, float)): return int(k)
        m = re.search(r"@(\d+)", str(k)); return int(m.group(1)) if m else int(k)
    ks = sorted(_parse_k(k) for k in recall_dict.keys())
    ys = [100.0*(recall_dict.get(f"R@{k}", recall_dict.get(k))) for k in ks]
    plt.figure(figsize=(6,4))
    plt.plot(ks, ys, marker="o", linewidth=2)
    plt.xticks(ks); plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.ylim(0,100); plt.xlabel("K"); plt.ylabel("Recall"); plt.title(title); plt.grid(True, alpha=0.3)
    plt.tight_layout(); 
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()

def plot_tsne(desc, labels, max_points=3000, save_path=None, title="t-SNE of descriptors"):
    n = min(len(desc), max_points); idx = np.random.choice(len(desc), n, replace=False)
    X = desc[idx].numpy(); y = labels[idx].numpy()
    X2 = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
    plt.figure(figsize=(6,5)); plt.scatter(X2[:,0], X2[:,1], s=6, c=y, cmap="tab20")
    plt.title(title); plt.xticks([]); plt.yticks([]); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()

# =========================
# DBoW2: Hierarchical K-Means Vocabulary
# =========================
class HKNode:
    __slots__ = ("level", "centers", "children", "word_id")
    def __init__(self, level:int):
        self.level   = level
        self.centers = None    # torch.Tensor [k, D] at internal nodes
        self.children: List[HKNode] = []
        self.word_id = None    # int at leaves

class HKMeansVocab:
    """
    Simple hierarchical k-means tree for BoW quantization (DBoW2-style).
    - Branching k, depth L  → vocabulary size k^L.
    - Uses sklearn KMeans per level; keeps centers in torch for fast traversal.
    """
    def __init__(self, k:int, L:int, dim:int, device:str="cpu"):
        self.k = k; self.L = L; self.dim = dim
        self.device = device
        self.root = HKNode(level=0)
        self.n_words = k**L
        self._next_word_id = 0

    def _fit_node(self, X: np.ndarray, node: HKNode, level:int):
        if level == self.L:  # leaf
            node.word_id = self._next_word_id
            self._next_word_id += 1
            return

        # cluster this node's data into k children
        if X.shape[0] < self.k:
            # degenerate: few points — duplicate centers with noise
            C = np.zeros((self.k, self.dim), np.float32)
            if X.shape[0] > 0: C[:X.shape[0]] = X
            node.centers = torch.from_numpy(C).to(self.device)
            for j in range(self.k):
                child = HKNode(level+1); node.children.append(child)
                self._fit_node(X if X.shape[0]>0 else np.zeros((1,self.dim),np.float32), child, level+1)
            return

        km = KMeans(n_clusters=self.k, init="k-means++", n_init=cfg.kmeans_n_init,
                    max_iter=cfg.kmeans_max_iter, random_state=cfg.seed, verbose=0)
        km.fit(X)
        node.centers = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(self.device)
        labels = km.labels_

        for j in range(self.k):
            child = HKNode(level+1)
            node.children.append(child)
            subset = X[labels == j]
            self._fit_node(subset, child, level+1)

    def fit(self, X: np.ndarray):
        assert X.ndim == 2 and X.shape[1] == self.dim
        self._next_word_id = 0
        self._fit_node(X, self.root, level=0)
        assert self._next_word_id == self.n_words, f"Built {self._next_word_id} words, expected {self.n_words}"

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N,D] local descriptors (torch, device can be cuda or cpu).
        Returns: word_ids [N] int64.
        """
        if x.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=x.device)
        word_ids = []
        for i in range(x.shape[0]):
            v = x[i:i+1]  # [1,D]
            node = self.root
            while node.level < self.L:
                # choose nearest child center
                # centers: [k, D]
                d = torch.cdist(v, node.centers)  # [1,k]
                idx = int(torch.argmin(d, dim=1).item())
                node = node.children[idx]
            word_ids.append(node.word_id)
        return torch.tensor(word_ids, dtype=torch.long, device=x.device)

# =========================
# Local descriptor extraction (conv5_3 per-location)
# =========================
@torch.no_grad()
def collect_local_descriptors(backbone: nn.Module, loader: DataLoader, device: str,
                              max_desc: int) -> np.ndarray:
    backbone.eval()
    pool = []
    total = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.to(device, non_blocking=True)
        fmap = backbone(imgs)            # [B,512,h,w]
        if cfg.normalize_input:
            fmap = F.normalize(fmap, p=2, dim=1)  # per (h,w)
        B, C, H, W = fmap.shape
        fmap = fmap.permute(0,2,3,1).contiguous().view(-1, C)  # [B*H*W,512]
        arr = fmap.detach().cpu().numpy().astype(np.float32)
        pool.append(arr)
        total += arr.shape[0]
        if total >= max_desc: break
    X = np.concatenate(pool, axis=0)
    if X.shape[0] > max_desc:
        idx = np.random.choice(X.shape[0], max_desc, replace=False)
        X = X[idx]
    return X

@torch.no_grad()
def fmap_to_local(fmap: torch.Tensor) -> torch.Tensor:
    # fmap: [B,512,h,w] -> list of per-image [Ni,512]
    B, C, H, W = fmap.shape
    feats = fmap.permute(0,2,3,1).contiguous().view(B, -1, C)  # [B,HW,512]
    return feats  # not normalized here (already normalized right after backbone)

# =========================
# Build BoW histograms + TF-IDF
# =========================
@torch.no_grad()
def build_bow_histograms(backbone: nn.Module, loader: DataLoader, vocab: HKMeansVocab, device: str):
    backbone.eval()
    bows = []
    labels = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, y_or_meta = batch[0], batch[1]
            y = _extract_labels(y_or_meta, batch_size=imgs.size(0))
        elif isinstance(batch, (list, tuple)):
            imgs = batch[0]; y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        else:
            imgs = batch; y = torch.full((imgs.size(0),), -1, dtype=torch.long)

        imgs = imgs.to(device, non_blocking=True)
        fmap = backbone(imgs)                      # [B,512,h,w]
        if cfg.normalize_input:
            fmap = F.normalize(fmap, p=2, dim=1)
        locals_per_img = fmap_to_local(fmap)       # [B,HW,512]

        for i in range(locals_per_img.size(0)):
            loc = locals_per_img[i]                # [HW,512]
            wids = vocab.quantize(loc)             # [HW]
            # term-frequency (counts)
            hist = torch.bincount(wids, minlength=vocab.n_words).float()
            # normalize L1 (tf)
            if hist.sum() > 0: hist /= hist.sum()
            bows.append(hist)
        labels.append(y.cpu())

    bows = torch.stack(bows, dim=0).cpu()          # [N, V]
    labels = torch.cat(labels, 0)
    return bows, labels  # TF histograms (L1-normalized)

@torch.no_grad()
def apply_tfidf(db_tf: torch.Tensor, q_tf: torch.Tensor):
    """
    db_tf, q_tf: [N,V], L1-normalized TF histograms.
    Computes IDF on DB, applies to both, then L2-normalizes for cosine.
    """
    N, V = db_tf.shape
    df = (db_tf > 0).sum(dim=0).clamp(min=1)  # document frequency per word
    idf = torch.log((N + 1.0) / (df.float() + 1.0)) + 1.0  # smooth IDF (>0)

    db_w = db_tf * idf
    q_w  = q_tf * idf
    # L2 normalize for cosine
    db_w = F.normalize(db_w, p=2, dim=1)
    q_w  = F.normalize(q_w,  p=2, dim=1)
    return db_w, q_w, idf

@torch.no_grad()
def sims_cosine(q: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
    # q:[Nq,V], db:[Nd,V] -> [Nq,Nd]
    return q @ db.T

@torch.no_grad()
def sims_hist_intersection(q: torch.Tensor, db: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """
    Histogram intersection: s(q,d) = sum_i min(q_i, d_i)
    Naive but chunked; use only for small vocab/sets.
    """
    Nq, V = q.shape; Nd = db.shape[0]
    out = torch.empty((Nq, Nd), dtype=torch.float32)
    for i in range(0, Nq, chunk):
        qb = q[i:i+chunk].unsqueeze(1)            # [b,1,V]
        # Broadcasting min over db rows
        # Memory caution: (b,Nd,V) — OK only for small b or small V
        m = torch.minimum(qb, db.unsqueeze(0))    # [b,Nd,V]
        out[i:i+chunk] = m.sum(dim=2)
    return out

# =========================
# Radius-based recall (same as your helper)
# =========================
EARTH_R = 6371008.8
def build_positive_sets_by_radius(db_ll_deg: np.ndarray, q_ll_deg: np.ndarray, radius_m: float = 25.0) -> List[set]:
    db_rad = np.radians(db_ll_deg); q_rad = np.radians(q_ll_deg)
    tree = BallTree(db_rad, metric='haversine')
    neigh = tree.query_radius(q_rad, r=radius_m / EARTH_R)
    return [set(arr.tolist()) for arr in neigh]

@torch.no_grad()
def recall_at_k_radius(sims: torch.Tensor, pos_sets: List[set], Ks: Tuple[int, ...], skip_no_pos: bool = True) -> Dict[str, float]:
    maxK = max(Ks)
    topk = torch.topk(sims, k=maxK, dim=1).indices.cpu().numpy()
    hits = np.zeros(len(Ks), dtype=np.int64); valid = 0
    for i, gt in enumerate(pos_sets):
        if skip_no_pos and len(gt) == 0: continue
        valid += 1
        for t, K in enumerate(Ks):
            if any(j in gt for j in topk[i, :K]): hits[t] += 1
    denom = max(1, valid)
    return {f"R@{k}": hits[t] / denom for t, k in enumerate(Ks)}

# ---- Add this if not already defined in this file ----
@torch.no_grad()
def recall_at_k_gpu(db_desc, db_labels, q_desc, q_labels, Ks, device=None, skip_unlabeled=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    db = F.normalize(db_desc.to(device), dim=1)
    q  = F.normalize(q_desc.to(device),  dim=1)
    db_labels = db_labels.to(device).long()
    q_labels  = q_labels.to(device).long()
    maxK = max(Ks); B = 4096
    hits = {K: 0 for K in Ks}; valid = 0
    for i in range(0, q.size(0), B):
        qb = q[i:i+B]; qlb = q_labels[i:i+B]
        if skip_unlabeled:
            m = (qlb >= 0)
            if m.sum() == 0: continue
            qb = qb[m]; qlb = qlb[m]
        sims = qb @ db.T
        topk = sims.topk(maxK, dim=1).indices
        retrieved = db_labels[topk]
        qlb = qlb.unsqueeze(1)
        for K in Ks:
            hits[K] += (retrieved[:, :K] == qlb).any(dim=1).sum().item()
        valid += qb.size(0)
    denom = max(1, valid)
    return {f"R@{K}": hits[K] / denom for K in Ks}


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
        dvec, _, _, _ = model(x)                 # on device
    return F.normalize(dvec.squeeze(0), p=2, dim=0).cpu()  #  1DCPU

# new: for training updates (no no_grad, stays on device)
def embed_train(model, x):
    dvec, _, _, _ = model(x)                     # on device, requires grad
    return F.normalize(dvec.squeeze(0), p=2, dim=0)        # device 1D




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

    # ---- RobotCar Seasons v2 loaders ----
    # train_loader, _ = get_RobotCar_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_RobotCar_Eval_db_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # q_loader, q_ds   = get_RobotCar_Eval_query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # ---- Tokyo loaders ----
    train_loader, _ = get_Tokyo_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    db_loader, db_ds = get_Tokyo_Eval_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    q_loader,  q_ds  = get_Tokyo_Query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # ---- Build backbone (same as NetVLAD) ----
    backbone = VGG16_Backbone(
        pretrained=cfg.pretrained_backbone,
        requires_grad=cfg.train_backbone     # usually False for DBoW2
    ).to(device).eval()

    # ---- (1) Collect local conv5_3 descriptors and build HK-Means vocab ----
    print("==> Sampling local descriptors for vocabulary…")
    X = collect_local_descriptors(
        backbone=backbone,
        loader=train_loader,
        device=device,
        max_desc=cfg.bow_max_desc
    )  # NumPy [N,512]

    print(f"==> Building HKMeans vocab: k={cfg.bow_k}, L={cfg.bow_L} → {cfg.bow_k ** cfg.bow_L} words")
    vocab = HKMeansVocab(k=cfg.bow_k, L=cfg.bow_L, dim=512, device=device)
    vocab.fit(X)  # builds the tree on CPU (sklearn) and stores centers as torch on device
    print(f"Vocabulary ready with {vocab.n_words} words.")

    # ---- (2) Build BoW histograms for DB and Query ----
    print("==> Vectorizing DB images (TF histograms)…")
    db_tf, db_labels = build_bow_histograms(backbone, db_loader, vocab, device)  # [N_db,V], [N_db]
    print("==> Vectorizing Query images (TF histograms)…")
    q_tf,  q_labels  = build_bow_histograms(backbone, q_loader,  vocab, device)  # [N_q,V],  [N_q]

    # ---- (3) TF-IDF weighting (default) or histogram intersection (optional) ----
    if cfg.use_hist_intersection:
        # Use raw TF (L1) with histogram intersection similarity
        print("==> Using histogram intersection scoring.")
        db_desc = db_tf.clone()
        q_desc  = q_tf.clone()
        sims    = sims_hist_intersection(q_desc, db_desc)   # [N_q, N_db]
    else:
        print("==> Using TF-IDF + cosine scoring.")
        db_desc, q_desc, idf = apply_tfidf(db_tf, q_tf)     # L2-normalized TF-IDF
        sims = sims_cosine(q_desc, db_desc)                 # [N_q, N_db]

    # ---- (4) Label-based Recall@K (Tokyo provides labels; no GPS radius) ----
    R = recall_at_k_gpu(db_desc, db_labels, q_desc, q_labels, Ks=cfg.recall_ks)
    print("==== Recall (labels) — DBoW2 (VGG16) on Tokyo ====")
    for k in cfg.recall_ks:
        print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")

    # ---- (5) Plots and qualitative gallery ----
    os.makedirs("runs/vis", exist_ok=True)
    plot_recall_curve(R, save_path="runs/vis/dbow2_tokyo_recall.png",
                      title="DBoW2 (VGG16) Recall@K — Tokyo")

    # Optional t-SNE on DB (TF-IDF space)
    try:
        plot_tsne(db_desc.cpu(), db_labels.cpu(), max_points=3000,
                  save_path="runs/vis/dbow2_tokyo_tsne_db.png",
                  title="t-SNE (DBoW2 TF-IDF, DB)")
    except Exception as e:
        print("t-SNE skipped:", e)

    # # Gallery
    # q_paths = safe_build_paths_from_dataset(q_ds)
    # db_paths = safe_build_paths_from_dataset(db_ds)

    # picked = random.sample(range(len(q_paths)), k=min(8, len(q_paths)))
    # make_retrieval_gallery(picked, sims, q_paths, db_paths, q_labels, db_labels,
    #                        K=5, save_dir="runs/vis", name_prefix="dbow2_tokyo_topk")
    # print("Saved DBoW2 Tokyo plots to runs/vis/")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # device = cfg.device
    # print("Device:", device)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.set_float32_matmul_precision("high")

    # # Pittsburgh 30k (5m protocol)
    # train_loader, _ = get_Pittsburg30k_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_Pittsburgh30k_Eval_db_dataloaders_5m(batch_size=cfg.batch_size, num_workers=cfg.num_workers, return_meta=False)
    # q_loader,  q_ds  = get_Pittsburgh30k_Eval_query_dataloaders_5m(batch_size=cfg.batch_size, num_workers=cfg.num_workers, return_meta=False)

    # # coords for radius-based GT
    # db_ll = db_ds.coords_deg_np()
    # q_ll  = q_ds.coords_deg_np()
    # assert db_ll is not None and q_ll is not None, "CSV must include lat/lon columns"

    # # -------------------- Build backbone --------------------
    # backbone = VGG16_Backbone(
    #     pretrained=cfg.pretrained_backbone,
    #     requires_grad=cfg.train_backbone
    # ).to(device).to(memory_format=torch.channels_last)

    # # -------------------- Build vocabulary --------------------
    # print(f"==> Collecting up to {cfg.bow_max_desc} local descriptors for HKMeans...")
    # X = collect_local_descriptors(backbone, train_loader, device, max_desc=cfg.bow_max_desc)
    # print(f"Collected: {X.shape[0]} x {X.shape[1]}")

    # D = X.shape[1]
    # vocab = HKMeansVocab(k=cfg.bow_k, L=cfg.bow_L, dim=D, device=device if torch.cuda.is_available() else "cpu")
    # print(f"==> Training HKMeans tree: k={cfg.bow_k}, L={cfg.bow_L} (|V|={cfg.bow_k**cfg.bow_L})")
    # t0 = time.time()
    # vocab.fit(X)
    # print(f"Vocab trained in {time.time()-t0:.1f}s")

    # # -------------------- Encode DB / Query as BoW (TF) --------------------
    # print("==> Building BoW histograms for DB...")
    # db_tf, db_labels = build_bow_histograms(backbone, db_loader, vocab, device)
    # print("==> Building BoW histograms for Query...")
    # q_tf,  q_labels  = build_bow_histograms(backbone, q_loader,  vocab, device)

    # # -------------------- TF-IDF weighting + normalization --------------------
    # print("==> Applying TF-IDF (IDF from DB) and normalizing...")
    # db_bow, q_bow, idf = apply_tfidf(db_tf, q_tf)

    # # -------------------- Similarity + Recall --------------------
    # print("==> Computing similarities and Recall@K (25 m)")
    # if cfg.use_hist_intersection:
    #     sims = sims_hist_intersection(q_bow, db_bow)   # slower; use for small sets
    # else:
    #     sims = sims_cosine(q_bow, db_bow)              # fast and robust

    # pos_sets = build_positive_sets_by_radius(db_ll, q_ll, radius_m=cfg.pos_radius_m)
    # R = recall_at_k_radius(sims, pos_sets, cfg.recall_ks, skip_no_pos=True)
    # print(f"Distance-based GT: {cfg.pos_radius_m} m | Queries with ≥1 positive: "
    #       f"{sum(1 for s in pos_sets if len(s)>0)}/{len(pos_sets)}")
    # print("==== Recall (25 m) ====")
    # for k in cfg.recall_ks:
    #     print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")

    # # -------------------- Visualizations --------------------
    # os.makedirs("runs/vis", exist_ok=True)
    # plot_recall_curve(R, save_path="runs/vis/dbow2_recall_curve_25m.png",
    #                   title="VGG16-DBoW2 Recall@K (25 m)")

    # # t-SNE on BoW (it’s dense but OK at these sizes)
    # plot_tsne(db_bow, db_labels, max_points=3000, save_path="runs/vis/dbow2_tsne_db.png",
    #           title="t-SNE (VGG16-DBoW2 DB BoW)")

    # # Qualitative galleries (uses cosine/HI sims already computed)
    # q_paths = safe_build_paths_from_dataset(q_ds)
    # db_paths = safe_build_paths_from_dataset(db_ds)
    # N_show = min(8, len(q_paths))
    # picked = random.sample(range(len(q_paths)), k=N_show)
    # make_retrieval_gallery(picked, sims, q_paths, db_paths, q_labels, db_labels,
    #                        K=5, save_dir="runs/vis", name_prefix="dbow2_topk")

    # print("Saved DBoW2 galleries and plots to runs/vis/")