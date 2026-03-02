# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 11:38:19 2025
Global residual place recognition 
v3- Changed to Summed not concatenation
v4- ADDED Slow EMA to cluster centers, not center updates at every times step
v5- Changed from \tau to \alpha for computing a_k
@author: Don Yates
"""
import os, time, math, random
from dataclasses import dataclass
from typing import Tuple, Dict, List
from torch.utils.data import Subset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from sklearn.neighbors import BallTree  # NEW
import re
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
import io
import copy
import psutil
import collections
from collections import deque

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained-ae", type=str, default=None,
                    help="Path to pretrained autoencoder weights")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from joint training checkpoint")
args = parser.parse_args()

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

Pretrain = True
Global_Train = True
EVAL_Live_Stream = False
# =========================
# Config
# =========================
@dataclass
class CFG:
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Embedding / centers
    latent_dim: int = 512
    K: int = 128
    alpha_init: float = 20.0
    concat: bool = False
    proj_dim: int = 512

    # Training
    batch_size: int = 64
    num_workers: int = 4
    epochs_warmup: int = 10
    epochs_joint: int = 8
    lr: float = 1e-3
    lr_centers: float = 5e-3
    triplet_margin: float = 0.2
    
    # ---- NEW: center update policy ----
    centers_update_mode: str = "grad"   # one of: "ema", "batch", "grad", "frozen"
    centers_momentum: float = 0.995     # used when centers_update_mode == "ema"
    centers_every: int = 1              # update frequency (in steps) for ema/batch

    # Loss weights
    w_rec: float = 1.0  #--- total reconstruction loss
    w_trip: float = 0.2 #--- total triplet loss
    w_div: float = 0.5  #--- total diversity loss
    w_cos: float = 0.75 #--- loss for center diversity
    w_ent: float = 0.9  #--- loss for assignment diversity 

    # KMeans (I have 32GB; 100k z’s ~ 200MB; safe)
    kmeans_samples: int = 20000
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 3
    kmeans_max_iter: int = 300

    # Eval
    recall_ks: Tuple[int, ...] = (1,2,3,4,5, 10, 15, 20, 25)

cfg = CFG()

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True  # speed
set_seed(cfg.seed)

# =========================
# --- VGG-16 backbone up to conv5_3, same as NetVLAD/AP-GeM ---
# =========================
class VGG16_Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, requires_grad: bool = True):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(weights=None)
        if not pretrained:
            # Good defaults when training from scratch
            for m in vgg.features.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # up to conv5_3 (index 30 is ReLU after conv5_3; [:31] keeps it)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:31])
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    def forward(self, x):
        return self.backbone(x)   # [B, 512, H', W']

# =========================
# GeM pooling
# =========================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p)) if learn_p else torch.tensor(p)
        self.eps = eps
    def forward(self, x):
        # x: [B,C,H,W]
        p = torch.clamp(self.p, min=1.0, max=8.0)
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p.view(1,1,1,1))
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / p.view(1,1,1,1))
        return x.squeeze(-1).squeeze(-1)  # [B,C]

# =========================
# Encoder
# =========================
# --- VGG16-based encoder that matches NetVLAD/AP-GeM features ---
class Encoder(nn.Module):
    """
    Uses the SAME VGG16 conv5_3 features as NetVLAD/AP-GeM, then GeM + (optional) FC to latent_dim.
    - If out_dim == 512, the FC becomes identity to keep features identical to AP-GeM's pooled dim.
    """
    def __init__(self, backbone: nn.Module, out_dim: int = 512, normalize_input: bool = True):
        super().__init__()
        self.backbone = backbone
        self.normalize_input = normalize_input
        self.gem = GeM(p=3.0, learn_p=True)
        self.fc = nn.Linear(512, out_dim) if out_dim != 512 else nn.Identity()

    def forward(self, x):
        f = self.backbone(x)                       # [B,512,h,w] from the SAME VGG16 as NetVLAD/AP-GeM
        if self.normalize_input:
            f = F.normalize(f, p=2, dim=1)         # match NetVLAD/AP-GeM normalization
        g = self.gem(f)                             # [B,512]
        z = self.fc(g)                              # [B,out_dim]
        return F.normalize(z, dim=1)

# =========================
# Decoder
# =========================
class Decoder(nn.Module):
    """Reconstruct to the same spatial size as the input by resizing at the end."""
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512*8*6), nn.ReLU(True)
        )
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
        xhat = self.up(h)  # ~256x192
        # Match input HxW for reconstruction loss
        xhat = F.interpolate(xhat, size=target_hw, mode="bilinear", align_corners=False)
        return xhat

# =========================
# Global-Residual VLAD head
# =========================
class GlobalResidualVLAD(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 D: int, K: int, alpha_init: float = 10.0,
                 concat: bool = False, proj_dim: int = 512,
                 out_l2: bool = True,          # L2 normalize final descriptor (NetVLAD-style)
                 per_cluster_alpha: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.K, self.D = K, D
        self.concat = concat
        self.out_l2 = out_l2

        # centers
        self.C = nn.Parameter(torch.randn(K, D))
        nn.init.orthogonal_(self.C)

        # α: learnable positive scale. Either global scalar or per-cluster.
        if per_cluster_alpha:
            # α_k for each cluster
            alpha0 = torch.full((K,), float(alpha_init))
            self.alpha_raw = nn.Parameter(torch.log(torch.expm1(alpha0)))  # softplus^-1
        else:
            # single global α
            self.alpha_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(alpha_init)))))

        self.per_cluster_alpha = per_cluster_alpha
        self.proj = nn.Linear((K*D) if concat else D, proj_dim)

    def alpha(self):
        a = F.softplus(self.alpha_raw) + 1e-6   # strictly positive
        return a.view(1, -1) if self.per_cluster_alpha else a  # [1,K] or scalar

    @torch.no_grad()                                    
    def update_centers_ema(self, z, a, momentum: float = cfg.centers_momentum, eps: float = 1e-6):
        Nk = a.sum(dim=0, keepdim=True).T.clamp_min(eps)   # [K,1]
        m  = (a.T @ z) / Nk                                # [K,D]
        self.C.data.mul_(momentum).add_((1 - momentum) * m)
        self.C.data = F.normalize(self.C.data, dim=1)      # keep centers well-spread

    def descriptor(self, z):
        # squared Euclidean distances to centers
        z2 = (z**2).sum(dim=1, keepdim=True)               # [B,1]
        c2 = (self.C**2).sum(dim=1, keepdim=True).T        # [1,K]
        d2 = (z2 + c2 - 2.0 * (z @ self.C.T)).clamp_min(0) # [B,K]

        # NetVLAD soft assignment: logits = -α * ||z - c||^2
        aparam = self.alpha()                               # scalar or [1,K]
        logits = -d2 * aparam                               # broadcast
        a = logits.softmax(dim=1)                           # [B,K]

        # VLAD residual aggregation
        res  = z[:, None, :] - self.C[None, :, :]           # [B,K,D]
        vlad = a[..., None] * res                           # [B,K,D]
        R    = vlad.reshape(z.size(0), -1) if self.concat else vlad.sum(dim=1)
        dvec = self.proj(R)
        if self.out_l2:
            dvec = F.normalize(dvec, dim=1)
        return dvec, a

    def forward(self, x):
        z = self.encoder(x)                                 # your encoder already L2-normalizes; fine
        dvec, a = self.descriptor(z)
        xhat = self.decoder(z, target_hw=(x.shape[-2], x.shape[-1]))
        return dvec, xhat, z, a

# =========================
# Losses
# =========================
def reconstruction_loss(xhat, x):  # simple L1; you can add SSIM if desired
    return (xhat - x).abs().mean()

def triplet_l2(Da, Dp, Dn, margin: float):
    # hinge on squared Euclidean distances
    di = (Da - Dp).pow(2).sum(dim=1)          # ||a - p||^2
    dn = (Da - Dn).pow(2).sum(dim=1)          # ||a - n||^2
    return F.relu(margin + di - dn).mean()

def diversity_loss(centers: torch.Tensor, assign: torch.Tensor, w_cos=0.5, w_ent=0.5):
    Cn = F.normalize(centers, dim=1)            # [K,D]
    G  = Cn @ Cn.T                               # [K,K]
    K  = G.size(0)
    off = G - torch.eye(K, device=G.device)
    L_cos = (off**2).sum() / (K*K - K + 1e-8)
    a = assign.clamp_min(1e-8)
    L_ent = -(a * a.log()).sum(dim=1).mean()
    return w_cos*L_cos + w_ent*L_ent

# =========================
# KMeans init on z
# =========================
@torch.no_grad()
def collect_z_samples(encoder: Encoder, loader: DataLoader, max_n: int, device: str):
    encoder.eval()
    chunks = []
    total = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        z = encoder(imgs).cpu().numpy()
        chunks.append(z)
        total += z.shape[0]
        if total >= max_n: break
    Z = np.concatenate(chunks, axis=0)
    return Z[:max_n]

def init_centers_kmeans(encoder: Encoder, train_loader: DataLoader, D: int, K: int, device: str) -> torch.Tensor:
    Z = collect_z_samples(encoder, train_loader, cfg.kmeans_samples, device)
    km = KMeans(n_clusters=K, init=cfg.kmeans_init, n_init=cfg.kmeans_n_init,
                max_iter=cfg.kmeans_max_iter, random_state=cfg.seed, verbose=0)
    km.fit(Z)
    C = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)
    return F.normalize(C, dim=1)

# =========================
# Train loops
# =========================
def train_warmup_ae(model: GlobalResidualVLAD, train_loader: DataLoader, device: str):
    if Pretrain:
        model.train()
        params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        
        for ep in range(cfg.epochs_warmup):
            t0 = time.time(); losses = []
            for imgs, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                
                # forward
                _, xhat, _, _ = model(imgs)
                Lrec = reconstruction_loss(xhat, imgs)
                
                #Ldiv = diversity_loss(model.C, a, w_cos=cfg.w_cos, w_ent=cfg.w_ent) * 0.2
                loss = Lrec # + cfg.w_div*Ldiv
                
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            print(f"[Warmup {ep+1}/{cfg.epochs_warmup}] loss={np.mean(losses):.4f} time={time.time()-t0:.1f}s")
        # --- Save Model ---
        torch.save(model.state_dict(), "weights/Pretrain_CA_resautoencoder.pth")
        print("Model saved ✅")

    else: #--- Load models weights
        print("Loading Model wieghts from Pretraining")  
        load_model_path = args
        model.load_state_dict(torch.load(load_model_path))  

def train_joint(model: GlobalResidualVLAD, trip_loader: DataLoader, recon_loader: DataLoader, device: str):
    if Global_Train:
        model.train()
        # always learn encoder/decoder/projection and alpha
        param_groups = [
            {"params": list(model.encoder.parameters()) +
                       list(model.decoder.parameters()) +
                       list(model.proj.parameters()),
             "lr": cfg.lr},
            {"params": [model.alpha_raw], "lr": cfg.lr_centers, "weight_decay": 0.0},
        ]
        # centers by gradient (if requested)
        if cfg.centers_update_mode == "grad":
            param_groups.append({"params": [model.C], "lr": cfg.lr_centers, "weight_decay": 0.0})
     
        opt = torch.optim.Adam(param_groups)
        recon_iter = iter(recon_loader)
        global_step = 0
        
        for ep in range(cfg.epochs_joint):
            t0 = time.time(); losses = []

            for anchor, pos, neg, _, _ in trip_loader:
                anchor = anchor.to(device, non_blocking=True)
                pos    = pos.to(device, non_blocking=True)
                neg    = neg.to(device, non_blocking=True)
    
                Da, xhat_a, z_a, a_a = model(anchor)
                Dp, _,     z_p, a_p  = model(pos)
                Dn, _,     z_n, a_n  = model(neg)
    
                Ltrip = triplet_l2(Da, Dp, Dn, cfg.triplet_margin)
                Lrec  = reconstruction_loss(xhat_a, anchor)
                Ldiv  = diversity_loss(model.C, a_a, w_cos=cfg.w_cos, w_ent=cfg.w_ent)
    
                loss = cfg.w_trip*Ltrip + cfg.w_rec*Lrec + cfg.w_div*Ldiv
                opt.zero_grad(); loss.backward(); opt.step()
                
                # gently update centers, helps with small batches:
                with torch.no_grad():
                    z_cat = torch.cat([z_a, z_p, z_n], dim=0)
                    a_cat = torch.cat([a_a, a_p, a_n], dim=0)
                    model.update_centers_ema(z_cat, a_cat, momentum=0.995)  # try 0.997–0.999 if still jumpy
                
                # ---- NEW: choose how centers update ----
                if cfg.centers_update_mode in ("ema", "batch") and (global_step % cfg.centers_every == 0):
                    # "ema" = slow; "batch" = hard recenter (momentum=0)
                    mom = cfg.centers_momentum if cfg.centers_update_mode == "ema" else 0.0
                    with torch.no_grad():
                        model.update_centers_ema(z_cat, a_cat, momentum=mom)
                
                losses.append(loss.item())
    
                # occasional pure recon step for stability
                try:
                    imgs, _ = next(recon_iter)
                except StopIteration:
                    recon_iter = iter(recon_loader)
                    imgs, _ = next(recon_iter)
                imgs = imgs.to(device, non_blocking=True)
                _, xhat, _, a2 = model(imgs)
                l2 = cfg.w_rec*reconstruction_loss(xhat, imgs) + (cfg.w_div*0.2)*diversity_loss(model.C, a2, cfg.w_cos, cfg.w_ent)
                opt.zero_grad(); l2.backward(); opt.step()
                global_step += 1
            
            with torch.no_grad():
                alpha_val = (F.softplus(model.alpha_raw).mean()).item()
            print(f"[Joint {ep+1}/{cfg.epochs_joint}] loss={np.mean(losses):.4f} alpha={alpha_val:.3f} time={time.time()-t0:.1f}s")
        # --- Save Model ---
        torch.save(model.state_dict(), "weights/GR_Model.pth")
        print("Model saved ✅")

    else: #--- Load models weights
        print("Loading Model wieghts from Previous Training")  
        load_model_path = args
        model.load_state_dict(torch.load(load_model_path)) 

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # gives TF32 a nudge

@torch.no_grad()
def compute_descriptors_gpu(model, loader, device):
    """
    Returns: descriptors [N,D], labels [N]
    Works with loaders that yield:
      - (imgs, labels_tensor)  OR
      - (imgs, meta_dict_with_label)  OR
      - imgs only  -> labels = -1
    """
    model.eval()
    descs, labels = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                imgs, y_or_meta = batch
                y = _extract_labels(y_or_meta, batch_size=imgs.size(0))
            else:
                imgs = batch[0]
                y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        else:
            imgs = batch
            y = torch.full((imgs.size(0),), -1, dtype=torch.long)

        imgs = imgs.to(device, non_blocking=True)
        d, _, _, _ = model(imgs)
        descs.append(F.normalize(d, dim=1).cpu())
        labels.append(y.cpu())

    return torch.cat(descs, 0), torch.cat(labels, 0)

@torch.no_grad()
def _extract_labels(y_or_meta, batch_size: int) -> torch.Tensor:
    """
    Accepts:
      - Tensor [B]
      - dict with 'label' -> tensor | list | tuple | numpy | scalar
      - list/tuple/None
    Returns LongTensor [B] with -1 for missing labels.
    """
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

@torch.no_grad()
def compute_desc_labels_assign(model, loader, device):
    """
    Works with:
      - (imgs, labels_tensor)
      - (imgs, meta_dict_with_label)
      - imgs only  -> labels = -1
    Returns: descs [N,D], labels [N], assigns [N,K]
    """
    model.eval()
    descs, labels, assigns = [], [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                imgs, y_or_meta = batch[0], batch[1]
                y = _extract_labels(y_or_meta, batch_size=imgs.size(0))
            else:
                imgs = batch[0]
                y = torch.full((imgs.size(0),), -1, dtype=torch.long)
        else:
            imgs = batch
            y = torch.full((imgs.size(0),), -1, dtype=torch.long)

        imgs = imgs.to(device, non_blocking=True)
        d, _, _, a = model(imgs)
        descs.append(F.normalize(d, dim=1).cpu())
        labels.append(y.cpu())
        assigns.append(a.cpu())

    return torch.cat(descs, 0), torch.cat(labels, 0), torch.cat(assigns, 0)

@torch.no_grad()
def recall_at_k_gpu(db_desc, db_labels, q_desc, q_labels, Ks):
    # keep descriptors L2-normalized (NetVLAD does this), but rank by Euclidean
    dists = torch.cdist(F.normalize(q_desc, dim=1), F.normalize(db_desc, dim=1), p=2)  # [Nq, Nd]
    out = {}
    for K in Ks:
        topk = torch.topk(dists, K, dim=1, largest=False).indices
        hits = (db_labels[topk] == q_labels.unsqueeze(1)).any(dim=1).float()
        out[f"R@{K}"] = hits.mean().item()
    return out

# --- simple helpers ---
def to_numpy_image(t):
    t = t.detach().cpu()
    if t.dim() == 3 and t.size(0) in (1,3):
        t = t.permute(1,2,0)
    t = t.clamp(0,1)
    return (t*255.0).round().to(torch.uint8).numpy()
#----------------------------------------------------------------------------

@torch.no_grad()
def preview_reconstruction(model, loader, device, n_show=3, title="AE warmup: input vs. reconstruction"):
    model.eval()
    # grab one small batch
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)[:n_show]
    dvec, xhat, _, _ = model(imgs)

    # to numpy
    inp = [to_numpy_image(imgs[i]) for i in range(len(imgs))]
    rec = [to_numpy_image(xhat[i]) for i in range(len(imgs))]

    # plot as 2-column grid
    cols = 2
    rows = len(inp)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.0))
    if rows == 1: axes = np.array([axes])
    fig.suptitle(title)
    for r in range(rows):
        axes[r,0].imshow(inp[r]); axes[r,0].set_title("Input"); axes[r,0].axis("off")
        axes[r,1].imshow(rec[r]); axes[r,1].set_title("Reconstruction"); axes[r,1].axis("off")
    plt.tight_layout(); plt.show()

def plot_recall_bars(recall_dict, save_path=None):
    keys = sorted(recall_dict.keys(), key=lambda s: int(s.split('@')[1]))
    vals = [recall_dict[k]*100 for k in keys]
    plt.figure(figsize=(5,4))
    plt.bar(keys, vals)
    plt.ylim(0, 100)
    plt.ylabel("Recall (%)")
    plt.title("Recall@K")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()

def plot_recall_curve(recall_dict, save_path=None, title="Recall@K"):
    """
    recall_dict: {"R@1": 0.63, "R@5": 0.81, ...} (values in [0,1])
    Plots a line chart of Recall vs K.
    """
    # Robustly parse Ks whether keys are "R@5" or just 5
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
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_recall_curves(curves: dict, save_path=None, title="Recall@K (25 m)"):
    """
    curves: {"Method A": R_dict_A, "Method B": R_dict_B, ...}
    Each R_dict is like {"R@1": 0.63, "R@5": 0.81, ...}
    """
    # union of all Ks
    all_ks = set()
    for R in curves.values():
        for k in R.keys():
            all_ks.add(int(re.search(r"@(\d+)", str(k)).group(1)) if "@" in str(k) else int(k))
    ks = sorted(all_ks)

    plt.figure(figsize=(7,4))
    for name, R in curves.items():
        ys = [100.0 * (R.get(f"R@{k}", R.get(k, 0.0))) for k in ks]
        plt.plot(ks, ys, marker="o", linewidth=2, label=name)

    plt.xticks(ks)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    plt.ylim(0, 100)
    plt.xlabel("K"); plt.ylabel("Recall")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_tsne(desc, labels, max_points=3000, save_path=None, title="t-SNE of descriptors"):
    n = min(len(desc), max_points)
    idx = np.random.choice(len(desc), n, replace=False)
    X = desc[idx].numpy()
    y = labels[idx].numpy()
    X2 = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30).fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], s=6, c=y, cmap="tab20")
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()

def plot_center_usage(assign_matrix, save_path=None, title="Center usage (soft assignments)"):
    # assign_matrix: [N, K]
    mean_assign = assign_matrix.mean(dim=0).numpy()
    Ks = [f"C{k}" for k in range(len(mean_assign))]
    plt.figure(figsize=(6,4))
    plt.bar(Ks, mean_assign)
    plt.ylabel("Mean assignment prob.")
    plt.title(title)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()

def build_paths_from_dataset(dataset):
    """Given your Pittsburg30k_Train_Dataset, reconstruct full file paths in loader order."""
    # Loader is shuffle=False for eval, so order matches CSV
    fnames = dataset.data.iloc[:, 0].astype(str).tolist()
    return [os.path.join(dataset.root_dir, f) for f in fnames]

def _resolve_with_roots(fname, roots):
    """Try to resolve fname against multiple roots, with a few fallbacks."""
    fpath = Path(fname)
    # Absolute path already?
    if fpath.is_absolute() and fpath.exists():
        return str(fpath)

    # Try root / fname (keeps any subdirs in fname)
    for root in roots:
        cand = Path(root) / fpath
        if cand.exists():
            return str(cand)

    # Try basenames under each root
    base = fpath.name
    for root in roots:
        cand = Path(root) / base
        if cand.exists():
            return str(cand)

    # Give up: return original (may 404 later, but we don't crash here)
    return str(fpath)

def safe_build_paths_from_dataset(dataset):
    """
    Return a list of absolute image paths in the SAME ORDER as the dataset.
    Supports:
      - Your DataFrame-based datasets (.data or .annotations) with single .root_dir
      - Multi-root datasets (e.g., TokyoEvalDatasetMultiRoot with .root_dirs and ._resolved_paths)
      - torchvision-style datasets with .imgs or .samples
      - torch.utils.data.Subset wrapping any of the above
    """
    # Unwrap Subset → underlying dataset
    base = dataset.dataset if isinstance(dataset, Subset) else dataset

    # --- Case 0: Multi-root dataset with pre-resolved paths (fast path)
    if hasattr(base, "_resolved_paths") and isinstance(getattr(base, "_resolved_paths"), list):
        paths = base._resolved_paths
        # Normalize to strings; fill any Nones by resolving now
        roots = getattr(base, "root_dirs", []) or [getattr(base, "root_dir", "")]
        roots = [str(r) for r in roots if r]
        filenames = None
        if hasattr(base, "annotations"):
            df = base.annotations
            # prefer explicit filename column if available
            if "filename" in df.columns:
                filenames = df["filename"].astype(str).tolist()
            else:
                filenames = df.iloc[:, 0].astype(str).tolist()
        elif hasattr(base, "data"):
            df = base.data
            filenames = df.iloc[:, 0].astype(str).tolist()

        out = []
        for i, p in enumerate(paths):
            if p is not None:
                out.append(str(p))
            else:
                # fallback resolve using the row's filename (if we have it) and roots
                name = filenames[i] if filenames is not None else ""
                out.append(_resolve_with_roots(name, roots))
        return out

    # --- Case 1: DataFrame in `.data` (your Pittsburg30k_* datasets, etc.)
    if hasattr(base, "data"):
        df = base.data
        roots = [getattr(base, "root_dir", "")]
        names = df.iloc[:, 0].astype(str).tolist()
        return [_resolve_with_roots(n, roots) for n in names]

    # --- Case 2: DataFrame in `.annotations` (db/query loaders, triplet loaders)
    if hasattr(base, "annotations"):
        df = base.annotations
        roots = []
        # Prefer multi-root if present, else single root_dir if present
        if hasattr(base, "root_dirs"):
            roots = [str(r) for r in base.root_dirs]
        elif hasattr(base, "root_dir"):
            roots = [str(getattr(base, "root_dir"))]
        else:
            roots = [""]

        # Prefer 'filename' column; else first col
        col = "filename" if "filename" in df.columns else df.columns[0]
        names = df[col].astype(str).tolist()
        return [_resolve_with_roots(n, roots) for n in names]

    # --- Case 3: torchvision ImageFolder-style datasets
    if hasattr(base, "imgs") and isinstance(base.imgs, list) and base.imgs and isinstance(base.imgs[0], (tuple, list)):
        return [str(p) for (p, _) in base.imgs]
    if hasattr(base, "samples") and isinstance(base.samples, list) and base.samples and isinstance(base.samples[0], (tuple, list)):
        return [str(p) for (p, _) in base.samples]

    # --- Last resort
    raise AttributeError(
        "Cannot infer file paths for this dataset. "
        f"Available attributes: {list(vars(base).keys())}. "
        "Expose one of: `.data`/`.annotations` (first col=filename) with `.root_dir`/`.root_dirs`, "
        "or `.imgs`/`.samples` like torchvision."
    )

def make_retrieval_gallery(q_idx_list, dists, q_paths, db_paths, q_labels, db_labels,
                           K=5, save_dir="runs/vis", name_prefix="gallery"):
    os.makedirs(save_dir, exist_ok=True)
    for qi in q_idx_list:
        # Top-K for this query
        topk = torch.topk(dists[qi], K).indices.tolist()
        rows = [q_paths[qi]] + [db_paths[j] for j in topk]
        labels = [int(q_labels[qi].item())] + [int(db_labels[j].item()) for j in topk]
        # Build a horizontal strip: [Query | #1 | #2 | ... | #K]
        images = [Image.open(p).convert("RGB") for p in rows]
        # Normalize sizes (thumb)
        H, W = 180, 240
        thumbs = [img.resize((W, H)) for img in images]
        # Annotate ranks and correctness
        draw_imgs = []
        q_lab = labels[0]
        for r, (thumb, lab) in enumerate(zip(thumbs, labels)):
            canvas = thumb.copy()
            d = ImageDraw.Draw(canvas)
            txt = "Q" if r == 0 else f"#{r}"
            ok = True if r == 0 else (lab == q_lab)
            color = (0,200,0) if (r == 0 or ok) else (220,0,0)
            # simple text box
            d.rectangle([5,5,50,28], fill=(0,0,0,140))
            d.text((8,8), txt, fill=color)
            draw_imgs.append(canvas)
        # Concatenate
        total_w = (K+1)*W
        strip = Image.new("RGB", (total_w, H), (255,255,255))
        for i, im in enumerate(draw_imgs):
            strip.paste(im, (i*W, 0))
        out_path = os.path.join(save_dir, f"{name_prefix}_q{qi}_K{K}.jpg")
        strip.save(out_path)

def make_retrieval_gallery_radius(q_idx_list, sims, q_paths, db_paths, pos_sets,
                                  K=5, save_dir="runs/vis", name_prefix="gallery25m"):
    os.makedirs(save_dir, exist_ok=True)
    topk_all = torch.topk(sims, K, dim=1).indices.cpu().numpy()
    H, W = 180, 240
    for qi in q_idx_list:
        topk = topk_all[qi]
        rows = [q_paths[qi]] + [db_paths[j] for j in topk]
        images = [Image.open(p).convert("RGB").resize((W, H)) for p in rows]

        draw_imgs = []
        for r, (thumb, db_idx) in enumerate(zip(images, [-1] + topk.tolist())):
            canvas = thumb.copy()
            d = ImageDraw.Draw(canvas)
            txt = "Q" if r == 0 else f"#{r}"
            ok = True if r == 0 else (db_idx in pos_sets[qi])
            color = (0, 200, 0) if ok else (220, 0, 0)
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

@torch.no_grad()
def recall_at_k_labels(sims: torch.Tensor,
                       q_labels: torch.Tensor,
                       db_labels: torch.Tensor,
                       Ks=(1,5,10),
                       skip_unlabeled: bool = True):
    """
    Radius-free recall using class labels.
    sims: [Nq, Nd] cosine similarities (higher is better)
    q_labels: [Nq] int labels (e.g., place IDs)
    db_labels: [Nd] int labels
    """
    assert sims.ndim == 2 and sims.size(0) == q_labels.numel()
    assert sims.size(1) == db_labels.numel()
    maxK = max(Ks)
    topk = torch.topk(sims, k=maxK, dim=1).indices  # [Nq, maxK]

    hits = torch.zeros(len(Ks), dtype=torch.long)
    valid = 0
    for i in range(sims.size(0)):
        qlab = int(q_labels[i])
        if skip_unlabeled and qlab < 0:
            continue
        valid += 1
        # labels of the retrieved top-K
        retrieved = db_labels[topk[i]]  # [maxK]
        for t, K in enumerate(Ks):
            if (retrieved[:K] == qlab).any():
                hits[t] += 1

    denom = max(1, valid)
    return {f"R@{k}": (hits[t].item() / denom) for t, k in enumerate(Ks)}

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

# --- pick_triplet from earlier (adapt to this memory) ---
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
    sample = dataset[idx]
    img = sample[0] if isinstance(sample, (list, tuple)) else sample
    return to_numpy_image(img)
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

def _fig_to_rgb_np(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    import PIL.Image as Image
    im = Image.open(io.BytesIO(arr)).convert("RGB")
    return np.array(im)

def _project_2d_from_centers(centers_cpu: torch.Tensor, cached=None):
    """
    centers_cpu: [K, D] float CPU tensor
    Returns: (Y [K,2] np.array, cache dict with P, mu, K, D)
    Always produces a 2D projection, even if rank<2.
    """
    K, D = centers_cpu.shape
    X = centers_cpu - centers_cpu.mean(0, keepdim=True)

    # Reuse cached basis if dims unchanged
    if cached and cached.get('K') == K and cached.get('D') == D and 'P' in cached and 'mu' in cached:
        P = cached['P']         # [D,2]
        mu = cached['mu']       # [D]
    else:
        # SVD for principal directions (may be rank-1!)
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            # extremely degenerate case; fall back to identity-ish
            Vh = torch.eye(D)

        if Vh.ndim == 1:
            Vh = Vh.unsqueeze(0)  # handle pathological returns

        # First direction
        if Vh.shape[0] >= 1:
            v1 = Vh[0].clone()               # [D]
        else:
            v1 = torch.zeros(D); v1[0] = 1.0

        # Second direction: if available, use it; else complete an orthonormal pair
        if Vh.shape[0] >= 2:
            v2 = Vh[1].clone()               # [D]
        else:
            # Build something orthogonal to v1
            # Start from a basis vector (or random) and Gram–Schmidt it.
            e = torch.zeros(D); e[0] = 1.0
            v2 = e - (torch.dot(e, v1) * v1)
            if torch.linalg.norm(v2) < 1e-8:
                r = torch.randn(D)
                v2 = r - (torch.dot(r, v1) * v1)
            n = torch.linalg.norm(v2)
            if n < 1e-12:   # D==1 case: no orthogonal direction exists
                v2 = torch.zeros(D)          # this yields a constant 2nd coord (fine for plotting)
            else:
                v2 = v2 / n

        # Normalize v1 as well (helps scaling)
        nv1 = torch.linalg.norm(v1)
        if nv1 > 0:
            v1 = v1 / nv1

        P = torch.stack([v1, v2], dim=1).contiguous()  # [D,2]
        mu = centers_cpu.mean(0)

        cached = {'P': P, 'mu': mu, 'K': K, 'D': D}

    # Project
    Y = (centers_cpu - mu) @ P      # [K,2]
    return Y.cpu().numpy(), cached

def make_embedding_panel(memory, z_cur_cpu, j_candidate=None, topk=5, cache_store=None):
    """
    memory: your OnlinePlaceMemory (must expose .centers: List[torch.Tensor] or tensor)
    z_cur_cpu: current embedding, 1D CPU tensor [D]
    j_candidate: int or None (candidate center id if you have one)
    topk: how many nearest centers to highlight (if no candidate)
    cache_store: dict you keep outside to cache projection basis
    Returns: (rgb_np, title_str)
    """

    # Gather centers -> [K,D] on CPU
    if len(memory.centers) == 0:
        # fallback placeholder
        fig = plt.figure(figsize=(4,3))
        plt.axis('off'); plt.text(0.5,0.5,"No centers yet", ha='center', va='center')
        rgb = _fig_to_rgb_np(fig); plt.close(fig)
        return rgb, "Latent Centers (empty)"

    C = torch.stack(memory.centers).float().cpu()  # [K,D]
    if cache_store is None:
        cache_store = {'proj': None}
    Y, cache_store['proj'] = _project_2d_from_centers(C, cached=cache_store.get('proj'))
    
    # Project current point
    P = cache_store['proj']['P']; mu = cache_store['proj']['mu']
    y_cur = ((z_cur_cpu.view(1,-1) - mu.view(1,-1)) @ P).cpu().numpy().reshape(-1)

    # Optionally find nearest centers to highlight when not revisiting
    highlight_ids = []
    if j_candidate is not None and j_candidate >= 0 and j_candidate < len(memory.centers):
        highlight_ids = [int(j_candidate)]
    else:
        # cosine on latent (in your code embeddings are l2-normalized; ok to use dot)
        sims = (C @ z_cur_cpu.view(-1)).cpu().numpy()
        order = np.argsort(-sims)[:max(1, min(topk, len(sims)))]
        highlight_ids = list(order)

    # Draw
    fig = plt.figure(figsize=(4.6,3.6))
    ax = plt.gca()
    ax.scatter(Y[:,0], Y[:,1], s=18, alpha=0.35, label="Centers")
    if len(highlight_ids):
        ax.scatter(Y[highlight_ids,0], Y[highlight_ids,1], s=38, alpha=0.9, marker='o', label="Nearest")
        for hid in highlight_ids:
            ax.text(Y[hid,0], Y[hid,1], f" {hid}", fontsize=8, alpha=0.9)

    ax.scatter([y_cur[0]],[y_cur[1]], s=60, marker='*', label="Current", zorder=5)
    ax.set_title("Latent Embedding (PCA2 on centers)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2); ax.legend(loc="best", fontsize=8)
    rgb = _fig_to_rgb_np(fig); plt.close(fig)

    if j_candidate is not None and len(highlight_ids)==1:
        ttl = f"Latent: cur vs cid={highlight_ids[0]}"
    else:
        ttl = "Latent: cur vs nearest centers"
    return rgb, ttl



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
    
    # #--- Dataloading
    # train_loader, _ = get_RobotCar_25_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # trip_loader,  _ = get_RobotCar_25_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_RobotCar_25_Eval_db_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # q_loader, q_ds   = get_RobotCar_25_Eval_query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)    
 
    train_loader, _ = get_RobotCar_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    trip_loader,  _ = get_RobotCar_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    db_loader, db_ds = get_RobotCar_Eval_db_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    q_loader, q_ds   = get_RobotCar_Eval_query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    dataset = "robotcar_seasons"
    
    # train_loader, _ = get_Tokyo_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # trip_loader,  _ = get_Tokyo_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_Tokyo_Eval_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # q_loader, q_ds = get_Tokyo_Query_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # train_loader, _ = get_Pittsburg30k_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # trip_loader,  _ = get_Pittsburg30k_Triplet_Train_dataloaders(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # db_loader, db_ds = get_Pittsburgh30k_Eval_db_dataloaders_5m(batch_size=cfg.batch_size, num_workers=cfg.num_workers,return_meta=False)
    # q_loader, q_ds = get_Pittsburgh30k_Eval_query_dataloaders_5m(batch_size=cfg.batch_size, num_workers=cfg.num_workers,return_meta=False)
    # # Grab coords as NumPy for BallTree positives (distance-based GT)
    # db_ll = db_ds.coords_deg_np()   # shape [N_db, 2] or None
    # q_ll  = q_ds.coords_deg_np()    # shape [N_q, 2] or None
    # assert db_ll is not None and q_ll is not None, "CSV must include lat/lon columns"

    # Model
    # enc = Encoder(out_dim=cfg.latent_dim)
    # dec = Decoder(in_dim=cfg.latent_dim)
    # model = GlobalResidualVLAD(enc, dec, D=cfg.latent_dim, K=cfg.K, tau_init=cfg.tau_init, concat=cfg.concat, proj_dim=cfg.proj_dim).to(device)
    # model = model.to(memory_format=torch.channels_last)
    
    # --- Build ONE backbone to share across methods ---
    backbone = VGG16_Backbone(
        pretrained=True,        # or False if training from scratch
        requires_grad=True      # set False if you want to freeze during AE warmup
    ).to(device)
    
    # --- Plug that backbone into GR-VLAD encoder ---
    enc = Encoder(backbone=backbone, out_dim=cfg.latent_dim, normalize_input=True)
    dec = Decoder(in_dim=cfg.latent_dim)
    
    model = GlobalResidualVLAD(enc, dec, D=cfg.latent_dim, K=cfg.K,
                           alpha_init=cfg.alpha_init, concat=cfg.concat, proj_dim=cfg.proj_dim,
                           out_l2=True, per_cluster_alpha=True).to(device)

    # 1) Warmup AE
    print("==> Warmup AE")
    train_warmup_ae(model, train_loader, device)
    preview_reconstruction(model, train_loader, device, n_show=3, title="After AE warmup")
    
    # 2) KMeans init in z-space
    print("==> KMeans init centers (z-space)")
    with torch.no_grad():
        C0 = init_centers_kmeans(model.encoder, train_loader, D=cfg.latent_dim, K=cfg.K, device=device)
        model.C.data.copy_(C0)
    # requires_grad depends on the mode
    if cfg.centers_update_mode == "grad":
        model.C.requires_grad_(True)
    else:
        model.C.requires_grad_(False)  # "ema", "batch", or "frozen"
    
    # 3) Joint Training
    print("==> Joint training (triplet + recon + diversity)")
    train_joint(model, trip_loader, train_loader, device)
    preview_reconstruction(model, train_loader, device, n_show=3, title="After Joint training")
    
    # # 4) Evaluation: distance-based Recall@K (25 m)
    # print("==> Compute descriptors (DB & Query)")
    # db_desc, db_lbls = compute_descriptors_gpu(model, db_loader, device)
    # q_desc,  q_lbls  = compute_descriptors_gpu(model, q_loader,  device)
    
    # # Euclidean dists for retrieval (both already L2-normalized)
    # db_n = F.normalize(db_desc, dim=1)
    # q_n  = F.normalize(q_desc,  dim=1)
    # dists = torch.cdist(q_n, db_n, p=2)  # [Nq, Nd]
    
    # # --- Label-based recall ---
    # R = recall_at_k_gpu(db_desc, db_lbls, q_desc, q_lbls, Ks=cfg.recall_ks)
    # print("==== Recall by labels (no radius) ====")
    # for k in cfg.recall_ks:
    #     print(f"R@{k}: {100*R[f'R@{k}']:.2f}%")

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
    # plot_recall_curve(R, save_path="runs/vis/recall_curve_labels.png", title="Recall@K (labels)")

    # # 2) Descriptors + assignments for DB (or both)
    # db_desc, db_labels, db_assign = compute_desc_labels_assign(model, db_loader, device)
    # q_desc,  q_labels,  q_assign  = compute_desc_labels_assign(model, q_loader,  device)
    
    # # 3) t-SNE (DB set; you can also do concatenated)
    # plot_tsne(db_desc, db_labels, max_points=3000, save_path="runs/vis/tsne_db.png", title="t-SNE (DB descriptors)")
    
    # # 4) Center usage histogram (DB)
    # plot_center_usage(db_assign, save_path="runs/vis/center_usage_db.png", title="Center usage (DB)")
    
    # # 5) Qualitative galleries: pick N random queries and show top-K
    # # Qualitative galleries with 25 m correctness
    # q_paths = safe_build_paths_from_dataset(q_ds)
    # db_paths = safe_build_paths_from_dataset(db_ds)
    # N_show = 8
    # picked = random.sample(range(len(q_paths)), k=min(N_show, len(q_paths)))
    # #make_retrieval_gallery_radius(picked, sims, q_paths, db_paths, pos_sets, K=5, save_dir="runs/vis", name_prefix="topk_25m")

    # make_retrieval_gallery(picked, dists, q_paths, db_paths, q_labels, db_labels,
    #                        K=5, save_dir="runs/vis", name_prefix="topk")
    # print("Saved galleries to runs/vis/")    

    embed_vis_cache = {'proj': None}  # holds PCA basis

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
        TAU_CENTER = 0.88     #--- How close the current frame needs to be to another cluster center to be considered for revisit
        TAU_MEMBER = 0.88     #--- Min similarity the current frame needs to be to be a revisit 
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
        latch_until = -1

        # ---- Episode state (keep it simple) ----
        revisit_active = False
        revisit_active_cid = None
        last_cid = None
            
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

            members_slice = None  # <-- set a default so it's always defined

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
                        m = 0.998
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
                # if cid is not None:
                #     memory.update_center(cid, z_t_teacher)  # or skip to freeze centers
 
            current_np = to_numpy_image(img[0])
            
            # # --- choose what to display on the right ---
            # right_np, right_title = None, ""
            
            # if state == "revisit" and (best_member is not None) and (best_member.get('idx') is not None):
            #     # keep your existing matched-image behavior
            #     try:
            #         right_np  = get_image_by_index(dataset, best_member['idx'])
            #         right_title = f"Cluster best (cid={j_candidate}, s={s_best:.2f}, idx={best_member['idx']})"
            #     except Exception:
            #         right_np, right_title = None, ""
            # else:
            #     # show the latent embedding panel (centers + current point)
            #     # NOTE: we use the TEACHER embedding here (z_t_teacher) for consistency with memory ops
            #     right_np, right_title = make_embedding_panel(
            #         memory=memory,
            #         z_cur_cpu=z_t_teacher.view(-1).cpu().float(),
            #         j_candidate=j_candidate if (state=="revisit") else None,
            #         topk=5,
            #         cache_store=embed_vis_cache
            #     )
            
            # --- episode latch ---
            if state == "revisit" and (cid is not None):
                revisit_active = True
                revisit_active_cid = cid
            elif state == "new_place":
                revisit_active = False
                revisit_active_cid = None

            # Left title from your state
            if state == "same_spot":
                left_title = f"t={i} SAME SPOT"
            elif state == "revisit":
                s_txt = f"{s_best:.2f}" if s_best is not None else "?"
                cid_txt = f"{j_candidate}" if j_candidate is not None else "?"
                left_title = f"t={i} REVISIT (cid={cid_txt}, best={s_txt})"
            else:
                left_title = f"NEW PLACE, s_cluster={s_center}"
                #left_title = f"t={i} NEW PLACE \n s_cluster={s_center}"
            
            # # Throttle visualization so training isn’t spammy
            # if (i % (SHOW_EVERY*10) == 0):  # note you already subsample by 10; adjust as you like
            #     show_pair(current_np, right_np, left_title=left_title, right_title=right_title)
            # prev_img_np = current_np    


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
            
            show_pair(current_np, right_np, left_title=left_title, right_title=right_title)
            prev_img_np = current_np
            # ----------------------------------------------------------
            
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
                
                    # # ---------- Metrics ----------
            # if revisit == 1 and pred_positive:
            #     TP += 1
            # elif revisit == 1 and not pred_positive:
            #     FN += 1
            # elif revisit == 0 and pred_positive:
            #     FP += 1
            # else:
            #     TN += 1
    
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
        #---------------------------------------------------------------------
    
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