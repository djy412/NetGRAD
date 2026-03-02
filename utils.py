"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.patches as patches
import collections
from config import CHANNELS


#*****************************************************************************
#--- Evaluate Critiron
#*****************************************************************************
def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Remap to 0..n-1 in case labels start at 1 (or are arbitrary)
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    yt = le_true.fit_transform(y_true)
    yp = le_pred.fit_transform(y_pred)

    n_true = len(le_true.classes_)
    n_pred = len(le_pred.classes_)
    C = np.zeros((n_pred, n_true), dtype=np.int64)
    np.add.at(C, (yp, yt), 1)  # build contingency table

    # Hungarian on a square cost matrix (pad with zeros if rectangular)
    if n_pred > n_true:
        C_pad = np.hstack([C, np.zeros((n_pred, n_pred - n_true), dtype=C.dtype)])
    elif n_true > n_pred:
        C_pad = np.vstack([C, np.zeros((n_true - n_pred, n_true), dtype=C.dtype)])
    else:
        C_pad = C

    row_ind, col_ind = linear_sum_assignment(C_pad.max() - C_pad)
    acc = C_pad[row_ind, col_ind].sum() / len(y_true)
    return acc
#----------------------------------------------------------------------------

#*****************************************************************************
#--- Evaluate Critiron
#*****************************************************************************
def cluster_acc_top_N(y_true, y_pred, n_clusters, topN=1):
    """
    Calculate clustering accuracy with Hungarian matching.
    Supports top-N predictions.

    y_true: (n_samples,) ground truth labels
    y_pred: (n_samples, topN) array if topN>1, else (n_samples,) array
    n_clusters: number of clusters
    topN: int, number of top predictions to consider
    """

    y_true = np.asarray(y_true).astype(np.int64)

    if topN == 1:
        # --- standard case ---
        assert y_pred.ndim == 1
        y_pred = np.asarray(y_pred).astype(np.int64)

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind_row, ind_col = linear_sum_assignment(w.max() - w)
        return w[ind_row, ind_col].sum() / y_pred.size

    else:
        # --- top-N case ---
        assert y_pred.ndim == 2 and y_pred.shape[1] == topN

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(y_pred.shape[0]):
            for pred in y_pred[i]:
                w[pred, y_true[i]] += 1

        ind_row, ind_col = linear_sum_assignment(w.max() - w)

        # Now check correctness per sample
        mapping = {r: c for r, c in zip(ind_row, ind_col)}

        correct = 0
        for i in range(y_pred.shape[0]):
            # mapped cluster IDs for all top-N predictions
            mapped_preds = [mapping[p] for p in y_pred[i] if p in mapping]
            if y_true[i] in mapped_preds:
                correct += 1

        return correct / y_pred.shape[0]
#----------------------------------------------------------------------------

#*****************************************************************************
#--- Evaluate Critiron
#*****************************************************************************
def calculate_purity(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster_index in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    return accuracy_score(y_true, y_voted_labels)
#----------------------------------------------------------------------------

#*****************************************************************************
#--- Set Seed
#*****************************************************************************
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#----------------------------------------------------------------------------

#*****************************************************************************
# --- simple helpers ---
def to_numpy_image(t):
    t = t.detach().cpu()
    if t.dim() == 3 and t.size(0) in (1,3):
        t = t.permute(1,2,0)
    t = t.clamp(0,1)
    return (t*255.0).round().to(torch.uint8).numpy()
#----------------------------------------------------------------------------

def draw_with_border(img_np, color, title=None, linewidth=6):
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.imshow(img_np); ax.axis('off')
    if title: ax.set_title(title)
    h, w = img_np.shape[:2]
    rect = patches.Rectangle((0,0), w, h, linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    plt.tight_layout(); plt.show()
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
#----------------------------------------------------------------------------

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

#--- Compute embeddings for Evaluation
#************************************************************************
def compute_embeddings(model, images, device):
    """
    images: tensor [B,C,H,W]
    returns: normalized embeddings [B, D]
    """
    with torch.no_grad():
        _, z, _ = model(images.to(device))
    z = F.normalize(z, p=2, dim=1)  # cosine normalization
    return z
#---------------------------------------------------------------------------

#--- Compute embedding similarity for Evaluation
#************************************************************************
def retrieve_top_k(query_emb, dataset_emb, k=5, metric="cosine"):
    """
    Returns top-k indices and scores
    """
    # Determine the max possible k (number of database embeddings)
    max_k = dataset_emb.shape[0]
    k = min(k, max_k)  # Prevent asking for more than available

    if metric == "cosine":
        sims = torch.matmul(query_emb, dataset_emb.T)
        scores, indices = torch.topk(sims, k=k, dim=-1)
    elif metric == "euclidean":
        dists = torch.cdist(query_emb, dataset_emb)
        scores, indices = torch.topk(-dists, k=k, dim=-1)  # negative for closest
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    return indices.cpu(), scores.cpu()
#----------------------------------------------------------------------------

#--- Recall Evaluation Function
#************************************************************************
def Compute_recall_at_N(dataset_emb, query_emb, dataset_labels, query_labels,
                        max_k=50, metric="cosine", device="cpu", chunk_size=5000):
    """
    dataset_emb: [N_db, D] torch tensor
    query_emb:   [N_q, D] torch tensor
    """
    if isinstance(dataset_labels, torch.Tensor):
        dataset_labels = dataset_labels.cpu().numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.cpu().numpy()

    N_q, D = query_emb.shape
    N_db = dataset_emb.shape[0]

    dataset_emb = dataset_emb.to(device)
    query_emb = query_emb.to(device)

    all_topk = []

    if metric == "cosine":
        dataset_emb = F.normalize(dataset_emb, p=2, dim=1)
        query_emb   = F.normalize(query_emb, p=2, dim=1)

        for start in range(0, N_q, chunk_size):
            end = min(start + chunk_size, N_q)
            q_chunk = query_emb[start:end]  # [chunk, D]
            sim = torch.matmul(q_chunk, dataset_emb.t())  # [chunk, N_db]
            _, topk_idxs = torch.topk(sim, k=max_k, dim=1, largest=True, sorted=True)
            all_topk.append(topk_idxs.cpu())
    elif metric == "euclidean":
        for start in range(0, N_q, chunk_size):
            end = min(start + chunk_size, N_q)
            q_chunk = query_emb[start:end]
            dists = torch.cdist(q_chunk, dataset_emb)  # [chunk, N_db]
            topk_idxs = torch.argsort(dists, dim=1)[:, :max_k]
            all_topk.append(topk_idxs.cpu())
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    topk_idxs_np = torch.cat(all_topk, dim=0).numpy()  # [N_q, max_k]

    # --- Recall computation ---
    retrieved_labels = dataset_labels[topk_idxs_np]  # [N_q, max_k]
    query_labels_exp = np.expand_dims(query_labels, axis=1)
    matches = (retrieved_labels == query_labels_exp)

    cumulative_matches = np.cumsum(matches.astype(np.int32), axis=1)
    cumulative_matches = (cumulative_matches > 0).astype(np.int32)

    recalls = cumulative_matches.sum(axis=0) / float(N_q)
    return recalls
#----------------------------------------------------------------------------

@torch.no_grad()
def batch_to_emb(model, imgs, device):
    model.eval()
    _, z, _ = model(imgs.to(device))
    return F.normalize(z, p=2, dim=1).cpu()  # [B, D]
#----------------------------------------------------------------------------


@torch.no_grad()
def streaming_topk_for_query(query_emb, db_loader, model, device, k=5, metric="cosine"):
    # query_emb: [D] on CPU
    query_emb = query_emb.to(device).unsqueeze(0)  # [1, D]
    if metric == "cosine":
        query_emb = F.normalize(query_emb, p=2, dim=1)

    best_scores = torch.full((k,), float("-inf"))
    best_indices = torch.full((k,), -1, dtype=torch.long)

    db_offset = 0
    for imgs, _labels in db_loader:
        db_emb = batch_to_emb(model, imgs, device)  # [B, D]
        if metric == "cosine":
            sims = (query_emb @ db_emb.to(device).T).squeeze(0).cpu()  # [B]
            batch_scores = sims
        elif metric == "euclidean":
            d = torch.cdist(query_emb, db_emb.to(device)).squeeze(0).cpu()  # [B]
            batch_scores = -d  # higher is better
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        # Top-k within this batch
        b = batch_scores.shape[0]
        topb = min(k, b)
        b_scores, b_idx = torch.topk(batch_scores, k=topb)

        # Merge with running best
        merged_scores = torch.cat([best_scores, b_scores])
        merged_idx    = torch.cat([best_indices, b_idx + db_offset])
        keep_scores, keep_pos = torch.topk(merged_scores, k=k)
        keep_idx = merged_idx[keep_pos]

        best_scores, best_indices = keep_scores, keep_idx
        db_offset += b

    return best_indices, best_scores  # both length k, CPU
#----------------------------------------------------------------------------

#--- Full Evaluation Function
#************************************************************************
def visualize_streaming_topk(model, query_loader, db_loader, db_dataset, device, metric="cosine", k=5, num_queries_to_show=8):
    shown = 0
    q_global_idx = 0
    with torch.no_grad():
        for q_imgs, _q_labels in query_loader:
            # Iterate items in the current batch so we can stop exactly at num_queries_to_show
            for i in range(q_imgs.size(0)):
                if shown >= num_queries_to_show:
                    return
                q_img = q_imgs[i:i+1]  # [1,C,H,W]
                q_emb = batch_to_emb(model, q_img, device).squeeze(0)  # [D]

                # Streaming top-k over the DB
                top_idx, top_scores = streaming_topk_for_query(q_emb, db_loader, model, device, k=k, metric=metric)

                # Plot
                plt.figure(figsize=(12, 3))
                plt.subplot(1, k+1, 1)
                plt.title(f"Query #{q_global_idx}")
                if CHANNELS == 3:
                    plt.imshow(q_img.squeeze().permute(1,2,0).cpu().numpy(), cmap="gray")
                else:
                    plt.imshow(q_img.squeeze().cpu().numpy(), cmap="gray")
                plt.axis("off")

                for t in range(k):
                    db_i = int(top_idx[t].item())
                    db_img, _ = db_dataset[db_i]  # loads just this one image from disk
                    plt.subplot(1, k+1, t+2)
                    plt.title(f"Top {t+1}\n{top_scores[t]:.3f}")
                    # db_img may be tensor; ensure CPU numpy for imshow
                    if torch.is_tensor(db_img):
                        if CHANNELS == 3:
                            im = db_img.squeeze().permute(1,2,0).cpu().numpy()
                        else:
                            im = db_img.squeeze().cpu().numpy()
                    else:
                        im = np.array(db_img)
                    plt.imshow(im, cmap="gray")
                    plt.axis("off")

                plt.tight_layout()
                plt.show()

                shown += 1
                q_global_idx += 1
            q_global_idx += (q_imgs.size(0) - i - 1)
#--------------------------------------------------











