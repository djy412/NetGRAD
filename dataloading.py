# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:36:56 2025

@author: djy41
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from skimage.io import imread
import numpy as np
import scipy.io
from PIL import Image, UnidentifiedImageError
from typing import Optional, Tuple
from pathlib import Path
import math

from typing import List, Optional, Dict, Union

def get_simple_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to tensor [0,1]
        # optionally normalize:
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    return transform

def get_grey_transform():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    return transform

def get_resize_transform():
    return transforms.Compose([
        transforms.Resize((240, 320)),   # ✅ half of 480×640
        transforms.ToTensor(),
    ])




# -------------------- Base utils --------------------
def _safe_open_rgb(path: str):
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        raise FileNotFoundError(f"Cannot open image: {path}") from e

def _resolve_img_path(row: pd.Series, root_dir: str) -> str:
    """ Prefer 'abs_path' if present; else join(root_dir, 'path'). """
    if "abs_path" in row and isinstance(row["abs_path"], str) and len(row["abs_path"]) > 0:
        return row["abs_path"]
    # fall back to relative path
    return os.path.join(root_dir, row["path"])

# =========================================================
# 1) Image+label dataset (for classification / supervised)
#    Expects CSV with at least: path[, abs_path], label
# =========================================================
class RobotCar_ImageLabel_Dataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None, drop_unlabeled: bool = True):
        """
        csv_file: filename (or full path). CSV must be comma-separated.
        Required columns: 'path' and 'label'. Optional: 'abs_path'.
        drop_unlabeled: if True, rows with label == -1 are dropped.
        """
        self.root_dir = root_dir
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(root_dir, csv_file)
        self.data = pd.read_csv(csv_path)  # comma-separated
        if drop_unlabeled and "label" in self.data.columns:
            self.data = self.data[self.data["label"] != -1].reset_index(drop=True)
        self.transform = transform

        # helpful sanity:
        missing_cols = [c for c in ["path", "label"] if c not in self.data.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols} in {csv_path}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        r = self.data.iloc[idx]
        img_path = _resolve_img_path(r, self.root_dir)
        img = _safe_open_rgb(img_path)

        label = int(r["label"]) if "label" in r and not (pd.isna(r["label"])) else -1
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================================================
# 2) Triplet dataset (anchor, positive, negative)
#    Expects CSV from our generator with columns:
#    anchor,positive,negative[, abs_anchor,abs_positive,abs_negative, cluster_id, pos_is_mirror]
# =========================================================
class RobotCar_Triplet_Train_Dataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        csv_file: 'triplets.csv' produced by the script.
        Uses absolute columns if they exist; otherwise joins with root_dir.
        Label = cluster_id if present, else -1.
        """
        self.root_dir = root_dir
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(root_dir, csv_file)
        self.ann = pd.read_csv(csv_path)  # comma-separated
        self.transform = transform

        # sanity
        need = ["anchor", "positive", "negative"]
        if any(c not in self.ann.columns for c in need):
            raise ValueError(f"Triplet CSV must have columns: {need}. Found: {list(self.ann.columns)}")

        # Optional columns we may leverage
        self.has_abs = all(c in self.ann.columns for c in ["abs_anchor", "abs_positive", "abs_negative"])
        self.has_cluster = "cluster_id" in self.ann.columns

    def __len__(self): return len(self.ann)

    def _resolve_triplet_paths(self, r: pd.Series) -> Tuple[str, str, str]:
        if self.has_abs:
            a = r["abs_anchor"]
            p = r["abs_positive"]
            n = r["abs_negative"]
            # Mirror positives may be under OUT_DIR/mirrors/... with abs paths present
        else:
            a = os.path.join(self.root_dir, r["anchor"])
            p = os.path.join(self.root_dir, r["positive"])
            n = os.path.join(self.root_dir, r["negative"])
        return a, p, n

    def __getitem__(self, idx: int):
        r = self.ann.iloc[idx]
        a_path, p_path, n_path = self._resolve_triplet_paths(r)

        a_img = _safe_open_rgb(a_path)
        p_img = _safe_open_rgb(p_path)
        n_img = _safe_open_rgb(n_path)

        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        label = int(r["cluster_id"]) if self.has_cluster and not pd.isna(r["cluster_id"]) else -1
        label = torch.tensor(label, dtype=torch.long)

        return a_img, p_img, n_img, label, idx

# =========================================================
# 3) Evaluation datasets (DB and Queries)
#    Expects CSVs with columns: path[, abs_path]
#    Labels are optional; we return -1 if missing.
# =========================================================
class RobotCar_Eval_Dataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None, return_meta: bool = True):
        """
        Generic eval dataset for DB or Query.
        Required columns: 'path' (and optionally 'abs_path').
        Will also pass through meta fields if present: condition, camera, x, y, label.
        """
        self.root_dir = root_dir
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(root_dir, csv_file)
        self.data = pd.read_csv(csv_path)
        if "path" not in self.data.columns:
            raise ValueError(f"Eval CSV must include 'path'. Columns found: {list(self.data.columns)}")
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        r = self.data.iloc[idx]
        img_path = _resolve_img_path(r, self.root_dir)
        img = _safe_open_rgb(img_path)
        if self.transform:
            img = self.transform(img)

        if not self.return_meta:
            return img

        # Build a compact meta dict (use NaN-safe conversions)
        def _get(name, default=None):
            return r[name] if name in r else default

        meta = {
            "path":    _get("path", ""),
            "abs_path": _get("abs_path", img_path),
            "condition": _get("condition", ""),
            "camera":    _get("camera", ""),
            "label":  int(_get("label", -1)) if not pd.isna(_get("label", -1)) else -1,
            "x": float(_get("x", math.nan)) if not pd.isna(_get("x", math.nan)) else math.nan,
            "y": float(_get("y", math.nan)) if not pd.isna(_get("y", math.nan)) else math.nan,
        }
        return img, meta

# =========================================================
# 4) Factory functions (mirroring your Pittsburgh style)
#    Adjust default root paths to your actual folders.
# =========================================================
def get_RobotCar_Train_dataloaders(
    root_dir=r"C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar Seasons/Dataset/",
    csv_file="train_large.csv",                 # <- a CSV of (path,label); you can create this from your clustered train rows
    batch_size=256, num_workers=8, drop_unlabeled=True
):
    transform = get_simple_transform()    # use your transform
    ds = RobotCar_ImageLabel_Dataset(csv_file=csv_file, root_dir=root_dir,
                                     transform=transform, drop_unlabeled=drop_unlabeled)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=num_workers)
    return loader, ds

def get_RobotCar_Triplet_Train_dataloaders(
    root_dir=r"C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar Seasons/Dataset/",
    csv_file="triplets_large.csv",
    batch_size=64, num_workers=4
):
    transform = get_simple_transform()    # use your transform
    ds = RobotCar_Triplet_Train_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers)
    return loader, ds

def get_RobotCar_Eval_db_dataloaders(
    root_dir=r"C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar Seasons/Dataset/",
    csv_file="database.csv",
    batch_size=256, num_workers=8
):
    transform = get_simple_transform()
    ds = RobotCar_Eval_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform, return_meta=True)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    return loader, ds

def get_RobotCar_Eval_query_dataloaders(
    root_dir=r"C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar Seasons/Dataset/",
    csv_file="queries.csv",
    batch_size=256, num_workers=8
):
    transform = get_simple_transform()
    ds = RobotCar_Eval_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform, return_meta=True)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    return loader, ds





#*****************************************************************************
#--- Creating the RobotCar 25 Training dataset    
#*****************************************************************************
class RobotCar_25_Train_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Important: use sep=";" if CSV is "filename;label"
        self.data = pd.read_csv(root_dir + csv_file, sep=";")  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 0])   # filename as string
        img_path = os.path.join(self.root_dir, img_name)
        
        image = _safe_open_rgb(img_path)#.convert("L")
        #label = torch.tensor(int(self.data.iloc[idx, 1]))  # label
        
        if self.transform:
            image = self.transform(image)

        return image, idx

#*****************************************************************************
#--- Loading the Training RobotCar 25 Street dataset    
#***************************************************************************** 
def get_RobotCar_25_Train_dataloaders(batch_size=256, num_workers=8):
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar/2014-11-25-09-18-32/train/"
    transform = get_grey_transform()  

    train_dataset = RobotCar_25_Train_Dataset(csv_file="data.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Creating the RobotCar 25 Triplet Train dataset 
#*****************************************************************************
class RobotCar_25_Triplet_Train_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(root_dir+csv_file,sep=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,0])
        anchor = _safe_open_rgb(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,1])
        positive = _safe_open_rgb(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,2])
        negative = _safe_open_rgb(img_path)
        label =torch.tensor(int(self.annotations.iloc[idx,3]))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, label, idx
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training RobotCar 25 dataset    
#***************************************************************************** 
def get_RobotCar_25_Triplet_Train_dataloaders(batch_size=256, num_workers=8):
    """Pittsburg30k dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar/2014-11-25-09-18-32/train/"

    transform = get_grey_transform()  
    train_dataset = RobotCar_25_Triplet_Train_Dataset(csv_file = "triplets.csv", root_dir = root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=4)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Creating the RobotCar 25 Training dataset    
#*****************************************************************************
class RobotCar_25_Eval_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Important: use sep=";" if CSV is "filename;label"
        self.data = pd.read_csv(root_dir + csv_file, sep=";")  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 0])   # filename as string
        img_path = os.path.join(self.root_dir, img_name)
        
        image = _safe_open_rgb(img_path)#.convert("L")
        label = torch.tensor(int(self.data.iloc[idx, 1]))  # label
        
        if self.transform:
            image = self.transform(image)

        return image, label
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training db RobotCar 25 dataset    
#***************************************************************************** 
def get_RobotCar_25_Eval_db_dataloaders(batch_size=256, num_workers=8):    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar/2014-11-28-12-07-13/eval/"
    transform = get_grey_transform()  

    train_dataset = RobotCar_25_Eval_Dataset(csv_file="database.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training query RobotCar 25 dataset    
#***************************************************************************** 
def get_RobotCar_25_Eval_query_dataloaders(batch_size=256, num_workers=8):   
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar/2014-11-28-12-07-13/eval/"
    transform = get_grey_transform()  

    train_dataset = RobotCar_25_Eval_Dataset(csv_file="query.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Creating the RobotCar 25 Training dataset    
#*****************************************************************************
class RobotCar_Seq_Eval_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Important: use sep=";" if CSV is "filename;label"
        self.data = pd.read_csv(root_dir + csv_file, sep=";")  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 0])   # filename as string
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("L")
        label = torch.tensor(int(self.data.iloc[idx, 1]))  # label
        revisit = torch.tensor(int(self.data.iloc[idx, 2]))  # 0 for not revisit, 1 for revisited  
        
        if self.transform:
            image = self.transform(image)

        return image, label, revisit
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training db RobotCar 25 dataset    
#***************************************************************************** 
def get_RobotCar_Seq_Eval_db_dataloaders(batch_size=256, num_workers=8):    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/RobotCar/2014-11-28-12-07-13/eval/"
    transform = get_grey_transform()  

    train_dataset = RobotCar_Seq_Eval_Dataset(csv_file="database.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************






#*****************************************************************************
#--- Creating the Pittsburgh30k Training dataset    
#*****************************************************************************
class Pittsburg30k_Train_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Important: use sep=";" if CSV is "filename;label"
        self.data = pd.read_csv(root_dir + csv_file, sep=";")  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 0])   # filename as string
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.data.iloc[idx, 1]))  # label
        
        if self.transform:
            image = self.transform(image)

        return image, label
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training Pittsburg30k Street dataset    
#***************************************************************************** 
def get_Pittsburg30k_Train_dataloaders(batch_size=256, num_workers=8):
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Pittsburgh30k_train/"
    transform = get_simple_transform()  

    train_dataset = Pittsburg30k_Train_Dataset(csv_file="train.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Creating the Pittsburg30k Triplet Train dataset 
#*****************************************************************************
class Pittsburg30k_Triplet_Train_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(root_dir+csv_file,sep=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,2])
        negative = imread(img_path)
        label =torch.tensor(int(self.annotations.iloc[idx,3]))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, label, idx
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training Pittsburg30k dataset    
#***************************************************************************** 
def get_Pittsburg30k_Triplet_Train_dataloaders(batch_size=256, num_workers=8):
    """Pittsburg30k dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Pittsburgh30k_Triplet_Train/"

    transform = get_simple_transform()  
    train_dataset = Pittsburg30k_Triplet_Train_Dataset(csv_file = "triplets.csv", root_dir = root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training db Pittsburg30k dataset    
#***************************************************************************** 
def get_Pittsburg30k_Eval_db_dataloaders(batch_size=256, num_workers=8):    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Pittsburgh30k_test/"
    transform = get_simple_transform()  

    train_dataset = Pittsburg30k_Train_Dataset(csv_file="database.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training query Pittsburg30k dataset    
#***************************************************************************** 
def get_Pittsburg30k_Eval_query_dataloaders(batch_size=256, num_workers=8):   
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Pittsburgh30k_test/"
    transform = get_simple_transform()  

    train_dataset = Pittsburg30k_Train_Dataset(csv_file="query.csv", root_dir=root_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers)
    
    return train_loader, train_dataset
#*****************************************************************************

# *****************************************************************************
# Generic CSV dataset with optional coords (eval/train-by-csv)
# *****************************************************************************
class PittsburghCSV(Dataset):
    """
    Expects a CSV with at least 'filename;label'.
    If the CSV also contains 'panoid;lat;lon', they will be parsed and exposed.

    Returns:
      - default:  (image_tensor, label_tensor)
      - if return_meta=True: (image_tensor, label_tensor, meta_dict)
            meta_dict keys: 'path' (abs), 'rel' (relative), 'panoid' (int, if present),
                            'lat' (float, if present), 'lon' (float, if present), 'index' (int)
    """
    def __init__(self, root_dir, csv_file, transform=None, return_meta: bool=False):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, csv_file)
        self.data = pd.read_csv(self.csv_path, sep=";")
        self.transform = transform
        self.return_meta = return_meta

        # Required columns
        if self.data.shape[1] < 2:
            raise ValueError(f"CSV must have at least 2 columns (filename;label). Got {self.data.shape[1]}")

        # Optional coords
        cols = {c.lower(): c for c in self.data.columns}
        self.has_coords = all(k in cols for k in ["panoid", "lat", "lon"])

        # Cache numpy arrays for fast access in eval code
        # (these are aligned with dataset order and safe even if you don't return_meta)
        self._filenames = self.data.iloc[:, 0].astype(str).str.replace("\\", "/").to_numpy()
        self._labels_np = self.data.iloc[:, 1].astype(int).to_numpy()

        if self.has_coords:
            self._panoids_np = self.data[cols["panoid"]].astype(int).to_numpy()
            self._lat_np     = self.data[cols["lat"]].astype(float).to_numpy()
            self._lon_np     = self.data[cols["lon"]].astype(float).to_numpy()
        else:
            self._panoids_np = None
            self._lat_np     = None
            self._lon_np     = None

    # ---- Convenience getters for eval ----
    def labels_np(self):        return self._labels_np
    def panoids_np(self):       return self._panoids_np
    def coords_deg_np(self):
        """Returns Nx2 [lat, lon] or None if coords not present."""
        if self._lat_np is None: return None
        return np.stack([self._lat_np, self._lon_np], axis=1)
    def relpaths(self):         return self._filenames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel = self._filenames[idx]
        img_path = os.path.join(self.root_dir, rel)
        img = Image.open(img_path).convert("RGB")

        label = torch.tensor(self._labels_np[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        if not self.return_meta:
            return img, label

        meta = {"path": img_path,
                "rel": rel,
                "index": int(idx)}
        if self.has_coords:
            meta.update({
                "panoid": int(self._panoids_np[idx]),
                "lat": float(self._lat_np[idx]),
                "lon": float(self._lon_np[idx]),
            })
        return img, label, meta
#*****************************************************************************

# ---------------------------------------------
# Evaluation loaders (5m-clustered labels + coords available)
# ---------------------------------------------
def get_Pittsburgh30k_Eval_db_dataloaders_5m(batch_size=256, num_workers=8, return_meta=True):
    root_dir = r"C:/Users/djy41/Desktop/PhD Work/Datasets/Pittsburgh30k_Test_5m"
    transform = get_simple_transform()  # or get_resize_transform()
    ds = PittsburghCSV(root_dir=root_dir, csv_file="database.csv",
                       transform=transform, return_meta=return_meta)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                    num_workers=num_workers, pin_memory=True)
    return dl, ds

def get_Pittsburgh30k_Eval_query_dataloaders_5m(batch_size=256, num_workers=8, return_meta=True):
    root_dir = r"C:/Users/djy41/Desktop/PhD Work/Datasets/Pittsburgh30k_Test_5m"
    transform = get_simple_transform()  # or get_resize_transform()
    ds = PittsburghCSV(root_dir=root_dir, csv_file="query.csv",
                       transform=transform, return_meta=return_meta)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                    num_workers=num_workers, pin_memory=True)
    return dl, ds








#*****************************************************************************
#--- Creating the Mapillary Street dataset    
#*****************************************************************************
class Mapillary_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the .csv file containing filenames.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to apply on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # first column has filenames
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")  # ensure RGB
        if self.transform:
            image = self.transform(image)

        return image


#*****************************************************************************
#--- Loading the Training Mapillary Street dataset    
#***************************************************************************** 
def get_Mapillary_dataloaders(batch_size=256, num_workers=8):
    """Mapillary dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Mapillary Streets1/training/images"
    csv_file = os.path.join(root_dir, "data.csv")

    transform = get_simple_transform()  
    train_dataset = Mapillary_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=4)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading Testing the Mapillary Street dataset    
#***************************************************************************** 
def get_Test_Mapillary_dataloaders(batch_size=256, num_workers=8):
    """Mapillary dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Mapillary Streets1/test/images"
    csv_file = os.path.join(root_dir, "data.csv")

    transform = get_resize_transform()  
    test_dataset = Mapillary_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=4)
    
    return test_loader, test_dataset
#*****************************************************************************



#*****************************************************************************
#--- Creating the Mapillary AND Tokyo dataset    
#*****************************************************************************
class Map_Tokyo_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the .csv file containing filenames.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to apply on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # first column has filenames
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")  # ensure RGB
        if self.transform:
            image = self.transform(image)

        return image, idx


#*****************************************************************************
#--- Loading the Training Mapillary Street dataset    
#***************************************************************************** 
def get_Map_Tokyo_dataloaders(batch_size=256, num_workers=8):
    """Mapillary dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Mapillary_Tokyo"
    csv_file = os.path.join(root_dir, "data.csv")

    transform = get_simple_transform()  
    train_dataset = Map_Tokyo_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    
    return train_loader, train_dataset
#*****************************************************************************


#*****************************************************************************
#--- Creating the Tokyo 24/7 Train dataset 
#*****************************************************************************
class TokyoPlacesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Important: use sep=";" if CSV is "filename;label"
        self.data = pd.read_csv(root_dir + csv_file, sep=";")  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx, 0])   # filename as string
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.data.iloc[idx, 2]))  # label
        
        if self.transform:
            image = self.transform(image)

        return image, label
#*****************************************************************************





#*****************************************************************************
#--- Loading the Training Tokyo 24/7 Street dataset    
#***************************************************************************** 
def get_Tokyo_Train_dataloaders(batch_size=256, num_workers=8):
    """Tokyo 24/7 dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Tokyo 24_7/train/"

    transform = get_simple_transform()  
    train_dataset = TokyoPlacesDataset(csv_file="train.csv", root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=4)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Creating the Tokyo 24/7 Triplet Train dataset 
#*****************************************************************************
class Tokyo_Triplet_Train_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(root_dir+csv_file,sep=';', engine='python')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,1])
        positive = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,2])
        negative = imread(img_path)
        label =torch.tensor(int(self.annotations.iloc[idx,3]))

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, label, idx
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training Tokyo 24/7 Street dataset    
#***************************************************************************** 
def get_Tokyo_Triplet_Train_dataloaders(batch_size=256, num_workers=8):
    """Tokyo 24/7 dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Tokyo 24_7/train/"

    transform = get_simple_transform()  
    train_dataset = Tokyo_Triplet_Train_Dataset(csv_file = "triplets.csv", root_dir = root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    
    return train_loader, train_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the Training Tokyo 24/7 Street dataset    
#***************************************************************************** 
def get_Tokyo_Triplet_Test_dataloaders(batch_size=256, num_workers=8):
    """Tokyo 24/7 dataloader with (640, 480) images."""    
    root_dir = "C:/Users/djy41/Desktop/PhD Work/Datasets/Tokyo 24_7/train/"

    transform = get_simple_transform()  
    test_dataset = Tokyo_Triplet_Train_Dataset(csv_file = "triplets.csv", root_dir = root_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    
    return test_loader, test_dataset
#*****************************************************************************

#*****************************************************************************
#--- Creating the Tokyo 24/7 Evaluation Dataset 
#*****************************************************************************
class Tokyo_Eval_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(root_dir+csv_file,sep=';', engine='python')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,0])
        img = imread(img_path)
        label =torch.tensor(int(self.annotations.iloc[idx,1]))

        if self.transform:
            img = self.transform(img)

        return img, label
#*****************************************************************************

#*****************************************************************************
class TokyoEvalDatasetMultiRoot(Dataset):
    def __init__(
        self,
        csv_file: Union[str, Path],
        root_dirs: List[Union[str, Path]],           # NEW: multiple roots
        transform=None,
        sep: Optional[str] = None,                    # auto-detect delimiter by default
        filename_col: str = "filename",
        label_col: str = "label",
        label_map: Optional[Dict[str, int]] = None,   # optional: map string labels -> int IDs
        strict_exists: bool = True                    # if True, raise if an image can't be found
    ):
        # Normalize roots
        self.root_dirs = [Path(p) for p in root_dirs]
        self.transform = transform
        self.filename_col = filename_col
        self.label_col = label_col
        self.label_map = label_map
        self.strict_exists = strict_exists

        # Load CSV robustly (auto sep, utf-8-sig to strip BOM)
        csv_path = Path(csv_file)
        if not csv_path.is_absolute() and len(self.root_dirs) == 1:
            csv_path = self.root_dirs[0] / csv_path
        self.annotations = pd.read_csv(csv_path, sep=sep, dtype=str, encoding="utf-8-sig", engine='python')
        # Normalize column names
        self.annotations.columns = [c.strip() for c in self.annotations.columns]
        if self.filename_col not in self.annotations.columns:
            raise KeyError(f"CSV missing '{self.filename_col}' column. Found: {list(self.annotations.columns)}")
        if self.label_col not in self.annotations.columns:
            raise KeyError(f"CSV missing '{self.label_col}' column. Found: {list(self.annotations.columns)}")

        # Basic cleanup
        self.annotations[self.filename_col] = self.annotations[self.filename_col].str.strip()
        self.annotations[self.label_col]    = self.annotations[self.label_col].str.strip()

        # Pre-resolve image paths for speed
        self._resolved_paths: List[Optional[Path]] = []
        for fname in self.annotations[self.filename_col].tolist():
            p = self._resolve_image_path(fname)
            if p is None and self.strict_exists:
                raise FileNotFoundError(f"Could not find image '{fname}' in any of: {self.root_dirs}")
            self._resolved_paths.append(p)

    def _resolve_image_path(self, fname: str) -> Optional[Path]:
        """
        Try to resolve the image file:
        - If absolute path: return if exists.
        - Try every root_dir / fname
        - Also try (root_dir / Path(fname).name) to tolerate CSV that stores just basenames.
        """
        fpath = Path(fname)
        # Absolute path?
        if fpath.is_absolute():
            return fpath if fpath.exists() else None

        # Try each root with the given relative path
        for root in self.root_dirs:
            cand = root / fpath
            if cand.exists():
                return cand

        # If CSV has subdirs like 'train/0001.jpg' but roots also point to those dirs,
        # or if CSV only has basenames, try basename under each root
        base = fpath.name
        for root in self.root_dirs:
            cand = root / base
            if cand.exists():
                return cand

        return None  # not found

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = self._resolved_paths[idx]
        if img_path is None:
            # Lazy fallback if strict_exists=False: try resolving again (in case FS changed)
            img_path = self._resolve_image_path(row[self.filename_col])
            if img_path is None:
                raise FileNotFoundError(f"Image not found for row {idx}: {row[self.filename_col]}")

        # Load image (RGB) with PIL (works well with torchvision transforms)
        with Image.open(img_path) as im:
            img = im.convert("RGB")

        # Label handling
        label_raw = row[self.label_col]
        if self.label_map is not None:
            # map string -> int id
            if label_raw not in self.label_map:
                raise KeyError(f"Label '{label_raw}' not in provided label_map")
            label = torch.tensor(self.label_map[label_raw], dtype=torch.long)
        else:
            # try int; if fails, keep as string tensor (or you can raise)
            try:
                label = torch.tensor(int(label_raw), dtype=torch.long)
            except ValueError:
                # keep original string (some pipelines prefer numeric; if so, pass label_map)
                label = label_raw

        if self.transform:
            img = self.transform(img)

        return img, label
#*****************************************************************************


#*****************************************************************************
#--- Loading the Eval Tokyo 24/7 Street dataset    
#***************************************************************************** 
def get_Tokyo_Eval_dataloaders(batch_size=256,num_workers=8,
    dataset_root=r"C:\Users\djy41\Desktop\PhD Work\Datasets\Tokyo 24_7",
    db_csv_name="database.csv",
    use_queries=False,
    shuffle=False
):
    test_dir  = Path(dataset_root) / "test"
    train_dir = Path(dataset_root) / "train"
    csv_file  = test_dir / db_csv_name

    transform = get_simple_transform()

    test_dataset = TokyoEvalDatasetMultiRoot(
        csv_file=csv_file,
        root_dirs=[test_dir, train_dir],  # search both
        transform=transform,
        sep=None,
        filename_col="filename",
        label_col="label",             # << integer column
        label_map=None,                   # no mapping needed
        strict_exists=True
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)
    return test_loader, test_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the Eval Tokyo 24/7 Street queries
#***************************************************************************** 
def get_Tokyo_Query_dataloaders(batch_size=256,num_workers=8,
    dataset_root=r"C:\Users\djy41\Desktop\PhD Work\Datasets\Tokyo 24_7",
    db_csv_name="queries.csv",
    use_queries=False,
    shuffle=False
):
    test_dir  = Path(dataset_root) / "test"
    train_dir = Path(dataset_root) / "train"
    csv_file  = test_dir / db_csv_name

    transform = get_simple_transform()

    test_dataset = TokyoEvalDatasetMultiRoot(
        csv_file=csv_file,
        root_dirs=[test_dir, train_dir],  # search both
        transform=transform,
        sep=None,
        filename_col="filename",
        label_col="label",             # << integer column
        label_map=None,                   # no mapping needed
        strict_exists=True
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=False)
    return test_loader, test_dataset
#*****************************************************************************


class MULTI_STL_10(Dataset):  
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MULTI_STL-10.mat')['X3'].astype(np.float32)
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x1 = x1.transpose(2,0,1) #Reorder to be in the format wanted (3, 96, 96)
        x2 = self.V2[idx]
        x2 = x2.transpose(2,0,1) 
        x3 = self.V3[idx]
        x3 = x3.transpose(2,0,1) 
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    cwd = os.getcwd()
    if dataset == "MULTI-STL-10": 
        dataset = MULTI_STL_10('./data/')
        dims = [27648, 27648]
        view = 2
        class_num = 10
        data_size = 5000   
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num



# ------------------------
# Usage
# ------------------------
if __name__ == "__main__":
    # root_dir = "C:/Users/djy41/Desktop/PhD/Datasets/Mapillary Streets1/training/images"
    # csv_file = os.path.join(root_dir, "data.csv")

    # transform = get_simple_transform()  
    # dataset = Mapillary_Dataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    dataloader, train_dataset = get_Pittsburg30k_Train_dataloaders(batch_size=64, num_workers=2)

    # test one batch
    for i, (img,labels) in enumerate(dataloader):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(img[0].permute(1, 2, 0).cpu())
        plt.title("img")
        plt.show()

        print("Anchor batch shape:", img.shape)  
        print("Labels: ", labels[0].item())
        break

