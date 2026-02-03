"""

Notes
- DINOv2 is loaded via torch.hub:
    torch.hub.load("facebookresearch/dinov2", DINO_NAME)
  This requires internet on first run, then uses the local cache.

"""

import os
import math
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode



# 1) CONFIG


BASE_DIR = r"/home/milkyway/MaungMyintSoe/data/raw_data"

RUN_NAME = "LOSO_Wrist_Physio_NoAct_DINOv2"
RESULTS_DIR = os.path.join(BASE_DIR, f"results_{RUN_NAME}")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EMBED_DIR = os.path.join(RESULTS_DIR, "embeddings")
for p in [CHECKPOINT_DIR, PLOTS_DIR, EMBED_DIR]:
    os.makedirs(p, exist_ok=True)

# Data
SUBJECT_IDS = [f"Subject{str(i).zfill(2)}" for i in range(1, 11)]
MANIFEST_SUFFIX = "_multiview_manifest_5s.csv"

# Views (robust: edit these only)
VIEW_ORDER = ["Wrist_Physio"]  
VIEW_TO_INDEX = {v: i for i, v in enumerate(VIEW_ORDER)}
N_VIEWS = len(VIEW_ORDER)

# Missing-view tolerance
MIN_VIEWS_PER_WINDOW = 1  # set 1 to tolerate missing views; set =N_VIEWS for strict fusion

# ---- View-column helpers (robust to any N_VIEWS) ----
REL_COLS = [f"rel_path_v{i}" for i in range(N_VIEWS)]
MASK_COLS = [f"mask_v{i}" for i in range(N_VIEWS)]

# Ensure MIN_VIEWS_PER_WINDOW is valid
if MIN_VIEWS_PER_WINDOW > N_VIEWS:
    print(f"[WARN] MIN_VIEWS_PER_WINDOW={MIN_VIEWS_PER_WINDOW} > N_VIEWS={N_VIEWS}. Setting MIN_VIEWS_PER_WINDOW={N_VIEWS}.")
    MIN_VIEWS_PER_WINDOW = N_VIEWS

# Exclude baseline activity codes
EXCLUDE_ACTIVITY_CODES = {22, 23}

# Groud truth label column
TRAIN_LABEL_COL = "gt_supervisor_Wkg"

# Image size 
IMAGE_SIZE = 224

# DINOv2 backbone
DINO_NAME = "dinov2_vitb14"  # recommended default
FREEZE_DINO = False         # True = feature extractor + MLP; False = fine-tune
LEARNING_RATE_DINO = 1e-5   # used only if FREEZE_DINO=False

# Training
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 6 # set to None to disable early stopping
BATCH_SIZE = 105
LEARNING_RATE_HEAD = 3e-4
WEIGHT_DECAY = 1e-3
RANDOM_SEED = 42
NUM_WORKERS = 4

# Inner validation split mode
INNER_VAL_MODE = "subject"  # options: "subject", "window"
N_VAL_SUBJECTS = 1          # used when INNER_VAL_MODE="subject"

# Residual learning over train-only ByActMean
USE_ACTIVITY_RESIDUAL = False

# Residual calibration: y = mu_act + alpha * resid_pred + beta (fit on VAL only)
CALIBRATE_RESIDUAL = False

# Resume per fold
RESUME_TRAINING = True

# LOSO controls
RUN_ALL_FOLDS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 2) UTILITIES


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}

def _normalize_rel_path(rel_path: str) -> str:
    if not isinstance(rel_path, str):
        return rel_path
    return rel_path.replace("\\", "/").strip()

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _meta_item(meta, key: str, i: int):
    v = meta[key]
    if torch.is_tensor(v):
        return v[i].item()
    return v[i]

def fit_alpha_beta(resid_true: np.ndarray, resid_pred: np.ndarray) -> Tuple[float, float]:
    """
    Fit resid_true ≈ alpha * resid_pred + beta
    """
    resid_true = np.asarray(resid_true, dtype=np.float64)
    resid_pred = np.asarray(resid_pred, dtype=np.float64)
    X = np.vstack([resid_pred, np.ones_like(resid_pred)]).T
    alpha, beta = np.linalg.lstsq(X, resid_true, rcond=None)[0]
    return float(alpha), float(beta)

def collate_keep_meta(batch):
    # batch is list of tuples: (images, mask, y, mu, meta)
    images, mask, y, mu, meta = zip(*batch)
    images = torch.stack(images, dim=0)
    mask = torch.stack(mask, dim=0)
    y = torch.stack(y, dim=0)
    mu = torch.stack(mu, dim=0)
    meta = list(meta)  # keep as-is (list of dicts)
    return images, mask, y, mu, meta


def resolve_dinov2_repo_dir() -> str:
    """
    Resolve the local torch.hub cache directory for the DINOv2 repo.
    This avoids any network calls (GitHub) during loading.
    """
    hub_dir = torch.hub.get_dir()
    candidates = [
        os.path.join(hub_dir, "facebookresearch_dinov2_main"),
        os.path.join(hub_dir, "facebookresearch_dinov2_master"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError(
        "DINOv2 torch.hub repo not found in cache. Looked in:\n"
        f"  {candidates[0]}\n"
        f"  {candidates[1]}\n"
        f"Current torch.hub.get_dir()={hub_dir}\n"
        "Fix: run once with internet to populate the cache, or manually clone the repo into the hub cache."
    )

def build_dinov2_preprocess(image_size: int):
    return T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])



# 3) LOAD + BUILD WINDOW-LEVEL TABLE (ONE ROW = ONE WINDOW)


def load_all_rows(base_dir: str, subject_ids: List[str]) -> pd.DataFrame:
    all_dfs = []
    for sid in subject_ids:
        root = os.path.join(base_dir, f"{sid}_multiview_360_5s")
        manifest_path = os.path.join(root, f"{sid}{MANIFEST_SUFFIX}")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing manifest for {sid}: {manifest_path}")

        df = pd.read_csv(manifest_path)
        print(f"[Manifest] Loaded {sid}: {len(df)} rows")

        required = [
            "subject_id", "activity_name", "window_idx",
            "view_name", "image_rel_path", "data_status",
            TRAIN_LABEL_COL, "activity_code"
        ]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"[{sid}] Missing required column '{c}' in {manifest_path}")

        df = df[df["data_status"] == "VALID"].copy()
        df = df[df["view_name"].isin(VIEW_ORDER)].copy()
        df["image_rel_path"] = df["image_rel_path"].apply(_normalize_rel_path)

        df["activity_code"] = df["activity_code"].apply(safe_int)
        df = df[df["activity_code"].notna()].copy()
        df = df[~df["activity_code"].isin(EXCLUDE_ACTIVITY_CODES)].copy()

        df[TRAIN_LABEL_COL] = pd.to_numeric(df[TRAIN_LABEL_COL], errors="coerce")
        df = df[df[TRAIN_LABEL_COL].notna()].copy()

        df["subject_id"] = df["subject_id"].fillna(sid).astype(str)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"[Manifest] Combined rows: {len(df_all)}")
    return df_all

def build_window_table(df_rows: pd.DataFrame) -> pd.DataFrame:
    df_rows = df_rows.copy()
    df_rows["group_id"] = (
        df_rows["subject_id"].astype(str) + "||" +
        df_rows["activity_name"].astype(str) + "||" +
        df_rows["window_idx"].astype(str)
    )

    window_rows = []
    for gid, g in df_rows.groupby("group_id"):
        r0 = g.iloc[0]
        y = float(r0[TRAIN_LABEL_COL])
        act_code = safe_int(r0["activity_code"])

        if "window_center_s" in g.columns and pd.notna(r0.get("window_center_s", np.nan)):
            wcs = float(r0["window_center_s"])
        else:
            wcs = np.nan

        rel_paths = [None] * N_VIEWS
        mask = [0] * N_VIEWS

        for _, rr in g.iterrows():
            vname = rr["view_name"]
            if vname in VIEW_TO_INDEX:
                vi = VIEW_TO_INDEX[vname]
                rel_paths[vi] = _normalize_rel_path(rr["image_rel_path"])
                mask[vi] = 1

        n_avail = int(sum(mask))

        row_out = {
            "group_id": gid,
            "subject_id": str(r0["subject_id"]),
            "activity_name": str(r0["activity_name"]),
            "activity_code": act_code,
            "window_idx": int(r0["window_idx"]),
            "window_center_s": wcs,
            "y": y,
            "n_views_available": n_avail,
        }

        for vi in range(N_VIEWS):
            row_out[f"rel_path_v{vi}"] = rel_paths[vi]
            row_out[f"mask_v{vi}"] = mask[vi]

        window_rows.append(row_out)

    df_win = pd.DataFrame(window_rows)
    df_win = df_win[df_win["n_views_available"] >= MIN_VIEWS_PER_WINDOW].copy().reset_index(drop=True)

    print(f"[Windows] Window-level samples: {len(df_win)}")
    print(df_win["n_views_available"].value_counts().sort_index())

    return df_win



# 4) TRAIN-ONLY ACTIVITY MEAN (for residual)


def build_baselines(train_df_win: pd.DataFrame):
    """
    Baselines use ONLY train labels.
    """
    y_train = train_df_win["y"].values.astype(np.float64)
    global_mean = float(np.mean(y_train))
    by_code_mean = train_df_win.groupby("activity_code")["y"].mean().to_dict()
    return global_mean, by_code_mean

def baseline_predict_global(n: int, global_mean: float) -> np.ndarray:
    return np.full(shape=(n,), fill_value=global_mean, dtype=np.float64)

def baseline_predict_by_code(act_codes: np.ndarray, global_mean: float, by_code_mean: dict) -> np.ndarray:
    out = np.zeros(len(act_codes), dtype=np.float64)
    for i, c in enumerate(act_codes):
        out[i] = by_code_mean.get(int(c), global_mean)
    return out



# 5) DATASET (V-SLOT IMAGE TENSOR + MASK + TARGET + MU_ACT)


class WindowFusionDataset(Dataset):
    """
    One item = one window.

    Returns:
      images: (V, 3, IMAGE_SIZE, IMAGE_SIZE) with zeros for missing views
      mask:   (V,)
      y_tgt:  scalar (y - mu_act) if residual else y
      mu_act: scalar (mu(activity) from TRAIN set) if residual else 0
      meta:   dict
    """
    def __init__(
        self,
        df_win: pd.DataFrame,
        base_dir: str,
        preprocess,
        image_size: int,
        use_activity_residual: bool = False,
        by_code_mean: Optional[dict] = None,
        global_mean: float = 0.0
    ):
        self.df = df_win.reset_index(drop=True)
        self.base_dir = base_dir
        self.preprocess = preprocess
        self.image_size = int(image_size)
        self.use_activity_residual = bool(use_activity_residual)
        self.by_code_mean = by_code_mean or {}
        self.global_mean = float(global_mean)

    def __len__(self):
        return len(self.df)

    def _load_one_view(self, subject_id: str, rel_path: str) -> Optional[Tuple[torch.Tensor, str]]:
        if not isinstance(rel_path, str) or len(rel_path) == 0:
            return None
        subject_root = os.path.join(self.base_dir, f"{subject_id}_multiview_360_5s")
        rel_path = _normalize_rel_path(rel_path)
        img_path = os.path.join(subject_root, rel_path)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(img)
            return img_tensor, img_path
        except Exception:
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["subject_id"]

        view_paths = [row[c] for c in REL_COLS]
        view_mask = [row[c] for c in MASK_COLS]

        images = []
        abs_paths = [None] * N_VIEWS
        effective_mask = [int(m) for m in view_mask]

        for vi in range(N_VIEWS):
            if effective_mask[vi] == 1:
                out = self._load_one_view(subject_id, view_paths[vi])
                if out is None:
                    images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
                    abs_paths[vi] = None
                    effective_mask[vi] = 0
                else:
                    img_tensor, img_abs = out
                    images.append(img_tensor)
                    abs_paths[vi] = img_abs
            else:
                images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
                abs_paths[vi] = None

        images = torch.stack(images, dim=0)  # (V,3,H,W)
        mask = torch.tensor(effective_mask, dtype=torch.float32)

        y = float(row["y"])
        act_code = int(row["activity_code"]) if pd.notna(row["activity_code"]) else -1

        if self.use_activity_residual:
            mu_act = float(self.by_code_mean.get(act_code, self.global_mean))
            y_tgt = float(y - mu_act)
        else:
            mu_act = 0.0
            y_tgt = float(y)

        y_tgt_t = torch.tensor(y_tgt, dtype=torch.float32)
        mu_act_t = torch.tensor(mu_act, dtype=torch.float32)

        meta = {
            "row_idx": int(idx),
            "group_id": str(row["group_id"]),
            "subject_id": str(row["subject_id"]),
            "activity_name": str(row["activity_name"]),
            "activity_code": act_code,
            "window_idx": int(row["window_idx"]),
            "window_center_s": float(row["window_center_s"]) if row["window_center_s"] == row["window_center_s"] else np.nan,
            "n_views_available": int(sum(effective_mask)),
            "mu_act": float(mu_act),
            "use_activity_residual": int(self.use_activity_residual),
        }
        for vi in range(N_VIEWS):
            meta[f"image_path_v{vi}"] = abs_paths[vi]

        return images, mask, y_tgt_t, mu_act_t, meta



# 6) MODEL: DINOv2 PER VIEW + MASKED MEAN FUSION + MLP


class DinoV2FusionRegressor(nn.Module):
    def __init__(self, dino_model, freeze_dino: bool = False, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.dino = dino_model
        self.freeze_dino = freeze_dino

        embed_dim = getattr(dino_model, "embed_dim", None) or getattr(dino_model, "num_features", None)
        if embed_dim is None:
            embed_dim = 768  # safe default for vitb

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 forward_features typically returns dict with CLS token.
        if hasattr(self.dino, "forward_features"):
            out = self.dino.forward_features(x)
            if isinstance(out, dict):
                if "x_norm_clstoken" in out:
                    feats = out["x_norm_clstoken"]
                elif "x_prenorm" in out:
                    feats = out["x_prenorm"][:, 0]
                elif "x_norm_patchtokens" in out:
                    feats = out["x_norm_patchtokens"].mean(dim=1)
                else:
                    raise RuntimeError(f"Unexpected DINOv2 forward_features keys: {list(out.keys())}")
            else:
                feats = out
        else:
            feats = self.dino(x)

        return feats.float()

    def encode_views(self, images_bv: torch.Tensor) -> torch.Tensor:
        if self.freeze_dino:
            with torch.no_grad():
                return self._encode(images_bv)
        return self._encode(images_bv)

    def forward(self, images: torch.Tensor, mask: torch.Tensor):
        B, V, C, H, W = images.shape
        images_flat = images.view(B * V, C, H, W)

        feats_flat = self.encode_views(images_flat)  # (B*V, D)
        D = feats_flat.shape[-1]
        feats = feats_flat.view(B, V, D)             # (B,V,D)

        m = mask.unsqueeze(-1)                       # (B,V,1)
        feats_masked = feats * m
        denom = m.sum(dim=1).clamp(min=1.0)          # (B,1)
        fused = feats_masked.sum(dim=1) / denom      # (B,D)

        pred = self.mlp(fused).squeeze(-1)           # (B,)
        return pred, fused



# 7) TRAIN / EVAL (metrics in ORIGINAL y space)

def _to_y_space(y_tgt: torch.Tensor, mu_act: torch.Tensor, use_resid: bool) -> torch.Tensor:
    return (y_tgt + mu_act) if use_resid else y_tgt

def train_one_epoch(model, loader, optimizer, criterion, epoch, num_epochs, use_resid: bool):
    model.train()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    num_batches = len(loader)
    num_samples = len(loader.dataset)
    print(f"\n[Train] Epoch {epoch}/{num_epochs} | {num_samples} windows in {num_batches} batches.")
    start_time = time.time()
    log_every = max(1, num_batches // 5)

    for b, (images, mask, y_tgt, mu_act, meta) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        y_tgt = y_tgt.to(DEVICE, non_blocking=True)
        mu_act = mu_act.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred_tgt, _ = model(images, mask)
        loss = criterion(pred_tgt, y_tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        y_true = _to_y_space(y_tgt, mu_act, use_resid).detach().cpu().numpy()
        y_pred = _to_y_space(pred_tgt, mu_act, use_resid).detach().cpu().numpy()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        if b % log_every == 0 or b == num_batches:
            elapsed = time.time() - start_time
            progress = b / num_batches
            eta = (elapsed / progress) - elapsed if progress > 0 else float("nan")
            print(f"[Train] Batch {b}/{num_batches} ({progress*100:.1f}%) | Elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    avg_mse = total_loss / len(loader.dataset)

    metrics = compute_regression_metrics(y_true_all, y_pred_all)
    metrics["MSE"] = float(avg_mse)
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, epoch, num_epochs, use_resid: bool, stage="Val"):
    model.eval()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []
    resid_true_all, resid_pred_all, mu_all = [], [], []

    num_batches = len(loader)
    num_samples = len(loader.dataset)
    print(f"\n[{stage}] Epoch {epoch}/{num_epochs} | {num_samples} windows in {num_batches} batches.")
    start_time = time.time()
    log_every = max(1, num_batches // 5)

    for b, (images, mask, y_tgt, mu_act, meta) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        y_tgt = y_tgt.to(DEVICE, non_blocking=True)
        mu_act = mu_act.to(DEVICE, non_blocking=True)

        pred_tgt, _ = model(images, mask)
        loss = criterion(pred_tgt, y_tgt)
        total_loss += loss.item() * images.size(0)

        y_true = _to_y_space(y_tgt, mu_act, use_resid).detach().cpu().numpy()
        y_pred = _to_y_space(pred_tgt, mu_act, use_resid).detach().cpu().numpy()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        if use_resid:
            resid_true_all.append(y_tgt.detach().cpu().numpy())
            resid_pred_all.append(pred_tgt.detach().cpu().numpy())
            mu_all.append(mu_act.detach().cpu().numpy())

        if b % log_every == 0 or b == num_batches:
            elapsed = time.time() - start_time
            progress = b / num_batches
            eta = (elapsed / progress) - elapsed if progress > 0 else float("nan")
            print(f"[{stage}] Batch {b}/{num_batches} ({progress*100:.1f}%) | Elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

    y_true_all = np.concatenate(y_true_all).astype(np.float64)
    y_pred_all = np.concatenate(y_pred_all).astype(np.float64)
    avg_mse = total_loss / len(loader.dataset)

    metrics = compute_regression_metrics(y_true_all, y_pred_all)
    metrics["MSE"] = float(avg_mse)

    if use_resid:
        resid_true_all = np.concatenate(resid_true_all).astype(np.float64)
        resid_pred_all = np.concatenate(resid_pred_all).astype(np.float64)
        mu_all = np.concatenate(mu_all).astype(np.float64)
    else:
        resid_true_all, resid_pred_all, mu_all = None, None, None

    return metrics, y_true_all, y_pred_all, resid_true_all, resid_pred_all, mu_all


@torch.no_grad()
def predict_with_embeddings(model, loader, use_resid: bool):
    model.eval()
    y_true_all, y_pred_all = [], []
    fused_list = []
    meta_rows = []

    for images, mask, y_tgt, mu_act, meta in loader:
        images = images.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        y_tgt = y_tgt.to(DEVICE, non_blocking=True)
        mu_act = mu_act.to(DEVICE, non_blocking=True)

        pred_tgt, fused = model(images, mask)

        y_true = _to_y_space(y_tgt, mu_act, use_resid).detach().cpu().numpy()
        y_pred = _to_y_space(pred_tgt, mu_act, use_resid).detach().cpu().numpy()

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        fused_list.append(fused.detach().cpu().numpy())

        bs = images.shape[0]
        mu_np = mu_act.detach().cpu().numpy().astype(np.float32)

        # meta is list of dicts due to collate_keep_meta
        for i in range(bs):
            mi = meta[i]
            row_meta = dict(mi)
            row_meta["mu_act"] = float(mu_np[i])
            row_meta["use_activity_residual"] = int(use_resid)
            meta_rows.append(row_meta)

    y_true_all = np.concatenate(y_true_all).astype(np.float32)
    y_pred_all = np.concatenate(y_pred_all).astype(np.float32)
    fused_all = np.vstack(fused_list).astype(np.float16)
    meta_df = pd.DataFrame(meta_rows)

    if len(meta_df) != len(y_true_all):
        raise RuntimeError(
            f"[predict_with_embeddings] meta rows ({len(meta_df)}) != y rows ({len(y_true_all)}). "
            f"Meta is being aggregated incorrectly."
        )

    return y_true_all, y_pred_all, fused_all, meta_df


# 8) PLOTTING UTILITIES


def plot_learning_curves_mean(history_by_fold: Dict[str, List[dict]], out_dir: str):
    rows = []
    for fold, hist in history_by_fold.items():
        for h in hist:
            rows.append({
                "fold": fold,
                "epoch": h["epoch"],
                "train_RMSE": h["train"]["RMSE"],
                "val_RMSE": h["val_used"]["RMSE"],
                "train_R2": h["train"]["R2"],
                "val_R2": h["val_used"]["R2"],
                "train_MAE": h["train"]["MAE"],
                "val_MAE": h["val_used"]["MAE"],
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    dfm = df.groupby("epoch", as_index=False).mean(numeric_only=True).sort_values("epoch")

    plt.figure(figsize=(7, 4))
    plt.plot(dfm["epoch"], dfm["train_RMSE"], label="Train RMSE (mean)")
    plt.plot(dfm["epoch"], dfm["val_RMSE"], label="Val RMSE (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Learning Curve: RMSE vs Epoch (Mean over LOSO folds)")
    plt.grid(True)
    plt.legend()
    p = os.path.join(out_dir, "learning_curve_RMSE.png")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(dfm["epoch"], dfm["train_R2"], label="Train R2 (mean)")
    plt.plot(dfm["epoch"], dfm["val_R2"], label="Val R2 (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.title("Learning Curve: R2 vs Epoch (Mean over LOSO folds)")
    plt.grid(True)
    plt.legend()
    p = os.path.join(out_dir, "learning_curve_R2.png")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(dfm["epoch"], dfm["train_MAE"], label="Train MAE (mean)")
    plt.plot(dfm["epoch"], dfm["val_MAE"], label="Val MAE (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Learning Curve: MAE vs Epoch (Mean over LOSO folds)")
    plt.grid(True)
    plt.legend()
    p = os.path.join(out_dir, "learning_curve_MAE.png")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()

def save_scatter(y_true, y_pred, out_path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.25, edgecolors="none")
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx], "k--", label="Ideal y=x")
    plt.xlabel("Ground truth (W/kg)")
    plt.ylabel("Prediction (W/kg)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_residual_plots(y_true, y_pred, out_dir):
    resid = (y_pred - y_true).astype(np.float64)

    plt.figure(figsize=(7, 4))
    plt.hist(resid, bins=60)
    plt.xlabel("Residual (Pred - GT) [W/kg]")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.grid(True)
    p = os.path.join(out_dir, "supGT_residual_hist.png")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(y_true, resid, alpha=0.25, edgecolors="none")
    plt.axhline(0.0)
    plt.xlabel("GT (W/kg)")
    plt.ylabel("Residual (Pred - GT)")
    plt.title("Residual vs GT")
    plt.grid(True)
    p = os.path.join(out_dir, "supGT_residual_vs_gt.png")
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()

def save_group_bars(df_pred: pd.DataFrame, out_dir: str):
    # per-subject
    rows = []
    for sid, g in df_pred.groupby("subject_id"):
        g2 = g.dropna(subset=["EE_true_Wkg", "EE_pred_Wkg"])
        if len(g2) < 10:
            continue
        m = compute_regression_metrics(g2["EE_true_Wkg"].values, g2["EE_pred_Wkg"].values)
        rows.append({"subject_id": sid, "RMSE": m["RMSE"], "R2": m["R2"], "N": len(g2)})

    if rows:
        d = pd.DataFrame(rows).sort_values("subject_id")

        plt.figure(figsize=(8, 4))
        plt.bar(d["subject_id"], d["RMSE"])
        plt.xlabel("Subject")
        plt.ylabel("RMSE")
        plt.title("Per-Subject RMSE (LOSO Test)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y")
        p = os.path.join(out_dir, "per_subject_RMSE.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(d["subject_id"], d["R2"])
        plt.xlabel("Subject")
        plt.ylabel("R2")
        plt.title("Per-Subject R2 (LOSO Test)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y")
        p = os.path.join(out_dir, "per_subject_R2.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()

    # per-activity_code RMSE
    rows = []
    for code, g in df_pred.groupby("activity_code"):
        g2 = g.dropna(subset=["EE_true_Wkg", "EE_pred_Wkg"])
        if len(g2) < 20:
            continue
        m = compute_regression_metrics(g2["EE_true_Wkg"].values, g2["EE_pred_Wkg"].values)
        rows.append({"activity_code": int(code), "RMSE": m["RMSE"], "N": len(g2)})

    if rows:
        d = pd.DataFrame(rows).sort_values("activity_code")
        plt.figure(figsize=(10, 4))
        plt.bar(d["activity_code"].astype(str), d["RMSE"])
        plt.xlabel("Activity Code")
        plt.ylabel("RMSE")
        plt.title("Per-Activity-Code RMSE (LOSO Test)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y")
        p = os.path.join(out_dir, "per_activity_code_RMSE.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()



# 9) EXCEL + EXPORTS


def save_excel(excel_path: str,
               df_pred: pd.DataFrame,
               overall_metrics: dict,
               fold_metrics: pd.DataFrame,
               history_by_fold: Dict[str, List[dict]]) -> None:
    import openpyxl  # required by ExcelWriter engine

    metrics_df = pd.DataFrame({"Metric": list(overall_metrics.keys()), "Value": list(overall_metrics.values())})

    hist_rows = []
    for fold, hist in history_by_fold.items():
        for h in hist:
            row = {
                "fold_test_subject": fold,
                "epoch": h["epoch"],
                "train_RMSE": h["train"]["RMSE"],
                "train_R2": h["train"]["R2"],
                "train_MAE": h["train"]["MAE"],
                "val_RMSE": h["val_used"]["RMSE"],
                "val_R2": h["val_used"]["R2"],
                "val_MAE": h["val_used"]["MAE"],
                "val_raw_RMSE": h["val_raw"]["RMSE"],
                "alpha": h.get("alpha", 1.0),
                "beta": h.get("beta", 0.0),
                "used_calibration": int(h.get("used_calibration", 0)),
            }
            if "val_baseline_global" in h:
                row["val_baseline_global_RMSE"] = h["val_baseline_global"]["RMSE"]
                row["val_baseline_byAct_RMSE"] = h["val_baseline_by_act"]["RMSE"]
            hist_rows.append(row)

    df_hist = pd.DataFrame(hist_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        df_pred.to_excel(w, sheet_name="Predictions_LOSO_Test", index=False)
        metrics_df.to_excel(w, sheet_name="Metrics_Overall", index=False)
        fold_metrics.to_excel(w, sheet_name="Metrics_PerFold", index=False)
        df_hist.to_excel(w, sheet_name="TrainHistory_AllFolds", index=False)

def export_embeddings_and_predictions_global(out_dir: str,
                                            embeddings: np.ndarray,
                                            y_pred: np.ndarray,
                                            y_true: np.ndarray,
                                            meta_df: pd.DataFrame) -> None:
    np.save(os.path.join(out_dir, "embeddings_window_fused.npy"), embeddings.astype(np.float16))
    np.save(os.path.join(out_dir, "preds.npy"), y_pred.astype(np.float32))
    np.save(os.path.join(out_dir, "y_true.npy"), y_true.astype(np.float32))
    meta_df.to_csv(os.path.join(out_dir, "meta.csv"), index=False)

    print(f"[Embeddings] Saved fused embeddings: {embeddings.shape} -> {os.path.join(out_dir, 'embeddings_window_fused.npy')}")
    print(f"[Embeddings] Saved preds: {y_pred.shape} -> {os.path.join(out_dir, 'preds.npy')}")
    print(f"[Embeddings] Saved meta: {len(meta_df)} rows -> {os.path.join(out_dir, 'meta.csv')}")



# 10) LOSO TRAINING


def _select_val_subjects(trainval_subjects: List[str], test_subject: str, seed: int, n_val: int) -> List[str]:
    rng = np.random.RandomState(seed + int(test_subject[-2:]))
    subs = list(trainval_subjects)
    rng.shuffle(subs)
    n_val = max(1, min(n_val, len(subs) - 1))
    return subs[:n_val]

def run_one_fold(df_win_all: pd.DataFrame, test_subject: str, base_dino_cpu, base_state: dict, repo_dir: str):
    print("\n=====================================================")
    print(f"[LOSO] TEST SUBJECT: {test_subject}")
    print("=====================================================")

    # Outer split
    df_test = df_win_all[df_win_all["subject_id"] == test_subject].copy().reset_index(drop=True)
    df_trainval = df_win_all[df_win_all["subject_id"] != test_subject].copy().reset_index(drop=True)

    if len(df_test) == 0:
        raise RuntimeError(f"No windows for test subject {test_subject}")

    # Inner split
    if INNER_VAL_MODE == "subject":
        trainval_subjects = sorted(df_trainval["subject_id"].unique().tolist())
        val_subjects = _select_val_subjects(trainval_subjects, test_subject, RANDOM_SEED, N_VAL_SUBJECTS)
        val_df = df_trainval[df_trainval["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)
        train_df = df_trainval[~df_trainval["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)
        print(f"[Split] INNER_VAL_MODE=subject | Val subjects: {val_subjects}")
    else:
        indices = np.arange(len(df_trainval))
        rng = np.random.RandomState(RANDOM_SEED + int(test_subject[-2:]))
        rng.shuffle(indices)
        n_val = int(len(indices) * 0.2)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
        val_df = df_trainval.iloc[val_idx].reset_index(drop=True)
        print("[Split] INNER_VAL_MODE=window (not recommended)")

    print(f"[Split] Train windows: {len(train_df)} | Val windows: {len(val_df)} | Test windows: {len(df_test)}")

    # Baselines built ONLY from train (original y)
    global_mean, by_code_mean = build_baselines(train_df)

    # Residual mu(activity) mapping built ONLY from train
    resid_by_code_mean = by_code_mean
    resid_global_mean = global_mean

    # Fold dirs
    fold_name = f"LOSO_{test_subject}"
    fold_ckpt_dir = os.path.join(CHECKPOINT_DIR, fold_name)
    fold_plot_dir = os.path.join(PLOTS_DIR, fold_name)
    fold_embed_dir = os.path.join(EMBED_DIR, fold_name)
    for p in [fold_ckpt_dir, fold_plot_dir, fold_embed_dir]:
        _ensure_dir(p)

    best_ckpt_path = os.path.join(fold_ckpt_dir, "best.pt")
    last_ckpt_path = os.path.join(fold_ckpt_dir, "last.pt")

    # Load DINOv2 + preprocess
    print(f"[Model] Building fresh DINOv2 {DINO_NAME} from preloaded backbone (no hub load in fold).")

    dino = copy.deepcopy(base_dino_cpu)

    dino.load_state_dict(base_state, strict=True)

    dino = dino.to(DEVICE).float()
    preprocess = build_dinov2_preprocess(IMAGE_SIZE)

    if FREEZE_DINO:
        for p in dino.parameters():
            p.requires_grad = False
        dino.eval()

    model = DinoV2FusionRegressor(
        dino_model=dino,
        freeze_dino=FREEZE_DINO,
        hidden_dim=512,
        dropout=0.2
    ).to(DEVICE)

    criterion = nn.MSELoss()

    if FREEZE_DINO:
        optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
    else:
        dino_params = [p for p in model.dino.parameters() if p.requires_grad]
        head_params = list(model.mlp.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": dino_params, "lr": LEARNING_RATE_DINO, "weight_decay": 0.01},
                {"params": head_params, "lr": LEARNING_RATE_HEAD, "weight_decay": WEIGHT_DECAY},
            ]
        )

    # Datasets
    train_ds = WindowFusionDataset(
        train_df, base_dir=BASE_DIR, preprocess=preprocess, image_size=IMAGE_SIZE,
        use_activity_residual=USE_ACTIVITY_RESIDUAL,
        by_code_mean=resid_by_code_mean,
        global_mean=resid_global_mean
    )
    val_ds = WindowFusionDataset(
        val_df, base_dir=BASE_DIR, preprocess=preprocess, image_size=IMAGE_SIZE,
        use_activity_residual=USE_ACTIVITY_RESIDUAL,
        by_code_mean=resid_by_code_mean,
        global_mean=resid_global_mean
    )
    test_ds = WindowFusionDataset(
        df_test, base_dir=BASE_DIR, preprocess=preprocess, image_size=IMAGE_SIZE,
        use_activity_residual=USE_ACTIVITY_RESIDUAL,
        by_code_mean=resid_by_code_mean,
        global_mean=resid_global_mean
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_keep_meta,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_keep_meta,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_keep_meta,
    )

    # Resume per fold
    history = []
    start_epoch = 1
    best_val_rmse = float("inf")

    if EARLY_STOP_PATIENCE is None:
        patience_left = None
    else:
        patience_left = int(EARLY_STOP_PATIENCE)

    best_alpha, best_beta = 1.0, 0.0

    if RESUME_TRAINING and (os.path.exists(last_ckpt_path) or os.path.exists(best_ckpt_path)):
        ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else best_ckpt_path
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            history = ckpt.get("history", [])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_rmse = float(ckpt.get("best_val_rmse", best_val_rmse))
            loaded_pat = ckpt.get("patience_left", patience_left)

            if EARLY_STOP_PATIENCE is None:
                patience_left = None
            else:
                if loaded_pat is None:
                    patience_left = int(EARLY_STOP_PATIENCE)
                else:
                    patience_left = int(loaded_pat)

            best_alpha = float(ckpt.get("alpha", best_alpha))
            best_beta = float(ckpt.get("beta", best_beta))
            print(f"[Resume] Loaded {ckpt_path}. Resuming at epoch {start_epoch} (best_val_rmse={best_val_rmse:.4f}).")

    if start_epoch > NUM_EPOCHS:
        print(f"[Info] Fold {test_subject}: start_epoch={start_epoch} > NUM_EPOCHS={NUM_EPOCHS}. Skipping training.")
    else:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            t0 = time.time()

            tr = train_one_epoch(model, train_loader, optimizer, criterion, epoch, NUM_EPOCHS, use_resid=USE_ACTIVITY_RESIDUAL)
            torch.cuda.empty_cache()

            va_raw, yv_true, yv_pred, resid_true, resid_pred, mu_val = eval_one_epoch(
                model, val_loader, criterion, epoch, NUM_EPOCHS, use_resid=USE_ACTIVITY_RESIDUAL, stage="Val"
            )
            torch.cuda.empty_cache()

            # Baselines on VAL (computed from train only, on ORIGINAL y)
            y_val = val_df["y"].values.astype(np.float64)
            base_global = compute_regression_metrics(y_val, baseline_predict_global(len(val_df), global_mean))
            val_codes = val_df["activity_code"].values
            base_byact = compute_regression_metrics(y_val, baseline_predict_by_code(val_codes, global_mean, by_code_mean))

            # Optional calibration (only meaningful for residual training)
            used_calibration = False
            alpha, beta = 1.0, 0.0
            va_used = va_raw
            if CALIBRATE_RESIDUAL and USE_ACTIVITY_RESIDUAL and (resid_true is not None) and (len(resid_true) > 10):
                alpha, beta = fit_alpha_beta(resid_true, resid_pred)
                yv_pred_cal = (mu_val + alpha * resid_pred + beta).astype(np.float64)
                va_cal = compute_regression_metrics(yv_true, yv_pred_cal)
                va_used = va_cal
                used_calibration = True

            history.append({
                "epoch": epoch,
                "train": tr,
                "val_raw": va_raw,
                "val_used": va_used,
                "alpha": alpha,
                "beta": beta,
                "used_calibration": used_calibration,
                "val_baseline_global": base_global,
                "val_baseline_by_act": base_byact,
            })

            dt_min = (time.time() - t0) / 60.0
            print(f"\n[Epoch {epoch}/{NUM_EPOCHS} SUMMARY] ({test_subject})")
            print(f"  Train | RMSE {tr['RMSE']:.4f} | R² {tr['R2']*100:.2f}% | MAE {tr['MAE']:.4f}")
            print(f"  Val   | RMSE {va_used['RMSE']:.4f} | R² {va_used['R2']*100:.2f}% | MAE {va_used['MAE']:.4f} "
                  f"{'(calibrated)' if used_calibration else '(raw)'}")
            print(f"  Baseline(GlobalMean) Val RMSE {base_global['RMSE']:.4f} | R² {base_global['R2']*100:.2f}%")
            print(f"  Baseline(ByActMean)  Val RMSE {base_byact['RMSE']:.4f} | R² {base_byact['R2']*100:.2f}%")
            if used_calibration:
                print(f"  Calib alpha={alpha:.4f}, beta={beta:.4f}")
            print(f"  Epoch time: {dt_min:.2f} min")

            last_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "best_val_rmse": best_val_rmse,
                "patience_left": patience_left,
                "alpha": best_alpha,
                "beta": best_beta,
                "config": {
                    "test_subject": test_subject,
                    "VIEW_ORDER": VIEW_ORDER,
                    "MIN_VIEWS_PER_WINDOW": MIN_VIEWS_PER_WINDOW,
                    "EXCLUDE_ACTIVITY_CODES": sorted(list(EXCLUDE_ACTIVITY_CODES)),
                    "TRAIN_LABEL_COL": TRAIN_LABEL_COL,
                    "DINO_NAME": DINO_NAME,
                    "FREEZE_DINO": FREEZE_DINO,
                    "INNER_VAL_MODE": INNER_VAL_MODE,
                    "N_VAL_SUBJECTS": N_VAL_SUBJECTS,
                    "USE_ACTIVITY_RESIDUAL": USE_ACTIVITY_RESIDUAL,
                    "CALIBRATE_RESIDUAL": CALIBRATE_RESIDUAL,
                    "IMAGE_SIZE": IMAGE_SIZE,
                }
            }
            torch.save(last_payload, last_ckpt_path)
            print(f"  [CKPT] Saved last: {last_ckpt_path}")

            improved = va_used["RMSE"] < best_val_rmse
            if improved:
                best_val_rmse = va_used["RMSE"]
                patience_left = EARLY_STOP_PATIENCE
                best_alpha, best_beta = alpha, beta

                best_payload = dict(last_payload)
                best_payload["best_val_rmse"] = best_val_rmse
                best_payload["patience_left"] = patience_left
                best_payload["alpha"] = best_alpha
                best_payload["beta"] = best_beta
                torch.save(best_payload, best_ckpt_path)
                print(f"  [CKPT] New best Val RMSE. Saved best: {best_ckpt_path}")
            else:
                if EARLY_STOP_PATIENCE is not None:
                    patience_left -= 1
                    print(f"  [EarlyStop] No improvement. Patience left: {patience_left}/{EARLY_STOP_PATIENCE}")
                    if patience_left <= 0:
                        print("  [EarlyStop] Stopping early.")
                        break
                else:
                    print("  [EarlyStop] No improvement, but EARLY_STOP_PATIENCE=None so continuing.")

    # Restore best
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        if isinstance(best_ckpt, dict) and "model_state" in best_ckpt:
            model.load_state_dict(best_ckpt["model_state"])
            best_val_rmse = float(best_ckpt.get("best_val_rmse", best_val_rmse))
            best_alpha = float(best_ckpt.get("alpha", 1.0))
            best_beta = float(best_ckpt.get("beta", 0.0))
            print(f"[Model] Restored best weights for {test_subject} (best_val_rmse={best_val_rmse:.4f}).")
            if USE_ACTIVITY_RESIDUAL and CALIBRATE_RESIDUAL:
                print(f"[Model] Using best calibration alpha={best_alpha:.4f}, beta={best_beta:.4f}")

    # Test predictions + embeddings
    y_true_test, y_pred_test, emb_test, meta_test = predict_with_embeddings(model, test_loader, use_resid=USE_ACTIVITY_RESIDUAL)

    # Apply best calibration on TEST (only for residual mode)
    if USE_ACTIVITY_RESIDUAL and CALIBRATE_RESIDUAL:
        mu_test = meta_test["mu_act"].values.astype(np.float64)
        resid_pred_test = (y_pred_test.astype(np.float64) - mu_test)
        y_pred_test_cal = (mu_test + best_alpha * resid_pred_test + best_beta).astype(np.float32)
        y_pred_test = y_pred_test_cal

    test_metrics = compute_regression_metrics(y_true_test, y_pred_test)

    # Baselines on TEST (computed from train only)
    test_codes = df_test["activity_code"].values
    base_global_test = compute_regression_metrics(y_true_test, baseline_predict_global(len(df_test), global_mean))
    base_byact_test = compute_regression_metrics(y_true_test, baseline_predict_by_code(test_codes, global_mean, by_code_mean))

    print(f"\n[LOSO Test Metrics] {test_subject}")
    print(f"  Model  | RMSE {test_metrics['RMSE']:.4f} | R² {test_metrics['R2']*100:.2f}% | MAE {test_metrics['MAE']:.4f}")
    print(f"  Global | RMSE {base_global_test['RMSE']:.4f} | R² {base_global_test['R2']*100:.2f}%")
    print(f"  ByAct  | RMSE {base_byact_test['RMSE']:.4f} | R² {base_byact_test['R2']*100:.2f}%")

    # Save fold embeddings/preds (keep filenames)
    np.save(os.path.join(fold_embed_dir, "embeddings_window_fused.npy"), emb_test.astype(np.float16))
    np.save(os.path.join(fold_embed_dir, "preds.npy"), y_pred_test.astype(np.float32))
    np.save(os.path.join(fold_embed_dir, "y_true.npy"), y_true_test.astype(np.float32))
    meta_test.to_csv(os.path.join(fold_embed_dir, "meta.csv"), index=False)

    return {
        "test_subject": test_subject,
        "history": history,
        "test_metrics": test_metrics,
        "test_baseline_global": base_global_test,
        "test_baseline_byact": base_byact_test,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
        "emb_test": emb_test,
        "meta_test": meta_test,
    }



# 11) MAIN


def main():
    print("==============================================")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {DEVICE}")
    print(f"DINO_NAME: {DINO_NAME} | FREEZE_DINO: {FREEZE_DINO}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"INNER_VAL_MODE: {INNER_VAL_MODE} | N_VAL_SUBJECTS: {N_VAL_SUBJECTS}")
    print(f"USE_ACTIVITY_RESIDUAL: {USE_ACTIVITY_RESIDUAL} | CALIBRATE_RESIDUAL: {CALIBRATE_RESIDUAL}")
    print(f"VIEW_ORDER: {VIEW_ORDER} | MIN_VIEWS_PER_WINDOW: {MIN_VIEWS_PER_WINDOW}")
    print("==============================================")

    set_random_seed(RANDOM_SEED)

    df_rows = load_all_rows(BASE_DIR, SUBJECT_IDS)
    df_win_all = build_window_table(df_rows)

    print("[Debug] Columns:", [c for c in df_win_all.columns if c.startswith("rel_path_v") or c.startswith("mask_v")])
    print(df_win_all["n_views_available"].value_counts(dropna=False).sort_index())


    # ------------------------------------------------------------
    # Preload DINOv2 ONCE (CPU) and keep a frozen copy of its weights.
    # This prevents repeated torch.hub calls per fold and avoids network/GitHub issues.
    # ------------------------------------------------------------
    repo_dir = resolve_dinov2_repo_dir()
    print(f"[Model] Preloading DINOv2 {DINO_NAME} from local hub cache: {repo_dir}")
    base_dino_cpu = torch.hub.load(repo_dir, DINO_NAME, source="local").cpu().float()
    base_dino_cpu.eval()
    base_state = {k: v.clone() for k, v in base_dino_cpu.state_dict().items()}

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_subject", type=str, default=None, help="Run only one LOSO fold, e.g. Subject01")
    args = parser.parse_args()

    if args.test_subject is not None:
        test_subjects = [args.test_subject]
    elif RUN_ALL_FOLDS:
        test_subjects = SUBJECT_IDS
    else:
        test_subjects = [SUBJECT_IDS[0]]

    history_by_fold = {}
    fold_summaries = []
    pooled_y_true = []
    pooled_y_pred = []
    pooled_emb = []
    pooled_meta = []

    for test_sid in test_subjects:
        fold_out = run_one_fold(df_win_all, test_sid, base_dino_cpu, base_state, repo_dir)

        history_by_fold[test_sid] = fold_out["history"]
        tm = fold_out["test_metrics"]
        bg = fold_out["test_baseline_global"]
        ba = fold_out["test_baseline_byact"]

        fold_summaries.append({
            "test_subject": test_sid,
            "Model_RMSE": tm["RMSE"],
            "Model_MAE": tm["MAE"],
            "Model_R2": tm["R2"],
            "Baseline_Global_RMSE": bg["RMSE"],
            "Baseline_Global_R2": bg["R2"],
            "Baseline_ByAct_RMSE": ba["RMSE"],
            "Baseline_ByAct_R2": ba["R2"],
        })

        pooled_y_true.append(fold_out["y_true_test"])
        pooled_y_pred.append(fold_out["y_pred_test"])
        pooled_emb.append(fold_out["emb_test"])
        pooled_meta.append(fold_out["meta_test"])

    # Pool LOSO test predictions
    y_true_all = np.concatenate(pooled_y_true).astype(np.float32)
    y_pred_all = np.concatenate(pooled_y_pred).astype(np.float32)
    emb_all = np.vstack(pooled_emb).astype(np.float16)
    meta_all = pd.concat(pooled_meta, ignore_index=True)

    overall = compute_regression_metrics(y_true_all, y_pred_all)
    print("\n[FINAL LOSO TEST] Pooled metrics:")
    print(f"  RMSE: {overall['RMSE']:.4f}")
    print(f"  MAE : {overall['MAE']:.4f}")
    print(f"  R2  : {overall['R2']:.4f} ({overall['R2']*100:.2f}%)")

    df_pred = meta_all.copy()
    df_pred["EE_true_Wkg"] = y_true_all
    df_pred["EE_pred_Wkg"] = y_pred_all

    # Root plots
    plot_learning_curves_mean(history_by_fold, PLOTS_DIR)
    save_scatter(
        y_true_all, y_pred_all,
        out_path=os.path.join(PLOTS_DIR, "scatter_GT_vs_pred_supGT.png"),
        title=f"LOSO Fusion | DINOv2={DINO_NAME} | R²={overall['R2']:.3f}, RMSE={overall['RMSE']:.3f}, MAE={overall['MAE']:.3f}"
    )
    save_residual_plots(y_true_all, y_pred_all, PLOTS_DIR)
    save_group_bars(df_pred, PLOTS_DIR)

    # Global embedding export
    export_embeddings_and_predictions_global(EMBED_DIR, emb_all, y_pred_all, y_true_all, meta_all)

    # Excel
    fold_metrics_df = pd.DataFrame(fold_summaries).sort_values("test_subject")
    overall_metrics = {
        "RUN_NAME": RUN_NAME,
        "BACKBONE": "DINOv2",
        "DINO_NAME": DINO_NAME,
        "FREEZE_DINO": str(FREEZE_DINO),
        "IMAGE_SIZE": IMAGE_SIZE,
        "VIEWS_USED": ", ".join(VIEW_ORDER),
        "MIN_VIEWS_PER_WINDOW": MIN_VIEWS_PER_WINDOW,
        "EXCLUDE_CODES": str(sorted(list(EXCLUDE_ACTIVITY_CODES))),
        "TRAIN_LABEL_COL": TRAIN_LABEL_COL,
        "NUM_EPOCHS": NUM_EPOCHS,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE_HEAD": LEARNING_RATE_HEAD,
        "LEARNING_RATE_DINO": LEARNING_RATE_DINO,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "INNER_VAL_MODE": INNER_VAL_MODE,
        "N_VAL_SUBJECTS": N_VAL_SUBJECTS,
        "USE_ACTIVITY_RESIDUAL": USE_ACTIVITY_RESIDUAL,
        "CALIBRATE_RESIDUAL": CALIBRATE_RESIDUAL,
        "RMSE_LOSO_TEST_POOLED": overall["RMSE"],
        "MAE_LOSO_TEST_POOLED": overall["MAE"],
        "R2_LOSO_TEST_POOLED": overall["R2"],
    }

    excel_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}.xlsx")
    save_excel(excel_path, df_pred, overall_metrics, fold_metrics_df, history_by_fold)

    print("\n[DONE]")
    print(f"Results:    {RESULTS_DIR}")
    print(f"Plots:      {PLOTS_DIR}")
    print(f"Embeddings: {EMBED_DIR}")
    print(f"Excel:      {excel_path}")


if __name__ == "__main__":
    main()
