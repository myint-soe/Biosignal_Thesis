"""

Key Signal Processing Rules:
1. Metabolics (HR, MV, BF, SpO2) -> 360s Context (Resampled to 256 pts).
2. Skin Temp                     -> 180s Context (Resampled to 256 pts).
3. EDA                           -> 180s  Context (Resampled to 256 pts).
4. EMG (Left/Right)              -> 5s   Window  (Native 1000Hz, no resampling).
"""

import os
import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.signal import stft
from pyts.image import GramianAngularField
from collections import defaultdict


# 1. CONFIGURATION

BASE_DIR = r"/home/milkyway/MaungMyintSoe/data/raw_data" 
SUBJECT_ID = "Subject10"
SOURCE_MANIFEST_PATH = r"/home/milkyway/MaungMyintSoe/data/raw_data/Subject10/Subject10_multiview_manifest_5s_withActCode_withSupGT.csv"

# Output settings
OUTPUT_ROOT = os.path.join(BASE_DIR, f"{SUBJECT_ID}_Global+EMG_Images_v4")
OUTPUT_MANIFEST = os.path.join(OUTPUT_ROOT, f"{SUBJECT_ID}_generated_manifest.csv")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- Time Window Strategies ---
# 1. Standard High-Speed (EMG)
WINDOW_SEC = 5.0              
# 2. Slow Metabolics
METAB_CONTEXT_SEC = 360.0     
# 3. Empatica Specifics
TEMP_CONTEXT_SEC = 180.0      # Very slow drift
EDA_CONTEXT_SEC = 180.0       # Phasic events

# Resampling target for all context-based signals
RESAMPLE_LEN = 256            

IMAGE_SIZE = 224              
SKIP_CODES = {22, 23}         

# Defines the exact order and list of columns for the output CSV
OUTPUT_COLUMNS = [
    "subject_id",
    "activity_name",
    "window_idx",
    "window_start_s",
    "window_end_s",
    "window_center_s",
    "data_status",
    "gt_supervisor_Wkg",
    "activity_code",
    "signal_name",     # <--- Added (Sensor Name)
    "image_filename",
    "image_rel_path",
    "image_generated",
    "missing_reason"
]

# 2. SIGNAL PROCESSING FUNCTIONS


def clean_signal(signal):
    """Interpolates NaNs and fills edges."""
    s = pd.Series(signal).interpolate("linear", limit_direction="both").fillna(0.0)
    return s.values

def normalize_image(img):
    """Scales image to 0-255."""
    vmin, vmax = np.min(img), np.max(img)
    if vmax - vmin < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - vmin) / (vmax - vmin)
    return (img * 255).astype(np.uint8)

def generate_cwt(signal, fs):
    """Red Channel: Continuous Wavelet Transform."""
    signal = signal - np.mean(signal)
    scales = np.arange(1, 65)
    coef, _ = pywt.cwt(signal, scales, "cmor1.5-1.0", sampling_period=1.0 / fs)
    cwt_img = cv2.resize(normalize_image(np.abs(coef)), (IMAGE_SIZE, IMAGE_SIZE))
    return cwt_img

def generate_stft(signal, fs, nperseg=64):
    """Green Channel: STFT (Spectrogram)."""
    signal = signal - np.mean(signal)
    n = len(signal)
    nperseg = min(nperseg, n) if n > 0 else nperseg
    if n < 4: return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    _f, _t, Zxx = stft(signal, fs, nperseg=nperseg)
    spec = cv2.resize(normalize_image(np.flipud(np.log1p(np.abs(Zxx)))), (IMAGE_SIZE, IMAGE_SIZE))
    return spec

def generate_gadf(signal):
    """Blue Channel: Gramian Angular Difference Field."""
    signal = signal - np.mean(signal)
    target_len = IMAGE_SIZE
    # Resize 1D signal to match image width for GADF
    if len(signal) != target_len and len(signal) > 1:
        signal = np.interp(np.linspace(0, len(signal)-1, target_len), np.arange(len(signal)), signal)
    elif len(signal) < 2:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    X = signal.reshape(1, -1)
    gadf = GramianAngularField(image_size=IMAGE_SIZE, method="difference")
    try:
        return normalize_image(gadf.fit_transform(X)[0])
    except:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

def signal_to_rgb(seg, fs, stft_nperseg=64):
    """Merges CWT, STFT, GADF into one RGB image."""
    r = generate_cwt(seg, fs)
    g = generate_stft(seg, fs, nperseg=stft_nperseg)
    b = generate_gadf(seg)
    return cv2.merge([r, g, b])


# 3. DATA HELPERS


def resample_to_fixed_length(t, x, start, end, out_len=256):
    """Extracts window [start, end] and resamples to out_len."""
    if t is None or x is None or len(t) < 2: return None, None
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    if i1 - i0 < 2: return None, None
    
    tt = t[i0:i1].astype(float)
    xx = clean_signal(x[i0:i1].astype(float))
    grid = np.linspace(start, end, out_len, endpoint=False)
    y = np.interp(grid, tt, xx)
    
    fs_eff = out_len / max(1e-9, end - start)
    return y, fs_eff

def slice_by_time(t, x, start, end):
    """Extracts window [start, end] preserving original samples."""
    if t is None or x is None: return None
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    if i1 <= i0: return None
    return clean_signal(x[i0:i1].astype(float))

def resolve_folder(subject_dir, possible_names):
    """Robust folder finder."""
    for name in possible_names:
        p = os.path.join(subject_dir, name)
        if os.path.isdir(p): return p
    if os.path.isdir(subject_dir):
        for d in os.listdir(subject_dir):
            if d.lower() in [n.lower() for n in possible_names]:
                return os.path.join(subject_dir, d)
    return None


# 4. SENSOR LOADERS


def load_metabolics(act_path):
    p = resolve_folder(act_path, ["Metabolics_System", "Metabolics System"])
    if not p: return None
    path = os.path.join(p, "Data.csv")
    if not os.path.exists(path): return None

    try:
        df = pd.read_csv(path, header=None).apply(pd.to_numeric, errors='coerce')
        t = df.iloc[:, 0].values
        # 5:BF, 6:VE, 7:SpO2, 8:HR
        return {
            "time": t,
            "hr": df.iloc[:, 8].values if df.shape[1]>8 else None,
            "ve": df.iloc[:, 6].values if df.shape[1]>6 else None,
            "bf": df.iloc[:, 5].values if df.shape[1]>5 else None,
            "spo2": df.iloc[:, 7].values if df.shape[1]>7 else None,
        }
    except: return None

def load_empatica(act_path):
    p = resolve_folder(act_path, ["Empatica_Physio", "Empatica Physio"])
    if not p: return None
    path = os.path.join(p, "Data.csv")
    if not os.path.exists(path): return None

    try:
        df = pd.read_csv(path, header=None)
        t = df.iloc[:, 0].values.astype(float)
        # 2:L_EDA, 3:L_Temp, 4:R_EDA, 5:R_Temp
        return {
            "time": t,
            "eda_l": df.iloc[:, 2].values,
            "temp_l": df.iloc[:, 3].values,
            "eda_r": df.iloc[:, 4].values,
            "temp_r": df.iloc[:, 5].values,
            "fs": 4.0 
        }
    except: return None

def load_emg(act_path):
    p = resolve_folder(act_path, ["EMG"])
    if not p: return None
    path = os.path.join(p, "Data.csv")
    if not os.path.exists(path): return None
    
    try:
        df = pd.read_csv(path, header=None)
        t = df.iloc[:, 0].values.astype(float)
        ch_start = 2 if df.shape[1] >= 18 else 1
        X = df.iloc[:, ch_start:].to_numpy()
        half = X.shape[1] // 2
        
        left_mag = np.mean(np.abs(np.nan_to_num(X[:, :half])), axis=1)
        right_mag = np.mean(np.abs(np.nan_to_num(X[:, half:])), axis=1)
        
        sr_path = os.path.join(p, "sampling rate")
        fs = 1000.0
        if os.path.exists(sr_path):
            try: fs = float(open(sr_path).read().strip())
            except: pass
        return {"time": t, "left": left_mag, "right": right_mag, "fs": fs}
    except: return None


# 5. EXECUTION LOOP


if __name__ == "__main__":
    print(f"🚀 Starting V4 Pipeline for {SUBJECT_ID}...")
    
    # Load Source Manifest
    src = pd.read_csv(SOURCE_MANIFEST_PATH) if SOURCE_MANIFEST_PATH.endswith('.csv') else pd.read_excel(SOURCE_MANIFEST_PATH)
    
    if "subject_id" in src.columns:
        src = src[src["subject_id"].astype(str) == SUBJECT_ID].copy()
    if "activity_code" in src.columns:
        src = src[~src["activity_code"].isin(SKIP_CODES)]

    # Locate Columns in Source
    act_col = next(c for c in ["activity_name", "Activity"] if c in src.columns)
    idx_col = next(c for c in ["window_idx", "win_idx"] if c in src.columns)
    start_col = next(c for c in ["window_start_s", "start"] if c in src.columns)
    end_col = next(c for c in ["window_end_s", "end"] if c in src.columns)
    ctr_col = next(c for c in ["window_center_s", "center"] if c in src.columns)
    
    # Locate Ground Truth Column
    gt_col = "gt_supervisor_Wkg"
    if gt_col not in src.columns:
        print(f"⚠️ Warning: '{gt_col}' not found in source manifest. Looking for alternatives...")
        possible_gts = ["label_EE_Wkg_net_6min", "ground_truth_value"]
        for p in possible_gts:
            if p in src.columns:
                gt_col = p
                print(f"   -> Using '{gt_col}' as ground truth source.")
                break
    
    manifest_rows = []
    stats = defaultdict(lambda: {"valid": 0, "skipped": 0})

    for activity_name, df_act in src.groupby(act_col):
        act_path = resolve_folder(os.path.join(BASE_DIR, SUBJECT_ID), [str(activity_name), str(activity_name).replace(" ", "_")])
        if not act_path: continue
            
        print(f"Processing: {activity_name}...")
        
        metab = load_metabolics(act_path)
        emp = load_empatica(act_path)
        emg = load_emg(act_path)
        
        activity_out = str(activity_name).replace(" ", "_")
        
        for _, row in df_act.iterrows():
            w_idx = int(row[idx_col])
            w_start = float(row[start_col])
            w_end = float(row[end_col])
            w_center = float(row[ctr_col])
            
            todo = []
            
            # --- 1. Metabolics (360s Context) ---
            if metab:
                m_start, m_end = w_center - METAB_CONTEXT_SEC/2, w_center + METAB_CONTEXT_SEC/2
                for key, name in [("hr", "HR"), ("ve", "MinuteVent"), ("bf", "BreathFreq"), ("spo2", "OxygenSat")]:
                    if metab[key] is not None:
                        seg, fs_eff = resample_to_fixed_length(metab["time"], metab[key], m_start, m_end, RESAMPLE_LEN)
                        todo.append((name, seg, fs_eff, RESAMPLE_LEN, 64))

            # --- 2. Empatica ---
            if emp:
                t_start, t_end = w_center - TEMP_CONTEXT_SEC/2, w_center + TEMP_CONTEXT_SEC/2
                for key, name in [("temp_l", "Temp_Left"), ("temp_r", "Temp_Right")]:
                    seg, fs_eff = resample_to_fixed_length(emp["time"], emp[key], t_start, t_end, RESAMPLE_LEN)
                    todo.append((name, seg, fs_eff, RESAMPLE_LEN, 64))
                    
                e_start, e_end = w_center - EDA_CONTEXT_SEC/2, w_center + EDA_CONTEXT_SEC/2
                for key, name in [("eda_l", "EDA_Left"), ("eda_r", "EDA_Right")]:
                    seg, fs_eff = resample_to_fixed_length(emp["time"], emp[key], e_start, e_end, RESAMPLE_LEN)
                    todo.append((name, seg, fs_eff, RESAMPLE_LEN, 64))

            # --- 3. EMG (5s Standard) ---
            if emg:
                seg_l = slice_by_time(emg["time"], emg["left"], w_start, w_end)
                seg_r = slice_by_time(emg["time"], emg["right"], w_start, w_end)
                todo.append(("EMG_LeftMag", seg_l, emg["fs"], 200, 64))
                todo.append(("EMG_RightMag", seg_r, emg["fs"], 200, 64))

            # Generate Images & Build Manifest Rows
            for sig_name, seg, fs_val, min_len, stft_win in todo:
                
                # Construct output row
                out_row = {
                    "subject_id": SUBJECT_ID,
                    "activity_name": activity_name,
                    "window_idx": w_idx,
                    "window_start_s": w_start,
                    "window_end_s": w_end,
                    "window_center_s": w_center,
                    "data_status": "PENDING",
                    "gt_supervisor_Wkg": row.get(gt_col, np.nan),
                    "activity_code": row.get("activity_code", -1),
                    "signal_name": sig_name,  # <--- Populated Here
                    "image_filename": "",
                    "image_rel_path": "",
                    "image_generated": False,
                    "missing_reason": ""
                }
                
                if fs_val is None or seg is None or len(seg) < min_len or np.std(seg) < 1e-6:
                    out_row["data_status"] = "SKIPPED"
                    out_row["image_generated"] = False
                    out_row["missing_reason"] = "Signal Missing/Poor Quality"
                    stats[activity_out]["skipped"] += 1
                    manifest_rows.append(out_row)
                    continue
                
                try:
                    img = signal_to_rgb(seg, fs_val, stft_nperseg=stft_win)
                    
                    out_dir = os.path.join(OUTPUT_ROOT, activity_out, sig_name)
                    os.makedirs(out_dir, exist_ok=True)
                    fname = f"{SUBJECT_ID}_{activity_out}_{sig_name}_{w_idx:05d}.png"
                    cv2.imwrite(os.path.join(out_dir, fname), img)
                    
                    # Update row with path info
                    out_row["image_filename"] = fname
                    # Calculates relative path: Activity/Signal/Filename.png
                    out_row["image_rel_path"] = os.path.join(activity_out, sig_name, fname)
                    out_row["data_status"] = "VALID"
                    out_row["image_generated"] = True
                    out_row["missing_reason"] = ""
                    
                    stats[activity_out]["valid"] += 1
                    manifest_rows.append(out_row)
                except Exception as e:
                    out_row["data_status"] = "ERROR"
                    out_row["image_generated"] = False
                    out_row["missing_reason"] = str(e)
                    manifest_rows.append(out_row)

    # Save Output
    if manifest_rows:
        df_out = pd.DataFrame(manifest_rows)
        # Ensure exact column order
        df_out = df_out[OUTPUT_COLUMNS] 
        
        df_out.to_csv(OUTPUT_MANIFEST, index=False)
        print(f"\n✅ Finished. Manifest: {OUTPUT_MANIFEST}")
        for act, s in stats.items():
            print(f"   {act}: {s['valid']} valid, {s['skipped']} skipped")