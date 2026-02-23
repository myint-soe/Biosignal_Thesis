"""
multiview_5s_from_manifest.py

Reads an existing manifest (Source) containing 5s windows and GT labels.
Generates 4 image views per window:
  1) Core_Accel_EMG
  2) Legs_IMU
  3) Wrist_Physio
  4) Metab_Global_NoVO2VCO2

NO LABEL COMPUTATION: Labels are copied directly from the source manifest.
NO BROCKWAY / NO MASS required.

Output:
  data/Subject01_multiview_360_5s_FromManifest/
    <Activity>/<View_Name>/*.png
  Manifest:
    <OUTPUT_ROOT>/Subject01_multiview_manifest_5s.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.signal import stft
from pyts.image import GramianAngularField
from collections import defaultdict


# CONFIGURATION

BASE_DIR = r"/home/milkyway/MaungMyintSoe/data/raw_data"
SUBJECT_ID = "Subject10"

# Source Manifest (Contains defined windows & Ground Truth)
SOURCE_MANIFEST_PATH = r"/home/milkyway/MaungMyintSoe/data/raw_data/Subject10/Subject10_multiview_manifest_5s_withActCode_withSupGT.csv"

LABEL_SMOOTH_SEC = 360.0   # Context size for Metab View (View 4)
IMAGE_SIZE = 224

# Output locations
SUBJECT_DIR = os.path.join(BASE_DIR, SUBJECT_ID)
OUTPUT_ROOT = os.path.join(BASE_DIR, f"{SUBJECT_ID}_multiview_360_5s")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
OUTPUT_MANIFEST = os.path.join(OUTPUT_ROOT, f"{SUBJECT_ID}_multiview_manifest_5s.csv")

# Columns to copy from Source to Output if they exist
KEEP_COLS = [
    "label_EE_Wkg_total_6min", 
    "label_EE_Wkg_net_6min", 
    "gt_supervisor_Wkg", 
    "activity_code",
    "ground_truth_value" 
]

print(f" Multiview Pipeline (From Manifest) for {SUBJECT_ID}")
print(f"    Source Manifest: {SOURCE_MANIFEST_PATH}")
print(f"    Output Root:     {OUTPUT_ROOT}")



# GENERIC HELPERS

def clean_signal(signal):
    s = pd.Series(signal)
    s = s.interpolate("linear", limit_direction="both")
    s = s.fillna(0.0)
    return s.values

def check_quality(seg, min_len=100, min_std=1e-6):
    if seg is None: return False
    if len(seg) < min_len: return False
    if np.std(seg) < min_std: return False
    return True

def normalize_image(img):
    vmin, vmax = np.min(img), np.max(img)
    if vmax - vmin < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - vmin) / (vmax - vmin)
    return (img * 255).astype(np.uint8)

def generate_cwt(signal, fs):
    signal = signal - np.mean(signal)
    scales = np.arange(1, 65)
    coef, _ = pywt.cwt(signal, scales, "cmor1.5-1.0", sampling_period=1.0/fs)
    cwt_img = cv2.resize(normalize_image(np.abs(coef)), (IMAGE_SIZE, IMAGE_SIZE))
    return cwt_img

def generate_stft(signal, fs, nperseg=64):
    signal = signal - np.mean(signal)
    n = len(signal)
    if n < 4: return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    nperseg = min(nperseg, n)
    _f, _t, Zxx = stft(signal, fs, nperseg=nperseg)
    spec = cv2.resize(normalize_image(np.flipud(np.log1p(np.abs(Zxx)))), (IMAGE_SIZE, IMAGE_SIZE))
    return spec

def generate_gadf(signal):
    signal = signal - np.mean(signal)
    target_len = IMAGE_SIZE
    if len(signal) > target_len:
        idx = np.linspace(0, len(signal) - 1, target_len).astype(int)
        signal = signal[idx]
    elif len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)), mode="edge")
    X = signal.reshape(1, -1)
    gadf = GramianAngularField(image_size=IMAGE_SIZE, method="difference")
    return normalize_image(gadf.fit_transform(X)[0])

def slice_by_time(t, x, start, end):
    if t is None or x is None: return None
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    if i1 <= i0: return None
    return x[i0:i1]

def read_sampling_rate(dir_path):
    sr_path = os.path.join(dir_path, "sampling rate")
    if os.path.exists(sr_path):
        try: return float(open(sr_path, "r").read().strip())
        except: pass
    return None



# SENSOR LOADERS

def load_metab_signals_only(act_path):
    """View 4 requires VE, HR, BF signals, but labels come from Manifest."""
    mdir = os.path.join(act_path, "Metabolics_System")
    data_path = os.path.join(mdir, "Data.csv")
    if not os.path.exists(data_path): return None

    df = pd.read_csv(data_path, header=None)
    # 0:Time, 5:BF, 6:VE, 8:HR
    if df.shape[1] < 9: return None 
    
    t = df[0].values.astype(float)
    bf = clean_signal(df[5].values.astype(float))
    ve = clean_signal(df[6].values.astype(float))
    hr = clean_signal(df[8].values.astype(float))
    
    fs = 1.0
    if len(t) > 1:
        dt = np.median(np.diff(t))
        if dt > 0: fs = 1.0/dt

    return {"time": t, "bf": bf, "ve": ve, "hr": hr, "fs": fs}

def load_adm_accel(act_path):
    acc_dir = os.path.join(act_path, "ADM_Accel")
    data_path = os.path.join(acc_dir, "Data.csv")
    if not os.path.exists(data_path): return None

    df = pd.read_csv(data_path, header=None)
    t = df[0].values

    def mag_from_block(start_col):
        ax = clean_signal(df[start_col].values)
        ay = clean_signal(df[start_col + 1].values)
        az = clean_signal(df[start_col + 2].values)
        return np.sqrt(ax**2 + ay**2 + az**2)

    waist_mag = mag_from_block(2)
    la, ra = mag_from_block(20), mag_from_block(29)
    lf, rf = mag_from_block(38), mag_from_block(47)
    leg_mag = np.nanmean(np.vstack([la, ra, lf, rf]), axis=0)

    fs = read_sampling_rate(acc_dir)
    if fs is None: fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 128.0

    return {"time": t, "waist_mag": waist_mag, "leg_mag": leg_mag, "fs": fs}

def load_emg(act_path):
    emg_dir = os.path.join(act_path, "EMG")
    data_path = os.path.join(emg_dir, "Data.csv")
    if not os.path.exists(data_path): return None

    df = pd.read_csv(data_path, header=None)
    t = df[0].values
    chans = [clean_signal(df[c].values) for c in range(1, df.shape[1])]
    emg_avg = np.mean(np.abs(chans), axis=0)

    fs = read_sampling_rate(emg_dir)
    if fs is None: fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 1000.0

    return {"time": t, "emg_avg": emg_avg, "fs": fs}

def load_empatica_accel(act_path):
    sdir = os.path.join(act_path, "Empatica_Accel")
    data_path = os.path.join(sdir, "Data.csv")
    if not os.path.exists(data_path): return None

    df = pd.read_csv(data_path, header=None)
    t = df[0].values
    
    fs = read_sampling_rate(sdir)
    if fs is None: 
        dt = np.diff(t)
        fs = 1.0 / np.median(dt[dt > 0]) if len(dt) > 0 else 32.0

    def mag_cols(start_col):
        if start_col + 2 >= df.shape[1]: return None
        ax = clean_signal(df.iloc[:, start_col].values)
        ay = clean_signal(df.iloc[:, start_col+1].values)
        az = clean_signal(df.iloc[:, start_col+2].values)
        return np.sqrt(ax**2 + ay**2 + az**2)

    lw, rw = mag_cols(2), mag_cols(5)
    if lw is None and rw is None: return None
    
    if lw is not None and rw is not None: wrist_mag = 0.5 * (lw + rw)
    elif lw is not None: wrist_mag = lw
    else: wrist_mag = rw

    return {"time": t, "wrist_mag": wrist_mag, "fs": fs}

def load_empatica_physio(act_path):
    sdir = os.path.join(act_path, "Empatica_Physio")
    data_path = os.path.join(sdir, "Data.csv")
    if not os.path.exists(data_path): return None

    df = pd.read_csv(data_path, header=None)
    t = df[0].values

    def get_ch(idx):
        if idx >= df.shape[1]: return None
        col = df.iloc[:, idx]
        return clean_signal(col.values) if not col.isna().all() else None

    eda1, eda2 = get_ch(2), get_ch(4)
    if eda1 is None and eda2 is None: return None

    streams = [x for x in [eda1, eda2] if x is not None]
    eda = np.nanmean(np.vstack(streams), axis=0)

    fs = 4.0
    if len(t) > 1:
        dt = np.median(np.diff(t))
        if dt > 0: fs = 1.0/dt

    return {"time": t, "eda": eda, "fs": fs}



# VIEW BUILDERS

def build_view_core(acc_data, emg_data, start, end):
    if acc_data is None or emg_data is None: return None
    seg_acc = slice_by_time(acc_data["time"], acc_data["waist_mag"], start, end)
    seg_emg = slice_by_time(emg_data["time"], emg_data["emg_avg"], start, end)
    if not (check_quality(seg_acc) and check_quality(seg_emg)): return None
    r = generate_cwt(seg_acc, acc_data["fs"])
    g = generate_stft(seg_emg, emg_data["fs"], nperseg=64)
    b = generate_gadf(seg_acc)
    return cv2.merge([r, g, b])

def build_view_legs(acc_data, start, end):
    if acc_data is None: return None
    seg_leg = slice_by_time(acc_data["time"], acc_data["leg_mag"], start, end)
    if not check_quality(seg_leg): return None
    r = generate_cwt(seg_leg, acc_data["fs"])
    g = generate_stft(seg_leg, acc_data["fs"], nperseg=64)
    b = generate_gadf(seg_leg)
    return cv2.merge([r, g, b])

def build_view_wrist(emp_acc, emp_phys, start, end):
    if emp_acc is None or emp_phys is None: return None
    seg_w = slice_by_time(emp_acc["time"], emp_acc["wrist_mag"], start, end)
    seg_e = slice_by_time(emp_phys["time"], emp_phys["eda"], start, end)
    if not (check_quality(seg_w) and check_quality(seg_e, min_len=10)): return None
    r = generate_cwt(seg_w, emp_acc["fs"])
    g = generate_stft(seg_e, emp_phys["fs"], nperseg=32)
    b = generate_gadf(seg_w)
    return cv2.merge([r, g, b])

def build_view_metab(meta, start, end):
    if meta is None: return None
    center = 0.5 * (start + end)
    half = LABEL_SMOOTH_SEC / 2.0
    
    s_ve = slice_by_time(meta["time"], meta["ve"], center-half, center+half)
    s_hr = slice_by_time(meta["time"], meta["hr"], center-half, center+half)
    s_bf = slice_by_time(meta["time"], meta["bf"], center-half, center+half)

    if s_ve is None or s_hr is None or s_bf is None: return None
    if len(s_ve)<8 or len(s_hr)<8 or len(s_bf)<8: return None

    def z(x):
        x = x - np.nanmean(x)
        std = np.nanstd(x)
        return x/std if std > 1e-9 else np.zeros_like(x)

    sig = np.nanmean(np.vstack([z(s_ve), z(s_hr), z(s_bf)]), axis=0)
    if not check_quality(sig, min_len=8): return None

    fs = meta["fs"]
    r = generate_cwt(sig, fs)
    g = generate_stft(sig, fs, nperseg=8)
    b = generate_gadf(sig)
    return cv2.merge([r, g, b])

VIEW_DEFS = [
    {"id": 1, "name": "Core_Accel_EMG", "builder": build_view_core},
    {"id": 2, "name": "Legs_IMU",       "builder": build_view_legs},
    {"id": 3, "name": "Wrist_Physio",   "builder": build_view_wrist},
    {"id": 4, "name": "Metab_Global_NoVO2VCO2", "builder": build_view_metab},
]



# MAIN EXECUTION


if __name__ == "__main__":
    if not os.path.exists(SOURCE_MANIFEST_PATH):
        raise FileNotFoundError(f"Source manifest not found: {SOURCE_MANIFEST_PATH}")

    df_src = pd.read_csv(SOURCE_MANIFEST_PATH)
    if "subject_id" in df_src.columns:
        df_src = df_src[df_src["subject_id"] == SUBJECT_ID].copy()

    # Determine column names flexibly
    act_col = next((c for c in ["activity_name", "Activity", "activity"] if c in df_src.columns), None)
    idx_col = next((c for c in ["window_idx", "win_idx"] if c in df_src.columns), None)
    start_col = next((c for c in ["window_start_s", "start"] if c in df_src.columns), None)
    end_col = next((c for c in ["window_end_s", "end"] if c in df_src.columns), None)

    if not all([act_col, idx_col, start_col, end_col]):
        raise ValueError("Source manifest missing required columns.")

    full_rows = []
    stats = defaultdict(lambda: {"valid": 0, "skipped": 0})
    total_valid_images = 0

    # Group by activity
    for activity, df_act in df_src.groupby(act_col):
        # Resolve path
        cands = [activity, activity.replace(" ", "_")]
        act_path = None
        for c in cands:
            p = os.path.join(SUBJECT_DIR, c)
            if os.path.isdir(p):
                act_path = p
                break
        
        if not act_path:
            continue

        print(f"\n Activity: {activity}")

        # Load sensors
        acc_data = load_adm_accel(act_path)
        emg_data = load_emg(act_path)
        emp_acc = load_empatica_accel(act_path)
        emp_phys = load_empatica_physio(act_path)
        meta_data = load_metab_signals_only(act_path)

        act_out_root = os.path.join(OUTPUT_ROOT, str(activity).replace(" ", "_"))
        os.makedirs(act_out_root, exist_ok=True)
        
        valid_count_activity = 0

        for _, row in df_act.iterrows():
            w_idx = int(row[idx_col])
            start = float(row[start_col])
            end = float(row[end_col])
            
            meta_payload = {
                "subject_id": SUBJECT_ID,
                "activity_name": activity,
                "window_idx": w_idx,
                "window_start_s": start,
                "window_end_s": end,
                "window_center_s": (start+end)/2,
                "data_status": "PENDING"
            }
            for c in KEEP_COLS:
                if c in row: meta_payload[c] = row[c]

            for view in VIEW_DEFS:
                out_row = meta_payload.copy()
                out_row["view_id"] = view["id"]
                out_row["view_name"] = view["name"]
                
                img = None
                try:
                    builder = view["builder"]
                    if view["name"] == "Core_Accel_EMG":
                        img = builder(acc_data, emg_data, start, end)
                    elif view["name"] == "Legs_IMU":
                        img = builder(acc_data, start, end)
                    elif view["name"] == "Wrist_Physio":
                        img = builder(emp_acc, emp_phys, start, end)
                    elif view["name"] == "Metab_Global_NoVO2VCO2":
                        img = builder(meta_data, start, end)
                except Exception as e:
                    out_row["data_status"] = "ERROR"
                    out_row["missing_reason"] = str(e)
                if img is not None:
                    v_dir = os.path.join(act_out_root, view["name"])
                    os.makedirs(v_dir, exist_ok=True)
                    fname = f"{SUBJECT_ID}_{str(activity).replace(' ', '_')}_win{w_idx:05d}_view{view['id']}.png"
                    
                    save_path = os.path.join(v_dir, fname)  # Define full path
                    cv2.imwrite(save_path, img)
                    
                    # <--- ADD THIS LINE HERE
                    out_row["image_rel_path"] = os.path.relpath(save_path, OUTPUT_ROOT) 
                    
                    out_row["image_filename"] = fname
                    out_row["data_status"] = "VALID"
                    out_row["image_generated"] = True
                    
                    stats[activity]["valid"] += 1
                    valid_count_activity += 1
                    total_valid_images += 1
                else:
                    out_row["data_status"] = "SKIPPED"
                    if "missing_reason" not in out_row:
                        out_row["missing_reason"] = "Signal Missing/Poor Quality"
                    stats[activity]["skipped"] += 1
                
                full_rows.append(out_row)

        print(f"   Valid images for {activity}: {valid_count_activity}")

    # Save Manifest
    if full_rows:
        df_out = pd.DataFrame(full_rows)
        df_out.to_csv(OUTPUT_MANIFEST, index=False)
        
        print("\n" + "="*80)
        print(f"📊 Manifest written to: {OUTPUT_MANIFEST}")
        print("\nPer-activity summary:")
        print(f"{'Activity':<35} | {'Valid':>8} | {'Skipped/Err':>12}")
        print("-" * 60)
        for act, s in stats.items():
            print(f"{act:<35} | {s['valid']:>8} | {s['skipped']:>12}")
        print("-" * 60)
        print(f" TOTAL VALID IMAGES GENERATED: {total_valid_images}")
    else:
        print("❌ No images generated.")
