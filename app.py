"""
Mouse Blink Analysis Platform
Integrates advanced signal processing from core.py with Anthropic-inspired warm design
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import streamlit as st
import tempfile
from io import BytesIO
import sqlite3
from datetime import datetime
import pywt
import warnings
from scipy.signal import find_peaks

# --- Page Config ---
# Important: set_page_config should be the first Streamlit command.
st.set_page_config(
    layout="wide",
    page_title="Mouse Blink Analysis",
    page_icon="👁",
    initial_sidebar_state="expanded"
)

# --- Anthropic Design System CSS ---
st.markdown("""
<style>
    :root {
        --background: #F7F0E8;
        --bg: #F7F0E8;
        --surface1: #EFE6DA;
        --surface2: #DDD0C0;
        --surface3: #C8B9A8;
        --border: #DDD0C0;
        --border-visible: #C8B9A8;
        --text1: #1F1815;
        --text2: #5F4B3F;
        --text3: #7A6252;
        --text4: #A8917F;
        --accent: #B85C38;
        --accent-subtle: #F8E6D8;
        --success: #22C55E;
        --success-bg: #F0FDF4;
        --warning: #F59E0B;
        --warning-bg: #FFFBEB;
        --error: #EF4444;
        --error-bg: #FEF2F2;
        --radius-control: 4px;
        --radius-component: 8px;
        --radius-container: 12px;
        --font-display: "Inter", system-ui, sans-serif;
        --font-mono: "JetBrains Mono", monospace;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text1);
        font-family: var(--font-display);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text1);
        font-weight: 500;
        letter-spacing: 0;
        line-height: 1.3;
    }

    [data-testid="stMetric"] {
        background-color: var(--surface1);
        border: none;
        border-radius: var(--radius-component);
        padding: 24px;
    }

    [data-testid="stHorizontalBlock"] > div {
        background-color: var(--surface1);
        border-radius: var(--radius-component);
        padding: 16px;
        margin: 4px;
    }

    .stButton > button {
        background-color: var(--text1);
        color: var(--background);
        border: none;
        border-radius: var(--radius-control);
        padding: 12px 20px;
        font-weight: 500;
        font-size: 14px;
        transition: all 150ms ease-out;
    }

    .stButton > button:hover {
        background-color: var(--surface2);
        transform: scale(0.98);
    }

    .stButton > button:active {
        transform: scale(0.96);
    }

    .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: var(--text1);
        border: 1px solid var(--border);
    }

    [data-testid="stFileUploadDropzone"] {
        background-color: var(--surface1);
        border: 2px dashed var(--border);
        border-radius: var(--radius-component);
        padding: 32px;
        transition: all 150ms ease-out;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--accent);
        background-color: var(--accent-subtle);
    }

    [data-testid="stSlider"] label {
        color: var(--text2);
        font-size: 14px;
        font-weight: 500;
    }

    .main-header {
        font-size: 36px;
        font-weight: 500;
        color: var(--text1);
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .sub-header {
        font-size: 16px;
        color: var(--text2);
        margin-bottom: 32px;
    }

    .section-header {
        font-size: 20px;
        font-weight: 500;
        color: var(--text1);
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    .info-box {
        background-color: var(--surface1);
        border-radius: var(--radius-component);
        padding: 16px 20px;
        color: var(--text2);
        margin: 16px 0;
    }

    .info-box.success {
        background-color: var(--success-bg);
        color: var(--success);
    }

    .info-box.warning {
        background-color: var(--warning-bg);
        color: var(--warning);
    }

    .info-box.error {
        background-color: var(--error-bg);
        color: var(--error);
    }

    .metric-card {
        background-color: var(--surface1);
        border-radius: var(--radius-component);
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 500;
        color: var(--text1);
        line-height: 1.1;
    }

    .metric-label {
        font-size: 14px;
        color: var(--text2);
        margin-top: 4px;
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface1);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text1);
    }

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 24px 0;
    }

    .streamlit-expanderHeader {
        background-color: var(--surface1);
        border-radius: var(--radius-control);
        color: var(--text1);
    }

    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
    }

    [data-testid="stDataFrame"] {
        background-color: var(--surface1);
        border-radius: var(--radius-component);
    }

    [data-testid="stVideo"] {
        border-radius: var(--radius-component);
        overflow: hidden;
    }

    .stProgress > div > div {
        background-color: var(--accent);
    }

    .stSpinner > div {
        border-color: var(--accent);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text3);
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text2);
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent);
    }

    .stDownloadButton > button {
        background-color: var(--surface2);
        color: var(--text1);
        border: 1px solid var(--border);
        border-radius: var(--radius-control);
        padding: 10px 16px;
    }

    .stDownloadButton > button:hover {
        background-color: var(--surface3);
    }

    .stTooltipContent {
        background-color: var(--surface2);
        color: var(--text1);
        border-radius: var(--radius-control);
        padding: 8px 12px;
    }

    .js-plotly-plot .plotly .modebar {
        background: var(--surface1);
    }
</style>
""", unsafe_allow_html=True)

# --- Import custom segmentation model ---
try:
    from ME_FPN_model import FPN as MyCustomFPN
except Exception as e:
    st.error(f"ERROR: Could not import custom model. Details: {e}")

    class MyCustomFPN:
        pass

# --- Global Constants ---
YOLO_MODEL_FILENAME = "detection model.pt"
SEG_MODEL_FILENAME = "segmentation model.pth"
DB_FILE = "analysis_history.db"


# --- Database Functions ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_timestamp TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            interp_blinks INTEGER,
            zero_blinks INTEGER,
            interp_avg_min_norm REAL,
            zero_avg_min_norm REAL,
            analysis_duration_s REAL
        )
    ''')
    conn.commit()
    conn.close()


def save_results_to_db(filename, stats):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_results (
            analysis_timestamp, original_filename, interp_blinks,
            zero_blinks, interp_avg_min_norm, zero_avg_min_norm, analysis_duration_s
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        stats.get('interp_blinks', 0),
        stats.get('zero_blinks', 0),
        stats.get('interp_avg_min_norm'),
        stats.get('zero_avg_min_norm'),
        stats.get('analysis_duration_s')
    ))
    conn.commit()
    conn.close()


def load_results_from_db():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_FILE)
    try:
        return pd.read_sql_query("SELECT * FROM analysis_results ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


# --- Model Loading ---
@st.cache_resource
def load_detection_model(path):
    if not os.path.exists(path):
        return None

    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Could not load YOLO detection model: {e}")
        return None


@st.cache_resource
def load_segmentation_model(path):
    if not os.path.exists(path):
        return None

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MyCustomFPN(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=1,
            in_channels=3
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Could not load segmentation model: {e}")
        return None


# --- Core Functions from core.py ---
def preprocess_for_segmentation(eye_crop, input_size=(256, 256)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img).unsqueeze(0).to(device)


def postprocess_segmentation(output_mask, original_crop_shape, seg_threshold):
    mask_numpy = output_mask.squeeze().cpu().detach().numpy()
    probabilities = 1 / (1 + np.exp(-mask_numpy))
    probabilities_resized = cv2.resize(
        probabilities,
        (original_crop_shape[1], original_crop_shape[0])
    )
    return np.sum((probabilities_resized > seg_threshold).astype(np.uint8))


def clean_numeric_signal(data):
    """
    Convert signal to safe float64 numpy array.
    Handles NaN, inf, object dtype, and empty values.
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)

    if data.size == 0:
        return data

    if np.any(~np.isfinite(data)):
        valid = np.isfinite(data)

        if not np.any(valid):
            return np.zeros_like(data, dtype=np.float64)

        x = np.arange(len(data))
        data = np.interp(x, x[valid], data[valid])

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.astype(np.float64)


def wavelet_denoise(data, wavelet='db4', level=2):
    """
    Robust wavelet denoising.

    This version prevents PyWavelets from crashing when:
    - input is empty
    - input contains NaN or inf
    - input is too short
    - requested decomposition level is too high
    - PyWavelets raises a low-level ValueError
    """
    data = clean_numeric_signal(data)

    if data.size == 0:
        return data

    if len(data) < 8:
        return data

    try:
        wavelet_obj = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(data_len=len(data), filter_len=wavelet_obj.dec_len)
        safe_level = min(level, max_level)

        if safe_level < 1:
            return data

        coeff = pywt.wavedec(data, wavelet_obj, mode="per", level=safe_level)

        if len(coeff) < 2:
            return data

        detail_coeff = coeff[-1]

        if detail_coeff.size == 0:
            return data

        sigma = (1 / 0.6745) * np.median(np.abs(detail_coeff))

        if not np.isfinite(sigma):
            return data

        uthresh = sigma * np.sqrt(2 * np.log(len(data)))

        if not np.isfinite(uthresh):
            return data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            coeff[1:] = [
                pywt.threshold(i, value=uthresh, mode='soft')
                for i in coeff[1:]
            ]

        result = pywt.waverec(coeff, wavelet_obj, mode="per")
        result = clean_numeric_signal(result)

        if len(result) < len(data):
            result = np.pad(result, (0, len(data) - len(result)), mode='edge')

        return result[:len(data)]

    except Exception:
        return data


def find_true_valleys(signal, peak_indices, search_window):
    signal = clean_numeric_signal(signal)
    valley_indices = []
    n = len(signal)

    if n == 0:
        return np.array([], dtype=int)

    for pk in peak_indices:
        pk = int(pk)

        if pk < 0 or pk >= n:
            continue

        start = pk
        end = min(n, pk + max(1, int(search_window)))

        if end <= start:
            continue

        local_valley = start + int(np.argmin(signal[start:end]))
        valley_indices.append(local_valley)

    return np.array(valley_indices, dtype=int)


def local_baseline_norm(values, roll_max_at_idx):
    values = clean_numeric_signal(values)
    roll_max_at_idx = clean_numeric_signal(roll_max_at_idx)

    if values.size == 0 or roll_max_at_idx.size == 0:
        return 0.0

    min_len = min(len(values), len(roll_max_at_idx))
    values = values[:min_len]
    roll_max_at_idx = roll_max_at_idx[:min_len]

    denominators = np.where(roll_max_at_idx > 0, roll_max_at_idx, 1.0)
    ratio = values / denominators
    ratio = ratio[np.isfinite(ratio)]

    if len(ratio) == 0:
        return 0.0

    return float(np.mean(ratio))


def normalize_by_initial_baseline(signal, fps, baseline_sec=1.2):
    signal = clean_numeric_signal(signal)

    if len(signal) == 0:
        return signal, 1.0

    baseline_frames = max(1, int(fps * baseline_sec))
    baseline_frames = min(baseline_frames, len(signal))

    initial_values = signal[:baseline_frames]
    valid = initial_values[initial_values > 0]

    if len(valid) == 0:
        valid = signal[signal > 0]

    if len(valid) == 0:
        baseline = 1.0
    else:
        baseline = np.percentile(valid, 95)

        if not np.isfinite(baseline) or baseline <= 0:
            baseline = 1.0

    return signal / baseline, baseline


# --- Core Analysis ---
def run_analysis(video_path, original_filename, yolo_model, seg_model, config):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None, None, None, None

    frame_data_list = []
    progress_bar = st.progress(0, text="Extracting eye features...")

    # Stage 1: Extract eye area sequence frame by frame
    for frame_count in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = (frame_count + 1) / fps
        current_area = 0
        status = 'no_eye'

        try:
            yolo_results = yolo_model.predict(
                frame,
                conf=config['YOLO_CONF_THRESHOLD'],
                classes=[0],
                verbose=False
            )

            eye_box = (
                yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
                if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0
                else None
            )

            if eye_box is not None:
                x1, y1, x2, y2 = eye_box

                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))

                if x2 > x1 and y2 > y1:
                    eye_crop = frame[y1:y2, x1:x2]

                    if eye_crop.size > 0:
                        input_tensor = preprocess_for_segmentation(eye_crop)

                        with torch.no_grad():
                            seg_output = seg_model(input_tensor)

                        current_area = postprocess_segmentation(
                            seg_output,
                            eye_crop.shape[:2],
                            config['SEG_THRESHOLD']
                        )
                        status = 'processed'

        except Exception:
            current_area = 0
            status = 'error'

        frame_data_list.append({
            'timestamp': timestamp,
            'area': float(current_area),
            'status': status
        })

        progress_bar.progress(
            min(1.0, (frame_count + 1) / total_frames),
            text=f"Analyzing... {frame_count + 1}/{total_frames}"
        )

    cap.release()
    progress_bar.empty()

    if not frame_data_list:
        return None, None, None, None

    df = pd.DataFrame(frame_data_list)
    df['area'] = pd.to_numeric(df['area'], errors='coerce').fillna(0.0)

    if df['area'].max() <= 0:
        return None, None, None, None

    window_frames = max(1, int(config['BASELINE_WINDOW_SEC'] * fps))
    min_distance_frames = max(1, int(fps * 0.15))
    valley_search_frames = max(1, int(config['VALLEY_SEARCH_FRAMES']))

    # ==========================================================
    # Method 1: Interpolation
    # ==========================================================
    df['interp_area'] = df['area'].replace(0, np.nan)
    df.loc[df['status'] != 'processed', 'interp_area'] = np.nan
    df['interp_area'] = df['interp_area'].interpolate(method='linear').bfill().ffill()

    if df['interp_area'].isna().all():
        return None, None, None, None

    df['interp_area'] = pd.to_numeric(df['interp_area'], errors='coerce').fillna(0.0)

    raw_sig_interp = clean_numeric_signal(df['interp_area'].to_numpy(dtype=np.float64))
    denoised_sig_interp = wavelet_denoise(raw_sig_interp)

    roll_max_interp = (
        df['interp_area']
        .rolling(window=window_frames, center=True, min_periods=1)
        .max()
        .to_numpy(dtype=np.float64)
    )
    roll_max_interp = clean_numeric_signal(roll_max_interp)

    drop_ratio_interp = (roll_max_interp - denoised_sig_interp) / (roll_max_interp + 1e-6)
    drop_ratio_interp = clean_numeric_signal(drop_ratio_interp)

    peaks_interp, _ = find_peaks(
        drop_ratio_interp,
        height=config['DROP_RATIO'],
        distance=min_distance_frames,
        prominence=0.10,
        wlen=max(1, int(fps * 2))
    )

    valley_idx_interp = find_true_valleys(raw_sig_interp, peaks_interp, valley_search_frames)

    # ==========================================================
    # Method 2: Zero method
    # ==========================================================
    raw_sig_zero = clean_numeric_signal(df['area'].to_numpy(dtype=np.float64))
    denoised_sig_zero = wavelet_denoise(raw_sig_zero)

    roll_max_zero = (
        df['area']
        .rolling(window=window_frames, center=True, min_periods=1)
        .max()
        .to_numpy(dtype=np.float64)
    )
    roll_max_zero = clean_numeric_signal(roll_max_zero)

    drop_ratio_zero = (roll_max_zero - denoised_sig_zero) / (roll_max_zero + 1e-6)
    drop_ratio_zero = clean_numeric_signal(drop_ratio_zero)

    peaks_zero, _ = find_peaks(
        drop_ratio_zero,
        height=config['DROP_RATIO'],
        distance=min_distance_frames,
        prominence=0.10,
        wlen=max(1, int(fps * 2))
    )

    valley_idx_zero = find_true_valleys(raw_sig_zero, peaks_zero, valley_search_frames)

    # ==========================================================
    # Metrics calculation
    # ==========================================================
    if len(valley_idx_interp) > 0:
        interp_avg_min_area = float(np.mean(raw_sig_interp[valley_idx_interp]))
        interp_avg_min_norm = local_baseline_norm(
            raw_sig_interp[valley_idx_interp],
            roll_max_interp[valley_idx_interp]
        )
    else:
        interp_avg_min_area = 0.0
        interp_avg_min_norm = 0.0

    if len(valley_idx_zero) > 0:
        zero_avg_min_area = float(np.mean(raw_sig_zero[valley_idx_zero]))
        zero_avg_min_norm = local_baseline_norm(
            raw_sig_zero[valley_idx_zero],
            roll_max_zero[valley_idx_zero]
        )
    else:
        zero_avg_min_area = 0.0
        zero_avg_min_norm = 0.0

    interp_raw_norm_area = local_baseline_norm(raw_sig_interp, roll_max_interp)
    interp_denoise_norm_area = local_baseline_norm(denoised_sig_interp, roll_max_interp)
    zero_raw_norm_area = local_baseline_norm(raw_sig_zero, roll_max_zero)
    zero_denoise_norm_area = local_baseline_norm(denoised_sig_zero, roll_max_zero)

    # ==========================================================
    # Visualization
    # ==========================================================
    raw_sig_interp_norm, interp_fixed_baseline = normalize_by_initial_baseline(
        raw_sig_interp,
        fps,
        config['BASELINE_WINDOW_SEC']
    )
    denoised_sig_interp_norm = denoised_sig_interp / interp_fixed_baseline
    roll_max_interp_norm = roll_max_interp / interp_fixed_baseline

    raw_sig_zero_norm, zero_fixed_baseline = normalize_by_initial_baseline(
        raw_sig_zero,
        fps,
        config['BASELINE_WINDOW_SEC']
    )
    denoised_sig_zero_norm = denoised_sig_zero / zero_fixed_baseline
    roll_max_zero_norm = roll_max_zero / zero_fixed_baseline

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={'hspace': 0.38}
    )

    ax1.plot(
        df['timestamp'],
        raw_sig_interp_norm,
        label='Raw normalized (Interpolated)',
        color='lightgray',
        alpha=0.6
    )
    ax1.plot(
        df['timestamp'],
        roll_max_interp_norm,
        label='Baseline normalized',
        color='green',
        linestyle='--',
        alpha=0.7
    )
    ax1.plot(
        df['timestamp'],
        denoised_sig_interp_norm,
        label='Denoised normalized',
        color='blue',
        linewidth=1.5
    )
    ax1.axhline(
        1.0,
        color='black',
        linestyle=':',
        linewidth=1,
        alpha=0.6,
        label='Initial baseline = 1'
    )

    if len(valley_idx_interp) > 0:
        ax1.scatter(
            df['timestamp'].iloc[valley_idx_interp],
            raw_sig_interp_norm[valley_idx_interp],
            color='red',
            s=80,
            zorder=5,
            label='Blink valley normalized'
        )

    ax1.set_title(f'Method 1: Missing values INTERPOLATED | Blinks: {len(peaks_interp)}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Eye Area')
    ax1.tick_params(axis='x', which='both', labelbottom=True)
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        df['timestamp'],
        raw_sig_zero_norm,
        label='Raw normalized (Direct 0)',
        color='lightgray',
        alpha=0.6
    )
    ax2.plot(
        df['timestamp'],
        roll_max_zero_norm,
        label='Baseline normalized',
        color='green',
        linestyle='--',
        alpha=0.7
    )
    ax2.plot(
        df['timestamp'],
        denoised_sig_zero_norm,
        label='Denoised normalized',
        color='purple',
        linewidth=1.5
    )
    ax2.axhline(
        1.0,
        color='black',
        linestyle=':',
        linewidth=1,
        alpha=0.6,
        label='Initial baseline = 1'
    )

    if len(valley_idx_zero) > 0:
        ax2.scatter(
            df['timestamp'].iloc[valley_idx_zero],
            raw_sig_zero_norm[valley_idx_zero],
            color='red',
            s=80,
            zorder=5,
            label='Blink valley normalized'
        )

    ax2.set_title(f'Method 2: Missing values kept as ZERO | Blinks: {len(peaks_zero)}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normalized Eye Area')
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=3.0)

    duration = float(df['timestamp'].max())

    stats = {
        "filename": original_filename,
        "analysis_duration_s": round(duration, 2),
        "interp_blinks": int(len(peaks_interp)),
        "interp_avg_min_area": round(float(interp_avg_min_area), 2),
        "interp_avg_min_norm": round(float(interp_avg_min_norm), 4),
        "interp_raw_avg_area": round(float(np.mean(raw_sig_interp)), 2),
        "interp_raw_norm_area": round(float(interp_raw_norm_area), 4),
        "interp_denoise_avg_area": round(float(np.mean(denoised_sig_interp)), 2),
        "interp_denoise_norm_area": round(float(interp_denoise_norm_area), 4),
        "zero_blinks": int(len(peaks_zero)),
        "zero_avg_min_area": round(float(zero_avg_min_area), 2),
        "zero_avg_min_norm": round(float(zero_avg_min_norm), 4),
        "zero_raw_avg_area": round(float(np.mean(raw_sig_zero)), 2),
        "zero_raw_norm_area": round(float(zero_raw_norm_area), 4),
        "zero_denoise_avg_area": round(float(np.mean(denoised_sig_zero)), 2),
        "zero_denoise_norm_area": round(float(zero_denoise_norm_area), 4),
    }

    return None, df, fig, stats


# --- Main App ---
def show_main_app():
    st.markdown(
        '<h1 class="main-header">👁 Mouse Blink Analysis</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Advanced signal processing for eye closure detection using wavelet transform and dynamic baseline analysis.</p>',
        unsafe_allow_html=True
    )

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Upload Video")

        uploaded_file = st.file_uploader(
            "Select a video file",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            st.session_state.analysis_results = None

        st.markdown("---")
        st.markdown("### Analysis Parameters")

        drop_ratio = st.slider(
            "Blink Depth Sensitivity",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Threshold for identifying a blink. Higher value = only deep closures count."
        )

        baseline_window = st.slider(
            "Baseline Window (seconds)",
            min_value=0.5,
            max_value=10.0,
            value=1.2,
            step=0.1,
            help="How many seconds to look back/forward to determine the normal open area."
        )

        valley_search = st.slider(
            "Valley Search Window",
            min_value=1,
            max_value=20,
            value=8,
            step=1,
            help="Frames to search forward for true blink valley after peak."
        )

        st.markdown("---")

        start_button = st.button(
            "🚀 Start Analysis",
            disabled=(uploaded_file is None),
            use_container_width=True
        )

    if start_button:
        init_db()

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                video_path = tmpfile.name

            try:
                with st.spinner("Loading AI models..."):
                    yolo_model = load_detection_model(YOLO_MODEL_FILENAME)
                    seg_model = load_segmentation_model(SEG_MODEL_FILENAME)

                if yolo_model is None:
                    st.error(f"YOLO model not found or failed to load: {YOLO_MODEL_FILENAME}")
                    return

                if seg_model is None:
                    st.error(f"Segmentation model not found or failed to load: {SEG_MODEL_FILENAME}")
                    return

                config = {
                    'YOLO_CONF_THRESHOLD': 0.6,
                    'SEG_THRESHOLD': 0.4,
                    'DROP_RATIO': drop_ratio,
                    'BASELINE_WINDOW_SEC': baseline_window,
                    'VALLEY_SEARCH_FRAMES': valley_search
                }

                video_bytes, results_df, results_fig, stats = run_analysis(
                    video_path,
                    uploaded_file.name,
                    yolo_model,
                    seg_model,
                    config
                )

                if video_bytes or results_df is not None:
                    st.session_state.analysis_results = {
                        "video_bytes": video_bytes,
                        "results_df": results_df,
                        "results_fig": results_fig,
                        "stats": stats
                    }

                    try:
                        save_results_to_db(uploaded_file.name, stats)
                        st.success("Analysis complete and saved to history.")
                    except Exception as e:
                        st.warning(f"Could not save to database: {e}")
                else:
                    st.error(
                        "Analysis failed. No valid eye area signal was detected. "
                        "Please try a clearer video with the eye visible for most frames."
                    )

            except Exception as e:
                st.error(f"Analysis failed due to an unexpected error: {e}")

            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        stats = results["stats"]
        results_df = results["results_df"]
        results_fig = results["results_fig"]

        st.markdown(
            '<h2 class="section-header">Analysis Results</h2>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Method 1 Blinks",
                f"{stats['interp_blinks']}",
                help="Using linear interpolation for missing frames"
            )

        with col2:
            st.metric(
                "Method 2 Blinks",
                f"{stats['zero_blinks']}",
                help="Using zero for missing frames"
            )

        with col3:
            interp_norm = stats.get('interp_avg_min_norm', 0)
            st.metric(
                "Avg Blink Depth",
                f"{interp_norm:.2%}",
                help="Average normalized area at blink valleys"
            )

        with col4:
            duration = stats.get('analysis_duration_s', 0)
            blink_rate = (
                stats.get('interp_blinks', 0) / duration * 60
                if duration > 0
                else 0
            )
            st.metric(
                "Blink Rate",
                f"{blink_rate:.1f}/min",
                help="Blinks per minute"
            )

        st.markdown("---")

        sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)

        with sub_col1:
            st.metric("Duration", f"{stats.get('analysis_duration_s', 0):.2f}s")

        with sub_col2:
            st.metric("Interp Norm", f"{stats.get('interp_raw_norm_area', 0):.4f}")

        with sub_col3:
            st.metric("Zero Norm", f"{stats.get('zero_raw_norm_area', 0):.4f}")

        with sub_col4:
            st.metric("Denoise Norm", f"{stats.get('interp_denoise_norm_area', 0):.4f}")

        st.markdown("### Signal Processing Visualization")
        st.pyplot(results_fig)

        buf = BytesIO()
        results_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)

        st.download_button(
            "📥 Download HD Plot",
            buf.getvalue(),
            f"{stats.get('filename', 'analysis')}_plot.png",
            "image/png",
            use_container_width=True
        )

        st.markdown("### Frame-by-Frame Data")

        with st.expander("View detailed data", expanded=False):
            st.dataframe(results_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Analysis History")

    init_db()
    history_df = load_results_from_db()

    if not history_df.empty:
        display_df = history_df.copy()
        display_df.columns = [
            'ID',
            'Timestamp',
            'Filename',
            'Method1 Blinks',
            'Method2 Blinks',
            'Interp Norm',
            'Zero Norm',
            'Duration'
        ]

        st.dataframe(display_df, use_container_width=True)

        csv_buffer = BytesIO()
        display_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_buffer.seek(0)

        st.download_button(
            "📥 Download History CSV",
            csv_buffer.getvalue(),
            "analysis_history.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("No analysis history yet. Upload a video to get started.")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: var(--text3); font-size: 12px;'>"
        "Mouse Blink Analysis Platform · Powered by Wavelet Transform Signal Processing"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    show_main_app()
