"""
Mouse Blink Analysis Platform
Integrates advanced signal processing from core.py with Anthropic-inspired warm design

"""

import os
import random
import re
import smtplib
import sqlite3
import ssl
import string
import tempfile
import time
import warnings
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO

# Reduce native-thread pressure before importing scientific stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("YOLO_VERBOSE", "False")

import streamlit as st

# Must be the first Streamlit command.
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
    [data-testid="stMetricLabel"] {
        color: var(--text2);
        font-size: 14px;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: var(--text1);
        font-size: 28px;
        font-weight: 600;
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
</style>
""", unsafe_allow_html=True)


# Light packages only at import time. torch / OpenCV / YOLO are loaded lazily
# so a Cloud libGL or RAM failure does not take down the login page.
_IMPORT_ERROR = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pywt
    from scipy.signal import find_peaks
except Exception as e:
    _IMPORT_ERROR = e
    plt = None
    np = None
    pd = None
    pywt = None
    find_peaks = None

try:
    import bcrypt
    _BCRYPT_ERROR = None
except Exception as e:
    bcrypt = None
    _BCRYPT_ERROR = e


# --- Global Constants ---
YOLO_MODEL_FILENAME = "detection model.pt"
SEG_MODEL_FILENAME = "segmentation model.pth"
DB_FILE = "analysis_history.db"


# ==========================================================
# Auth Helper Functions
# ==========================================================

def get_secret(name, default=""):
    """
    Read config from Streamlit secrets first, then environment variables.

    Streamlit Cloud Secrets can be:
    SMTP_HOST = "smtp.gmail.com"

    Or:
    [smtp]
    host = "smtp.gmail.com"
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    try:
        if "smtp" in st.secrets:
            key_map = {
                "SMTP_HOST": "host",
                "SMTP_PORT": "port",
                "SMTP_USER": "user",
                "SMTP_PASSWORD": "password",
                "SMTP_FROM": "from",
            }
            smtp_key = key_map.get(name)
            if smtp_key and smtp_key in st.secrets["smtp"]:
                return st.secrets["smtp"][smtp_key]
    except Exception:
        pass

    return os.environ.get(name, default)


def is_valid_email(email):
    email = email.strip().lower()
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, hashed_password):
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def generate_verification_code():
    return "".join(random.choices(string.digits, k=6))


def send_verification_email(to_email, code):
    smtp_host = str(get_secret("SMTP_HOST", "")).strip()
    smtp_port = int(str(get_secret("SMTP_PORT", "587") or "587"))
    smtp_user = str(get_secret("SMTP_USER", "")).strip()
    smtp_password = str(get_secret("SMTP_PASSWORD", "")).strip()
    smtp_from = str(get_secret("SMTP_FROM", smtp_user)).strip()

    if not smtp_host or not smtp_user or not smtp_password or not smtp_from:
        return False, (
            "SMTP 邮件配置不完整。请在云平台的 Secrets / Environment Variables 中配置 "
            "SMTP_HOST、SMTP_PORT、SMTP_USER、SMTP_PASSWORD、SMTP_FROM。"
        )

    subject = "Mouse Blink Analysis 注册验证码"

    text_body = f"""
您好！

您的注册验证码是：

{code}

该验证码 10 分钟内有效，请勿泄露给他人。

如果这不是您本人操作，请忽略此邮件。

Mouse Blink Analysis Platform
""".strip()

    html_body = f"""
    <div style="font-family: Arial, sans-serif; color: #1F1815; line-height: 1.6;">
        <h2>Mouse Blink Analysis 注册验证码</h2>
        <p>您好！</p>
        <p>您的注册验证码是：</p>
        <div style="
            font-size: 28px;
            font-weight: bold;
            letter-spacing: 4px;
            background: #F7F0E8;
            padding: 16px 24px;
            border-radius: 8px;
            display: inline-block;
            color: #B85C38;
        ">
            {code}
        </div>
        <p>该验证码 <strong>10 分钟内有效</strong>，请勿泄露给他人。</p>
        <p>如果这不是您本人操作，请忽略此邮件。</p>
        <p style="color: #7A6252; font-size: 12px;">
            Mouse Blink Analysis Platform
        </p>
    </div>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        context = ssl.create_default_context()

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, [to_email], msg.as_string())

        return True, "验证码已发送，请检查邮箱。"

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP 认证失败。请检查发件邮箱和应用专用密码是否正确。"
    except Exception as e:
        return False, f"验证码发送失败：{e}"


# ==========================================================
# Database Functions
# ==========================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS verification_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            used INTEGER DEFAULT 0
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            analysis_timestamp TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            interp_blinks INTEGER,
            zero_blinks INTEGER,
            interp_avg_min_norm REAL,
            zero_avg_min_norm REAL,
            analysis_duration_s REAL,
            mean_pfa_normalized REAL,
            mean_minimum_pfa_normalized REAL,
            blink_frequency_per_min REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    existing_columns = [
        row[1]
        for row in c.execute("PRAGMA table_info(analysis_results)").fetchall()
    ]

    if "user_id" not in existing_columns:
        c.execute("ALTER TABLE analysis_results ADD COLUMN user_id INTEGER")

    if "mean_pfa_normalized" not in existing_columns:
        c.execute("ALTER TABLE analysis_results ADD COLUMN mean_pfa_normalized REAL")

    if "mean_minimum_pfa_normalized" not in existing_columns:
        c.execute("ALTER TABLE analysis_results ADD COLUMN mean_minimum_pfa_normalized REAL")

    if "blink_frequency_per_min" not in existing_columns:
        c.execute("ALTER TABLE analysis_results ADD COLUMN blink_frequency_per_min REAL")

    conn.commit()
    conn.close()


def user_exists(email):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    row = c.execute(
        "SELECT id FROM users WHERE email = ?",
        (email.strip().lower(),)
    ).fetchone()
    conn.close()
    return row is not None


def save_verification_code(email, code):
    email = email.strip().lower()
    now = time.time()
    expires_at = now + 10 * 60

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute(
        "DELETE FROM verification_codes WHERE email = ? AND expires_at < ?",
        (email, now)
    )

    c.execute('''
        INSERT INTO verification_codes (email, code, created_at, expires_at, used)
        VALUES (?, ?, ?, ?, 0)
    ''', (email, code, now, expires_at))

    conn.commit()
    conn.close()


def check_verification_code(email, code):
    email = email.strip().lower()
    code = code.strip()
    now = time.time()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    row = c.execute('''
        SELECT id, expires_at, used
        FROM verification_codes
        WHERE email = ? AND code = ?
        ORDER BY id DESC
        LIMIT 1
    ''', (email, code)).fetchone()

    if not row:
        conn.close()
        return False, "验证码错误，请重新输入。"

    code_id, expires_at, used = row

    if used:
        conn.close()
        return False, "验证码已使用，请重新获取。"

    if now > expires_at:
        conn.close()
        return False, "验证码已过期，请重新获取。"

    c.execute(
        "UPDATE verification_codes SET used = 1 WHERE id = ?",
        (code_id,)
    )
    conn.commit()
    conn.close()

    return True, "验证码验证成功。"


def register_user(email, password):
    email = email.strip().lower()

    if not is_valid_email(email):
        return False, "请输入有效邮箱。"

    if len(password) < 6:
        return False, "密码至少需要 6 位。"

    if user_exists(email):
        return False, "该邮箱已注册，请直接登录。"

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        c.execute('''
            INSERT INTO users (email, password_hash, created_at)
            VALUES (?, ?, ?)
        ''', (
            email,
            hash_password(password),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        return True, "注册成功，请登录。"
    except Exception as e:
        return False, f"注册失败：{e}"
    finally:
        conn.close()


def authenticate_user(email, password):
    email = email.strip().lower()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    row = c.execute('''
        SELECT id, email, password_hash
        FROM users
        WHERE email = ?
    ''', (email,)).fetchone()

    conn.close()

    if not row:
        return False, None, "该邮箱尚未注册。"

    user_id, user_email, password_hash_value = row

    if not verify_password(password, password_hash_value):
        return False, None, "密码错误。"

    return True, {"id": user_id, "email": user_email}, "登录成功。"


def save_results_to_db(user_id, filename, stats):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO analysis_results (
            user_id,
            analysis_timestamp,
            original_filename,
            interp_blinks,
            zero_blinks,
            interp_avg_min_norm,
            zero_avg_min_norm,
            analysis_duration_s,
            mean_pfa_normalized,
            mean_minimum_pfa_normalized,
            blink_frequency_per_min
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        stats.get('interp_blinks', 0),
        stats.get('zero_blinks', 0),
        stats.get('interp_avg_min_norm'),
        stats.get('zero_avg_min_norm'),
        stats.get('analysis_duration_s'),
        stats.get('mean_pfa_normalized'),
        stats.get('mean_minimum_pfa_normalized'),
        stats.get('blink_frequency_per_min')
    ))
    conn.commit()
    conn.close()


def load_results_from_db(user_id):
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_FILE)

    try:
        return pd.read_sql_query(
            """
            SELECT *
            FROM analysis_results
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            conn,
            params=(user_id,)
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def delete_analysis_records(user_id, record_ids):
    if not record_ids:
        return False, "请选择要删除的记录。"

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    try:
        placeholders = ",".join(["?"] * len(record_ids))
        query = f"""
            DELETE FROM analysis_results
            WHERE user_id = ?
            AND id IN ({placeholders})
        """
        c.execute(query, [user_id] + list(record_ids))
        conn.commit()
        return True, f"已删除 {c.rowcount} 条记录。"
    except Exception as e:
        return False, f"删除失败：{e}"
    finally:
        conn.close()


_MODEL_LOAD_ERRORS = {}


# --- Model Loading ---
@st.cache_resource
def load_detection_model(path):
    if not os.path.exists(path):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception as e:
        _MODEL_LOAD_ERRORS["yolo"] = str(e)
        return None


@st.cache_resource
def load_segmentation_model(path):
    if not os.path.exists(path):
        return None
    try:
        import torch
        from ME_FPN_model import FPN as MyCustomFPN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyCustomFPN(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=1,
            in_channels=3
        )
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        _MODEL_LOAD_ERRORS["seg"] = str(e)
        return None


# --- Core Functions ---
def preprocess_for_segmentation(eye_crop, input_size=(256, 256)):
    import cv2
    import torch
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    import cv2

    mask_numpy = output_mask.squeeze().cpu().detach().numpy()
    probabilities = 1 / (1 + np.exp(-mask_numpy))
    probabilities_resized = cv2.resize(
        probabilities,
        (original_crop_shape[1], original_crop_shape[0])
    )
    return np.sum((probabilities_resized > seg_threshold).astype(np.uint8))


def clean_numeric_signal(data):
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
    data = clean_numeric_signal(data)

    if data.size == 0:
        return data

    if len(data) < 8:
        return data

    try:
        wavelet_obj = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(
            data_len=len(data),
            filter_len=wavelet_obj.dec_len
        )
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
    import cv2
    import torch

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

    valley_idx_interp = find_true_valleys(
        raw_sig_interp,
        peaks_interp,
        valley_search_frames
    )

    # ==========================================================
    # Method 2: Zero method
    # Still calculated, but no Method 2 graph is displayed.
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

    valley_idx_zero = find_true_valleys(
        raw_sig_zero,
        peaks_zero,
        valley_search_frames
    )

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

    duration = float(df['timestamp'].max())

    mean_pfa_normalized = float(interp_raw_norm_area)
    mean_minimum_pfa_normalized = float(interp_avg_min_norm)
    blink_frequency_per_min = (len(peaks_interp) / duration * 60) if duration > 0 else 0.0

    # ==========================================================
    # Visualization
    # Only Method 1 is displayed on the website.
    # ==========================================================
    raw_sig_interp_norm, interp_fixed_baseline = normalize_by_initial_baseline(
        raw_sig_interp,
        fps,
        config['BASELINE_WINDOW_SEC']
    )

    denoised_sig_interp_norm = denoised_sig_interp / interp_fixed_baseline
    roll_max_interp_norm = roll_max_interp / interp_fixed_baseline

    fig, ax1 = plt.subplots(
        1,
        1,
        figsize=(14, 6)
    )

    ax1.plot(
        df['timestamp'],
        raw_sig_interp_norm,
        label='Raw normalized',
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
            label='Blink valley'
        )

    ax1.set_title(f'Palpebral Fissure Area Signal | Blinks: {len(peaks_interp)}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Palpebral Fissure Area')
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

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

        "mean_pfa_normalized": round(mean_pfa_normalized, 4),
        "mean_minimum_pfa_normalized": round(mean_minimum_pfa_normalized, 4),
        "blink_frequency_per_min": round(blink_frequency_per_min, 2),
    }

    return None, df, fig, stats
def show_auth_page():
    if _BCRYPT_ERROR is not None:
        st.error("Password library failed to import. Check that `bcrypt` is in requirements.txt.")
        st.exception(_BCRYPT_ERROR)
        return

    init_db()

    st.markdown(
        '<h1 class="main-header">👁 Mouse Blink Analysis</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Please log in or register to use the analysis platform.</p>',
        unsafe_allow_html=True
    )

    if "auth_email_for_register" not in st.session_state:
        st.session_state.auth_email_for_register = ""
    if "verification_sent" not in st.session_state:
        st.session_state.verification_sent = False

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.markdown("### Login")

        login_email = st.text_input(
            "Email",
            key="login_email"
        )
        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):
            if not login_email or not login_password:
                st.warning("Please enter both email and password.")
            else:
                success, user, message = authenticate_user(login_email, login_password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.analysis_results = None
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    with register_tab:
        st.markdown("### Register")

        register_email = st.text_input(
            "Email",
            key="register_email"
        )

        col_send, col_info = st.columns([1, 2])

        with col_send:
            if st.button("Send Verification Code", use_container_width=True):
                normalized_email = register_email.strip().lower()

                if not normalized_email:
                    st.warning("Please enter your email first.")
                elif not is_valid_email(normalized_email):
                    st.warning("Please enter a valid email address.")
                elif user_exists(normalized_email):
                    st.warning("This email is already registered. Please log in.")
                else:
                    code = generate_verification_code()
                    saved_ok = False

                    try:
                        save_verification_code(normalized_email, code)
                        saved_ok = True
                    except Exception as e:
                        st.error(f"Could not save verification code: {e}")

                    if saved_ok:
                        sent_ok, send_message = send_verification_email(
                            normalized_email,
                            code
                        )

                        if sent_ok:
                            st.session_state.auth_email_for_register = normalized_email
                            st.session_state.verification_sent = True
                            st.success(send_message)
                        else:
                            st.error(send_message)

        with col_info:
            st.caption("The verification code is valid for 10 minutes.")

        verification_code = st.text_input(
            "Verification Code",
            key="register_code"
        )
        register_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )
        register_password_confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="register_password_confirm"
        )

        if st.button("Create Account", use_container_width=True):
            normalized_email = register_email.strip().lower()

            if not normalized_email:
                st.warning("Please enter your email.")
            elif not is_valid_email(normalized_email):
                st.warning("Please enter a valid email address.")
            elif not verification_code:
                st.warning("Please enter the verification code.")
            elif not register_password or not register_password_confirm:
                st.warning("Please enter your password twice.")
            elif register_password != register_password_confirm:
                st.warning("The two passwords do not match.")
            elif len(register_password) < 6:
                st.warning("Password must be at least 6 characters.")
            else:
                code_ok, code_message = check_verification_code(
                    normalized_email,
                    verification_code
                )

                if not code_ok:
                    st.error(code_message)
                else:
                    success, message = register_user(
                        normalized_email,
                        register_password
                    )

                    if success:
                        st.success(message)
                        st.session_state.verification_sent = False
                        st.session_state.auth_email_for_register = ""
                    else:
                        st.error(message)


# --- Main App ---
def show_main_app():
    if _IMPORT_ERROR is not None:
        st.error(
            "App dependencies failed to import. On Streamlit Cloud this is usually "
            "a missing package."
        )
        st.exception(_IMPORT_ERROR)
        return

    init_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None

    if not st.session_state.authenticated or st.session_state.user is None:
        show_auth_page()
        return

    current_user = st.session_state.user
    current_user_id = current_user["id"]

    st.markdown(
        '<h1 class="main-header">👁 Mouse Blink Analysis</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Advanced signal processing for normalized palpebral fissure area and blink frequency analysis.</p>',
        unsafe_allow_html=True
    )

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Account")
        st.caption(f"Logged in as: {current_user['email']}")

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.analysis_results = None
            st.rerun()

        st.markdown("---")
        st.markdown("### Upload Video")
        uploaded_file = st.file_uploader(
            "Select a video file",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("_active_upload") != file_id:
                st.session_state._active_upload = file_id
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
                    extra = _MODEL_LOAD_ERRORS.get("yolo")
                    st.error(f"YOLO model not found or failed to load: {YOLO_MODEL_FILENAME}")
                    if extra:
                        st.caption(extra)
                    return

                if seg_model is None:
                    extra = _MODEL_LOAD_ERRORS.get("seg")
                    st.error(f"Segmentation model not found or failed to load: {SEG_MODEL_FILENAME}")
                    if extra:
                        st.caption(extra)
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
                        save_results_to_db(
                            current_user_id,
                            uploaded_file.name,
                            stats
                        )
                        st.success("Analysis complete and saved to your history.")
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

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Mean PFA (Normalized)",
                f"{stats.get('mean_pfa_normalized', 0):.4f}",
                help="Mean palpebral fissure area normalized by the local baseline."
            )

        with col2:
            st.metric(
                "Mean Minimum PFA (Normalized)",
                f"{stats.get('mean_minimum_pfa_normalized', 0):.4f}",
                help="Mean minimum palpebral fissure area at detected blink valleys, normalized by the local baseline."
            )

        with col3:
            st.metric(
                "Blink Frequency",
                f"{stats.get('blink_frequency_per_min', 0):.2f} /min",
                help="Blink frequency calculated as blinks per minute."
            )

        st.markdown("---")
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
    history_df = load_results_from_db(current_user_id)

    if not history_df.empty:
        display_df = history_df.copy()

        if "mean_pfa_normalized" not in display_df.columns:
            display_df["mean_pfa_normalized"] = np.nan

        if "mean_minimum_pfa_normalized" not in display_df.columns:
            display_df["mean_minimum_pfa_normalized"] = display_df.get(
                "interp_avg_min_norm",
                np.nan
            )

        if "blink_frequency_per_min" not in display_df.columns:
            display_df["blink_frequency_per_min"] = np.where(
                display_df["analysis_duration_s"] > 0,
                display_df["interp_blinks"] / display_df["analysis_duration_s"] * 60,
                0
            )

        display_df = display_df[[
            "id",
            "analysis_timestamp",
            "original_filename",
            "mean_pfa_normalized",
            "mean_minimum_pfa_normalized",
            "blink_frequency_per_min"
        ]]

        display_df.columns = [
            "ID",
            "Timestamp",
            "Filename",
            "Mean PFA (Normalized)",
            "Mean Minimum PFA (Normalized)",
            "Blink Frequency (Blinks/min)"
        ]

        st.dataframe(display_df, use_container_width=True)

        st.markdown("#### Delete Records")

        record_options = display_df["ID"].tolist()

        selected_record_ids = st.multiselect(
            "Select records to delete",
            options=record_options,
            help="Only your own records are shown and can be deleted."
        )

        delete_col1, delete_col2 = st.columns([1, 3])

        with delete_col1:
            delete_button = st.button(
                "Delete Selected",
                disabled=(len(selected_record_ids) == 0),
                use_container_width=True
            )

        with delete_col2:
            if selected_record_ids:
                st.caption(f"{len(selected_record_ids)} record(s) selected.")

        if delete_button:
            success, message = delete_analysis_records(
                current_user_id,
                selected_record_ids
            )

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

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
        "Mouse Blink Analysis Platform · Mean PFA · Mean Minimum PFA · Blink Frequency"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    show_main_app()
