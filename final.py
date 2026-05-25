import hashlib
import pickle
import re
import random
import sqlite3
import os
import zipfile
import io
import smtplib
import ssl
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import requests
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# ============================================================
#  GMAIL SMTP CONFIG
# ============================================================
def _load_secret(key: str, default=None):
    try:
        val = st.secrets.get(key)
        if val is not None: return val
        val = st.secrets.get(key.lower())
        if val is not None: return val
        val = st.secrets.get(key.upper())
        if val is not None: return val
    except Exception:
        pass
    val = os.environ.get(key)
    if val is not None: return val
    val = os.environ.get(key.upper())
    if val is not None: return val
    return os.environ.get(key.lower(), default)

SMTP_SENDER_EMAIL  = _load_secret("smtp_sender_email")
SMTP_APP_PASSWORD  = _load_secret("smtp_app_password")
SMTP_SENDER_NAME   = _load_secret("smtp_sender_name")
OTP_EXPIRY_MINUTES = int(_load_secret("otp_expiry_minutes", 10))

# ============================================================
#  SECTION 0 — PATHS & DATABASE
# ============================================================
BASE_DIR        = Path(__file__).resolve().parent
DB_FILE         = BASE_DIR / "movie_app.db"
SIMILARITY_FILE = BASE_DIR / "similarity.pkl"
MOVIE_DICT_FILE = BASE_DIR / "movies.pkl"
MOVIELENS_DIR   = BASE_DIR / "movielens"
RATINGS_CSV     = MOVIELENS_DIR / "ratings.csv"
ML_MOVIES_CSV   = MOVIELENS_DIR / "movies.csv"

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                dob           TEXT NOT NULL,
                is_adult      INTEGER NOT NULL DEFAULT 0,
                verified      INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                title    TEXT NOT NULL,
                UNIQUE(username, title)
            );
            CREATE TABLE IF NOT EXISTS movies (
                id    INTEGER PRIMARY KEY,
                title TEXT NOT NULL UNIQUE,
                tags  TEXT
            );
            CREATE TABLE IF NOT EXISTS user_ratings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                movie_id   INTEGER NOT NULL,
                rating     REAL NOT NULL DEFAULT 1.0,
                UNIQUE(username, movie_id)
            );
            CREATE TABLE IF NOT EXISTS otp_store (
                email      TEXT PRIMARY KEY,
                otp        TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                purpose    TEXT NOT NULL DEFAULT 'verify'
            );
        """)
        count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        if count == 0 and MOVIE_DICT_FILE.exists():
            try:
                raw = pickle.load(open(MOVIE_DICT_FILE, "rb"))
                df  = pd.DataFrame(raw)
                rows = []
                for i, row in df.iterrows():
                    movie_id = int(row["movie_id"]) if "movie_id" in df.columns else i
                    title    = str(row["title"])
                    tags     = str(row.get("tags", "")) if "tags" in df.columns else ""
                    rows.append((movie_id, title, tags))
                conn.executemany("INSERT OR IGNORE INTO movies (id, title, tags) VALUES (?,?,?)", rows)
            except Exception as e:
                st.warning(f"Could not seed movies table: {e}")

init_db()

# ============================================================
#  SECTION 1 — OTP EMAIL SYSTEM
# ============================================================

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def save_otp(email_addr: str, otp: str, purpose: str = "verify"):
    expires = (datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO otp_store (email, otp, expires_at, purpose) VALUES (?,?,?,?)",
            (email_addr.lower().strip(), otp, expires, purpose)
        )

def verify_otp(email_addr: str, otp_input: str) -> Tuple[bool, str]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT otp, expires_at FROM otp_store WHERE email=?",
            (email_addr.lower().strip(),)
        ).fetchone()
    if not row:
        return False, "OTP not found. Please request a new one."
    if datetime.now() > datetime.fromisoformat(row["expires_at"]):
        return False, f"OTP expired ({OTP_EXPIRY_MINUTES} min). Please request a new one."
    if row["otp"] != otp_input.strip():
        return False, "Incorrect OTP. Please try again."
    with get_conn() as conn:
        conn.execute("DELETE FROM otp_store WHERE email=?", (email_addr.lower().strip(),))
    return True, "OTP verified!"

def send_otp_email(to_email: str, otp: str, purpose: str = "verify") -> Tuple[bool, str]:
    subject_map = {
        "verify":   "CineMatch — Email Verification OTP",
        "login":    "CineMatch — Login OTP",
        "register": "CineMatch — Registration OTP",
    }
    subject = subject_map.get(purpose, "CineMatch — OTP Code")
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="margin:0;padding:0;background:#0f0c29;font-family:'Segoe UI',sans-serif;">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f0c29;padding:40px 0;">
        <tr><td align="center">
          <table width="480" cellpadding="0" cellspacing="0"
            style="background:linear-gradient(160deg,#1a0a2e,#16213e);
                   border-radius:16px;border:1px solid rgba(139,92,246,0.3);
                   padding:40px;box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <tr><td align="center" style="padding-bottom:24px;">
              <div style="font-size:36px;">🎬</div>
              <h1 style="color:#c4b5fd;font-size:24px;margin:8px 0;">CineMatch</h1>
              <p style="color:rgba(167,139,250,0.7);font-size:13px;margin:0;">Movie Recommendation System</p>
            </td></tr>
            <tr><td style="padding:24px 0;border-top:1px solid rgba(139,92,246,0.2);
                           border-bottom:1px solid rgba(139,92,246,0.2);">
              <p style="color:rgba(210,200,255,0.85);font-size:15px;margin:0 0 20px;">
                Your verification code is below. It expires in
                <strong style="color:#f9a8d4;">{OTP_EXPIRY_MINUTES} minutes</strong>.
              </p>
              <div style="background:rgba(124,58,237,0.2);border:1.5px solid rgba(139,92,246,0.5);
                          border-radius:12px;padding:24px;text-align:center;">
                <div style="font-size:42px;font-weight:800;letter-spacing:12px;
                             color:#c4b5fd;font-family:monospace;">{otp}</div>
              </div>
              <p style="color:rgba(167,139,250,0.6);font-size:12px;margin:16px 0 0;text-align:center;">
                If you did not request this, please ignore this email.
              </p>
            </td></tr>
            <tr><td style="padding-top:20px;">
              <p style="color:rgba(150,130,200,0.5);font-size:11px;text-align:center;margin:0;">
                &copy; 2025 CineMatch &middot; Automated email, please do not reply.
              </p>
            </td></tr>
          </table>
        </td></tr>
      </table>
    </body>
    </html>
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{SMTP_SENDER_NAME} <{SMTP_SENDER_EMAIL}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(SMTP_SENDER_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_SENDER_EMAIL, to_email, msg.as_string())
        return True, "OTP sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail authentication failed. Please check your App Password."
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# ============================================================
#  SECTION 2 — USER HELPERS
# ============================================================

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def calculate_age(dob_str: str) -> int:
    dob   = datetime.strptime(dob_str, "%Y-%m-%d").date()
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def email_exists(email_addr: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email=?",
                           (email_addr.lower().strip(),)).fetchone()
    return row is not None

def get_user_by_email(email_addr: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?",
                           (email_addr.lower().strip(),)).fetchone()
    return dict(row) if row else None

def register_user_full(username: str, email_addr: str, password: str, dob: str) -> Tuple[bool, str]:
    age      = calculate_age(dob)
    is_adult = 1 if age >= 18 else 0
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username,email,password_hash,dob,is_adult,verified) VALUES (?,?,?,?,?,1)",
                (username.strip(), email_addr.lower().strip(), _hash(password), dob, is_adult)
            )
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "This username is already taken."
        return False, "This email is already registered."
    except Exception as e:
        return False, str(e)

def authenticate_user(email_addr: str, password: str) -> Tuple[bool, str, Optional[dict]]:
    user = get_user_by_email(email_addr)
    if not user:
        return False, "Email not registered.", None
    if user["password_hash"] != _hash(password):
        return False, "Incorrect password.", None
    return True, "Login successful!", user

def get_user_age(username: str) -> Optional[int]:
    with get_conn() as conn:
        row = conn.execute("SELECT dob FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        return None
    return calculate_age(row["dob"])

def is_user_adult(username: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT is_adult FROM users WHERE username=?", (username,)).fetchone()
    return bool(row["is_adult"]) if row else False

def get_watchlist(username):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT title FROM watchlist WHERE username=? ORDER BY id", (username,)
        ).fetchall()
    return [r["title"] for r in rows]

def add_to_watchlist(username, title):
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO watchlist (username,title) VALUES (?,?)", (username, title))
            row = conn.execute("SELECT id FROM movies WHERE title=?", (title,)).fetchone()
            if row:
                conn.execute(
                    "INSERT OR IGNORE INTO user_ratings (username,movie_id,rating) VALUES (?,?,?)",
                    (username, row["id"], 1.0))
        return True
    except sqlite3.IntegrityError:
        return False

def remove_from_watchlist(username, title):
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM watchlist WHERE username=? AND title=?", (username, title)
        )
        mrow = conn.execute("SELECT id FROM movies WHERE title=?", (title,)).fetchone()
        if mrow:
            conn.execute("DELETE FROM user_ratings WHERE username=? AND movie_id=?",
                         (username, mrow["id"]))
    return cur.rowcount > 0

def cb_add(username, title):
    ok = add_to_watchlist(username, title)
    st.session_state.wl_flash = ("added", title, ok)

def cb_remove(username, title):
    ok = remove_from_watchlist(username, title)
    st.session_state.wl_flash = ("removed", title, ok)

RATING_AGE_MAP = {
    "G":0,"PG":10,"PG-13":13,"R":17,"NC-17":18,
    "UNRATED":0,"NOT RATED":0,"NR":0
}

def min_age_for_rating(label):
    if not label: return 0
    lab = str(label).upper().strip()
    if lab.startswith("TV-"):
        try: return int(lab.split("-")[1])
        except: return 14
    return RATING_AGE_MAP.get(lab, 0)

# ============================================================
#  SECTION 3 — MOVIELENS DATA
# ============================================================
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_movielens():
    if RATINGS_CSV.exists() and ML_MOVIES_CSV.exists():
        return True
    try:
        MOVIELENS_DIR.mkdir(exist_ok=True)
        with st.spinner("Downloading MovieLens dataset (first time only)..."):
            r = requests.get(MOVIELENS_URL, timeout=60)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for name in z.namelist():
                if name.endswith("ratings.csv"):
                    RATINGS_CSV.write_bytes(z.open(name).read())
                elif name.endswith("/movies.csv"):
                    ML_MOVIES_CSV.write_bytes(z.open(name).read())
        return True
    except Exception as e:
        st.error(f"MovieLens download failed: {e}")
        return False

@st.cache_data(show_spinner=False)
def load_movielens():
    if not RATINGS_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(RATINGS_CSV), pd.read_csv(ML_MOVIES_CSV)

# ============================================================
#  SECTION 4 — NLP MOOD DETECTOR
# ============================================================
MOOD_GENRE_MAP = {
    "sad":"Drama","happy":"Comedy","romantic":"Romance","thriller":"Thriller",
    "action":"Action","scared":"Horror","inspired":"Biography","adventure":"Adventure",
    "mystery":"Mystery","animated":"Animation","family":"Family","sci-fi":"Sci-Fi",
    "bored":"Comedy","angry":"Action","nostalgic":"Drama","anxious":"Thriller",
}
KEYWORD_MOOD = {
    "sad":"sad","sadness":"sad","sorrow":"sad","depress":"sad","depressed":"sad",
    "unhappy":"sad","miserable":"sad","gloomy":"sad","grief":"sad","heartbreak":"sad",
    "cry":"sad","lonely":"sad","hopeless":"sad","devastated":"sad","melancholy":"sad",
    "happy":"happy","happiness":"happy","joy":"happy","joyful":"happy","excited":"happy",
    "cheerful":"happy","great":"happy","fantastic":"happy","wonderful":"happy","fun":"happy",
    "romantic":"romantic","romance":"romantic","love":"romantic","crush":"romantic",
    "affection":"romantic","passion":"romantic",
    "thriller":"thriller","suspense":"thriller","tension":"thriller",
    "anxious":"anxious","anxiety":"anxious","nervous":"anxious","stressed":"anxious",
    "action":"action","fight":"action","adventure":"adventure","adrenaline":"action",
    "angry":"angry","anger":"angry","rage":"angry","furious":"angry",
    "scared":"scared","fear":"scared","horror":"scared","terrified":"scared","spooky":"scared",
    "inspired":"inspired","motivated":"inspired","hopeful":"inspired","determined":"inspired",
    "bored":"bored","boredom":"bored","idle":"bored","restless":"bored",
    "nostalgic":"nostalgic","nostalgia":"nostalgic","childhood":"nostalgic","memories":"nostalgic",
    "sci-fi":"sci-fi","space":"sci-fi","futuristic":"sci-fi","robot":"sci-fi","alien":"sci-fi",
    "animated":"animated","cartoon":"animated","family":"family","kids":"family",
    "mystery":"mystery","detective":"mystery","whodunit":"mystery",
}
NEGATION_WORDS = {"not","no","never","n't","dont","don't","cannot","can't","isn't","hardly"}
MOOD_OPPOSITE  = {"happy":"sad","sad":"happy","romantic":"sad","action":"bored","scared":"happy"}
MOOD_EMOJI     = {
    "sad":"😢 Sad","happy":"😊 Happy","romantic":"❤️ Romantic","thriller":"😰 Thriller",
    "action":"💥 Action","scared":"😱 Horror","inspired":"🌟 Inspired",
    "adventure":"🗺️ Adventure","mystery":"🔍 Mystery","animated":"🎨 Animated",
    "family":"👨‍👩‍👧 Family","sci-fi":"🚀 Sci-Fi","bored":"😑 Bored",
    "angry":"😤 Angry","nostalgic":"🕰️ Nostalgic","anxious":"😬 Anxious",
}
_SUFFIX_RULES = [
    ("ness",""),("ing",""),("tion",""),("ed",""),
    ("ly",""),("ies","y"),("er",""),("est","")
]

def _lemmatize(word):
    for s, r in _SUFFIX_RULES:
        if word.endswith(s) and len(word)-len(s) >= 3:
            return word[:len(word)-len(s)] + r
    return word

def _tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s'/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()

def detect_mood(user_text):
    if not user_text or not user_text.strip():
        return "happy", "Comedy", 0.0
    text_lower = user_text.lower().strip()
    tokens     = _tokenize(text_lower)
    scores     = {}
    for phrase, mood in KEYWORD_MOOD.items():
        if " " in phrase and phrase in text_lower:
            scores[mood] = scores.get(mood, 0) + 1.5
    for i, token in enumerate(tokens):
        window  = tokens[max(0,i-3):i]
        negated = any(w in NEGATION_WORDS for w in window)
        for candidate in [token, _lemmatize(token)]:
            if candidate in KEYWORD_MOOD:
                mood   = KEYWORD_MOOD[candidate]
                weight = 1.0
                if negated:
                    flipped = MOOD_OPPOSITE.get(mood)
                    if flipped:
                        scores[flipped] = scores.get(flipped, 0) + weight * 0.8
                    weight = -0.3
                scores[mood] = scores.get(mood, 0) + weight
                break
    if not scores:
        return "happy", "Comedy", 0.3
    best_mood  = max(scores, key=lambda m: scores[m])
    best_score = scores[best_mood]
    total      = sum(abs(v) for v in scores.values()) or 1
    confidence = min(round(best_score/total, 2), 1.0)
    if confidence < 0:
        confidence = 0.3
        best_mood  = "happy"
    return best_mood, MOOD_GENRE_MAP.get(best_mood, "Drama"), confidence

def mood_label(mood): return MOOD_EMOJI.get(mood, f"🎬 {mood.capitalize()}")

# ============================================================
#  SECTION 5 — PAGE CONFIG & GLOBAL CSS
# ============================================================
_sidebar_state = "expanded" if st.session_state.get("auth_stage") == "app" else "collapsed"
st.set_page_config(
    page_title="🎬 CineMatch — Movie Recommendation System",
    layout="wide",
    initial_sidebar_state=_sidebar_state
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stApp"] {
    background: #07060f !important;
    font-family: 'Outfit', sans-serif !important;
    color: rgba(220,210,255,0.9) !important;
}

[data-testid="stHeader"]  { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer    { display: none !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg,#1a0a2e 0%,#16213e 60%,#0f3460 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.2) !important;
}
[data-testid="stSidebar"] * { color: rgba(220,210,255,0.92) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #c4b5fd !important;
    border-bottom: 1px solid rgba(139,92,246,0.2);
    padding-bottom: 6px;
}
[data-testid="stSidebar"] button {
    background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
    border: none !important; border-radius: 8px !important;
    color: #fff !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(139,92,246,0.2) !important; }

.auth-card {
    background: rgba(25, 20, 50, 0.45) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 20px !important;
    padding: 40px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    margin-top: 20px;
    margin-bottom: 20px;
}
.auth-logo { text-align: center; margin-bottom: 28px; }
.auth-logo .icon {
    font-size: 52px; line-height: 1; margin-bottom: 10px;
    display: block; filter: drop-shadow(0 0 20px rgba(196,181,253,0.4));
}
.auth-logo h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 28px !important; font-weight: 900 !important;
    background: linear-gradient(135deg,#c4b5fd 0%,#f9a8d4 50%,#93c5fd 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important; letter-spacing: -0.5px;
}
.auth-logo p {
    color: rgba(167,139,250,0.6) !important;
    font-size: 13px !important; margin: 6px 0 0 !important;
}
.auth-step-label {
    font-size: 11px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(139,92,246,0.7); margin-bottom: 6px;
}
.auth-title {
    font-size: 20px; font-weight: 700;
    color: #e2d9ff !important; margin: 0 0 4px !important;
}
.auth-subtitle {
    font-size: 13px; color: rgba(167,139,250,0.6) !important;
    margin: 0 0 24px !important;
}
.otp-hint {
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 10px; padding: 12px 16px;
    font-size: 13px; color: rgba(196,181,253,0.85) !important;
    margin-bottom: 16px; display: flex; align-items: center; gap: 10px;
}
.age-block {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px; padding: 20px;
    text-align: center; margin-top: 16px;
}
.age-block h3 { color: #fca5a5 !important; font-size: 18px !important; margin: 0 0 8px !important; }
.age-block p  { color: rgba(252,165,165,0.75) !important; font-size: 13px !important; margin: 0 !important; }
.adult-block {
    background: rgba(20,184,166,0.1);
    border: 1px solid rgba(20,184,166,0.3);
    border-radius: 12px; padding: 16px;
    text-align: center; margin-top: 10px;
}
.adult-block p { color: #5eead4 !important; font-size: 13px !important; margin: 0 !important; }

.step-dots { display: flex; justify-content: center; gap: 8px; margin-bottom: 28px; }
.step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(139,92,246,0.2); transition: all 0.3s;
}
.step-dot.active {
    background: #7c3aed; box-shadow: 0 0 8px rgba(124,58,237,0.6);
    width: 24px; border-radius: 4px;
}
.step-dot.done { background: rgba(139,92,246,0.5); }

[data-testid="stMainBlockContainer"] input[type="text"],
[data-testid="stMainBlockContainer"] input[type="email"],
[data-testid="stMainBlockContainer"] input[type="password"],
[data-testid="stMainBlockContainer"] textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 10px !important;
    color: rgba(220,210,255,0.95) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important;
}
[data-testid="stMainBlockContainer"] input:focus {
    border-color: rgba(167,139,250,0.65) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}
[data-testid="stMainBlockContainer"] input::placeholder {
    color: rgba(139,92,246,0.35) !important;
}
[data-testid="stMainBlockContainer"] label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: rgba(196,181,253,0.8) !important;
}

[data-testid="stMainBlockContainer"] button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.3px;
    border-radius: 10px !important; border: none !important;
    background: linear-gradient(135deg,#7c3aed,#4338ca) !important;
    color: #fff !important; transition: all 0.2s !important;
}
[data-testid="stMainBlockContainer"] button:hover {
    opacity: 0.88 !important; transform: translateY(-1px) !important;
}

[data-testid="stMainBlockContainer"] h1 {
    background: linear-gradient(90deg,#c4b5fd,#f9a8d4,#93c5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-size: 2rem !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 900 !important;
}
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 { color: #c4b5fd !important; }
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span { color: rgba(210,200,255,0.88) !important; }

[data-testid="stSelectbox"]>div>div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: rgba(220,210,255,0.95) !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-radius: 10px !important;
}
hr { border-color: rgba(139,92,246,0.2) !important; }

.hybrid-badge {
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(20,184,166,0.15);border:1px solid rgba(20,184,166,0.4);
    border-radius:20px;padding:6px 14px;font-size:13px;font-weight:600;
    color:#5eead4;margin-bottom:10px;
}
.content-badge {
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.4);
    border-radius:20px;padding:6px 14px;font-size:13px;font-weight:600;
    color:#c4b5fd;margin-bottom:10px;
}
.mood-badge {
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(124,58,237,0.18);border:1px solid rgba(139,92,246,0.4);
    border-radius:20px;padding:8px 18px;font-size:15px;font-weight:600;
    color:#c4b5fd;margin-bottom:14px;
}

.movie-card {
    border-radius:14px;padding:14px;background:rgba(255,255,255,0.04);
    display:flex;gap:14px;align-items:flex-start;margin-bottom:16px;
    color:rgba(220,210,255,0.92);border:1px solid rgba(139,92,246,0.2);
}
.movie-card-img {
    width:130px;height:190px;object-fit:cover;border-radius:10px;
    flex-shrink:0;border:1px solid rgba(139,92,246,0.15);
}
.movie-card-body { display:flex;flex-direction:column;gap:6px; }
.movie-title     { font-size:18px;font-weight:700;margin:0;color:#c4b5fd; }
.meta-row        { font-size:13px;color:rgba(167,139,250,0.65);display:flex;gap:10px;flex-wrap:wrap; }
.overview        { font-size:14px;color:rgba(210,200,255,0.82);line-height:1.5;margin-top:6px; }

.conf-wrap {
    background:rgba(255,255,255,0.08);border-radius:8px;height:6px;
    width:120px;overflow:hidden;display:inline-block;vertical-align:middle;
}
.conf-bar { height:100%;border-radius:8px;background:linear-gradient(90deg,#7c3aed,#a78bfa); }

.restricted-banner {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 10px; padding: 12px 16px;
    color: rgba(252,165,165,0.85) !important;
    font-size: 13px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

def get_base64_image(image_path: Path) -> str:
    import base64
    if image_path.exists():
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            pass
    return ""

def inject_theme_css(auth_mode=True):
    if auth_mode:
        collage_path = BASE_DIR / "movie_collage.png"
        b64_str = get_base64_image(collage_path)
        if b64_str:
            bg_css = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: linear-gradient(135deg, rgba(7,6,15,0.92) 0%, rgba(20,10,45,0.78) 50%, rgba(7,6,15,0.95) 100%), url("data:image/png;base64,{b64_str}") !important;
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}
            div[data-testid="column"]:nth-of-type(2) {{
                background: rgba(15,10,30,0.65) !important;
                backdrop-filter: blur(20px) !important;
                -webkit-backdrop-filter: blur(20px) !important;
                border: 1px solid rgba(139,92,246,0.25) !important;
                border-radius: 24px !important;
                padding: 40px !important;
                box-shadow: 0 15px 45px rgba(0,0,0,0.5), 0 0 20px rgba(139,92,246,0.15) !important;
            }}
            </style>
            """
        else:
            bg_css = """
            <style>
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #07060f 0%, #150b28 50%, #07060f 100%) !important;
            }
            div[data-testid="column"]:nth-of-type(2) {
                background: rgba(15,10,30,0.65) !important;
                backdrop-filter: blur(20px) !important;
                -webkit-backdrop-filter: blur(20px) !important;
                border: 1px solid rgba(139,92,246,0.25) !important;
                border-radius: 24px !important;
                padding: 40px !important;
                box-shadow: 0 15px 45px rgba(0,0,0,0.5), 0 0 20px rgba(139,92,246,0.15) !important;
            }
            </style>
            """
    else:
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #07060f !important;
        }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)

# ============================================================
#  SECTION 6 — SESSION STATE INIT
# ============================================================
_defaults = {
    "auth_stage":     "email",
    "auth_email":     "",
    "auth_is_new":    False,
    "user":           None,
    "user_data":      None,
    "wl_flash":       None,
    "last_recs":      ([], []),
    "last_mood_info": None,
    "mood_titles":    [],
    "selected_movie": None,
    "rec_mode":       "content",
    "hybrid_alpha":   0.5,
    "otp_sent":       False,
    "otp_send_error": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
#  SECTION 7 — OMDB FETCH
# ============================================================
OMDB_API_KEY       = _load_secret("omdb_api_key", "9cfbd39b")
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Image"
ERROR_POSTER       = "https://via.placeholder.com/500x750?text=Error"

@st.cache_data(ttl=86400)
def fetch_movie_data(title):
    try:
        url = (f"http://www.omdbapi.com/?t={requests.utils.requote_uri(title)}"
               f"&apikey={OMDB_API_KEY}&plot=short")
        d = requests.get(url, timeout=6).json()
        if d.get("Response") == "True":
            poster = d.get("Poster","") if d.get("Poster","") not in ("","N/A") else PLACEHOLDER_POSTER
            return {
                "poster":   poster,
                "rating":   d.get("imdbRating","N/A"),
                "released": d.get("Released","N/A"),
                "runtime":  d.get("Runtime","N/A"),
                "genre":    d.get("Genre","N/A"),
                "plot":     d.get("Plot","N/A"),
                "rated":    d.get("Rated","N/A"),
            }
    except Exception:
        pass
    return {
        "poster":   ERROR_POSTER,
        "rating":   "N/A",
        "released": "N/A",
        "runtime":  "N/A",
        "genre":    "N/A",
        "plot":     "Overview not available.",
        "rated":    "N/A",
    }

# ============================================================
#  SECTION 8 — RECOMMENDER (Content + Collaborative + Hybrid)
# ============================================================
@st.cache_data()
def load_recommender():
    if not MOVIE_DICT_FILE.exists():
        return pd.DataFrame({"title":[]}), []
    raw = pickle.load(open(MOVIE_DICT_FILE,"rb"))
    df  = pd.DataFrame(raw)
    sim = pickle.load(open(SIMILARITY_FILE,"rb")) if SIMILARITY_FILE.exists() else []
    return df, sim

movies, similarity = load_recommender()

def get_content_scores(movie_title):
    if movies.empty or not len(similarity):
        return {}
    try:
        idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return {}
    raw  = {movies.iloc[i].title: float(s) for i, s in enumerate(similarity[idx]) if i != idx}
    maxs = max(raw.values(), default=1) or 1
    return {t: s/maxs for t, s in raw.items()}

# FIX: Removed @st.cache_data from build_svd_model because numpy arrays + complex
# objects don't always serialize cleanly with Streamlit's cache. Using st.cache_resource instead.
@st.cache_resource(show_spinner=False)
def build_svd_model():
    if not RATINGS_CSV.exists() or not ML_MOVIES_CSV.exists():
        return None
    try:
        ratings   = pd.read_csv(RATINGS_CSV)
        ml_movies = pd.read_csv(ML_MOVIES_CSV)
        popular   = ratings.groupby("movieId").size()
        popular   = popular[popular >= 10].index
        ratings   = ratings[ratings["movieId"].isin(popular)]
        user_ids  = ratings["userId"].unique().tolist()
        movie_ids = ratings["movieId"].unique().tolist()
        u2i = {u: i for i, u in enumerate(user_ids)}
        m2i = {m: i for i, m in enumerate(movie_ids)}
        rows_i = ratings["userId"].map(u2i).values
        cols_i = ratings["movieId"].map(m2i).values
        data_v = ratings["rating"].values.astype(float)
        mat    = csr_matrix((data_v,(rows_i,cols_i)), shape=(len(user_ids),len(movie_ids)))
        mat_d  = mat.toarray()
        umean  = np.true_divide(mat_d.sum(1),(mat_d!=0).sum(1).clip(min=1))
        mat_c  = mat_d.copy()
        mask   = mat_d != 0
        mat_c[mask] -= umean.reshape(-1, 1).repeat(mat_d.shape[1], 1)[mask]
        k = min(50, len(user_ids)-1, len(movie_ids)-1)
        U, sigma, Vt = svds(mat_c, k=k)
        mid2title = {}
        for _, row in ml_movies.iterrows():
            clean = re.sub(r"\s*\(\d{4}\)\s*$","",str(row["title"])).strip()
            mid2title[int(row["movieId"])] = clean
        return U, sigma, Vt, user_ids, movie_ids, mid2title, m2i, umean
    except Exception as e:
        st.warning(f"SVD model build failed: {e}")
        return None

def get_item_based_collab_scores(movie_title, n=100):
    # FIX: Removed @st.cache_data — depends on cache_resource object which can't be hashed
    model = build_svd_model()
    if model is None:
        return {}
    try:
        U, sigma, Vt, user_ids, movie_ids, mid2title, m2i, umean = model
        ml_movies  = pd.read_csv(ML_MOVIES_CSV)
        query      = re.sub(r"\s*\(\d{4}\)\s*$","",movie_title.lower().strip()).strip()
        target_mid = None
        for _, row in ml_movies.iterrows():
            ml_clean = re.sub(r"\s*\(\d{4}\)\s*$","",str(row["title"])).strip().lower()
            if query == ml_clean:
                target_mid = int(row["movieId"]); break
        if target_mid is None or target_mid not in m2i:
            return {}
        item_matrix = Vt.T * sigma
        target_idx  = m2i[target_mid]
        target_vec  = item_matrix[target_idx]
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return {}
        norms = np.linalg.norm(item_matrix, axis=1).clip(min=1e-10)
        sims  = item_matrix.dot(target_vec) / (norms * target_norm)
        top_indices = np.argsort(sims)[::-1]
        scores = {}
        for idx in top_indices:
            if idx == target_idx:
                continue
            mid   = movie_ids[idx]
            title = mid2title.get(mid)
            if title:
                scores[title] = float(sims[idx])
            if len(scores) >= n:
                break
        if scores:
            mn, mx = min(scores.values()), max(scores.values())
            rng    = (mx-mn) or 1
            scores = {t:(s-mn)/rng for t,s in scores.items()}
        return scores
    except Exception as e:
        st.warning(f"Collaborative scoring failed: {e}")
        return {}

def fuzzy_match_titles(collab_scores, content_titles):
    matched       = {}
    content_lower = {t.lower(): t for t in content_titles}
    for cl_title, score in collab_scores.items():
        cl_lower = cl_title.lower()
        if cl_lower in content_lower:
            matched[content_lower[cl_lower]] = score
            continue
        for ct_lower, ct_orig in content_lower.items():
            if cl_lower in ct_lower or ct_lower in cl_lower:
                if ct_orig not in matched:
                    matched[ct_orig] = score
                break
    return matched

def get_hybrid_recommendations(movie_title, n=5, alpha=0.5):
    content_scores = get_content_scores(movie_title)
    if not content_scores:
        return [], [], "no_data"
    content_titles = list(content_scores.keys())
    raw_collab     = get_item_based_collab_scores(movie_title, n=200)
    collab_scores  = fuzzy_match_titles(raw_collab, content_titles)
    mode   = "hybrid" if collab_scores else "content"
    hybrid = {}
    for t in set(content_scores.keys()):
        c = content_scores.get(t, 0.0)
        f = collab_scores.get(t, 0.0)
        hybrid[t] = alpha * c + (1.0-alpha) * f
    top     = sorted(hybrid, key=lambda x: hybrid[x], reverse=True)[:n]
    details = [fetch_movie_data(t) for t in top]
    return top, details, mode

# FIX: Removed nltk PorterStemmer dependency entirely.
# Simple substring genre matching works reliably without extra library installs.
def recommend_by_emotion(emotion_text, n=5):
    mood, genre, confidence = detect_mood(emotion_text)

    if movies.empty:
        return [], [], mood, genre, confidence

    genre_lower = genre.lower()

    # Try matching genre in tags (case-insensitive substring)
    matched_movies = movies[movies['tags'].str.lower().str.contains(genre_lower, na=False)]

    # Fallback: try partial stems (e.g. "comedy" -> "comed", "action" -> "action")
    if len(matched_movies) < n and len(genre_lower) > 4:
        stem = genre_lower[:max(4, len(genre_lower)-2)]
        matched_movies = movies[movies['tags'].str.lower().str.contains(stem, na=False)]

    # Fallback: drama
    if len(matched_movies) < n:
        matched_movies = movies[movies['tags'].str.lower().str.contains('drama', na=False)]

    # Ultimate fallback
    if matched_movies.empty:
        matched_movies = movies

    sample_size = min(len(matched_movies), n * 3)
    sampled_df  = matched_movies.sample(
        n=sample_size,
        random_state=random.randint(1, 10000)
    )

    filtered = []
    for _, row in sampled_df.iterrows():
        title = row['title']
        det   = fetch_movie_data(title)
        filtered.append((title, det))
        if len(filtered) >= n:
            break

    return [x[0] for x in filtered], [x[1] for x in filtered], mood, genre, confidence

# ============================================================
#  SECTION 9 — AUTH PAGES
# ============================================================

def auth_logo():
    st.markdown("""
    <div class="auth-logo">
      <span class="icon">🎬</span>
      <h1>CineMatch</h1>
      <p>Your personal movie companion</p>
    </div>
    """, unsafe_allow_html=True)

def step_dots(current: int):
    dots = ""
    for i in range(3):
        if i == current:
            dots += '<div class="step-dot active"></div>'
        elif i < current:
            dots += '<div class="step-dot done"></div>'
        else:
            dots += '<div class="step-dot"></div>'
    st.markdown(f'<div class="step-dots">{dots}</div>', unsafe_allow_html=True)

def page_email():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        auth_logo()
        step_dots(0)
        st.markdown('<div class="auth-step-label">Step 1 of 3</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Enter your email</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-subtitle">We will send a verification code to your email address.</div>',
            unsafe_allow_html=True
        )

        email_input = st.text_input(
            "Email Address",
            placeholder="you@example.com",
            key="email_field"
        )

        # FIX: Show error/warning BEFORE the button so it persists across reruns
        if st.session_state.otp_send_error:
            st.warning(st.session_state.otp_send_error)

        if st.button("Send OTP →", use_container_width=True):
            if not email_input or "@" not in email_input:
                st.error("Please enter a valid email address.")
            else:
                clean_email  = email_input.strip()
                is_new_user  = not email_exists(clean_email)
                otp          = generate_otp()
                purpose      = "register" if is_new_user else "login"

                print(f"\n🔑 [CineMatch Dev/Debug] OTP for {clean_email} ({purpose}) is: {otp}\n")

                save_otp(clean_email, otp)
                ok, msg = send_otp_email(clean_email, otp, purpose=purpose)

                st.session_state.auth_email  = clean_email
                st.session_state.auth_is_new = is_new_user
                st.session_state.otp_sent    = True

                if not ok:
                    st.session_state.otp_send_error = (
                        f"⚠️ **Email delivery failed:** {msg}\n\n"
                        f"**Local/Dev Fallback Active:** OTP printed to terminal. "
                        f"Check your console, copy the 6-digit code and enter it below."
                    )
                else:
                    st.session_state.otp_send_error = ""

                st.session_state.auth_stage = "otp"
                st.rerun()

def page_otp():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        auth_logo()
        step_dots(1)
        st.markdown('<div class="auth-step-label">Step 2 of 3</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Verify your email</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="auth-subtitle">A 6-digit code was sent to '
            f'<strong style="color:#c4b5fd">{st.session_state.auth_email}</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="otp-hint">📧 &nbsp;Check your inbox — the code expires in '
            f'<strong>{OTP_EXPIRY_MINUTES} minutes</strong>. Also check your spam folder.</div>',
            unsafe_allow_html=True
        )

        # FIX: Show persistent delivery-failure warning on OTP page too
        if st.session_state.otp_send_error:
            st.warning(st.session_state.otp_send_error)

        otp_input = st.text_input(
            "Enter 6-Digit OTP",
            placeholder="● ● ● ● ● ●",
            max_chars=6,
            key="otp_field"
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Back", use_container_width=True):
                st.session_state.auth_stage     = "email"
                st.session_state.otp_sent       = False
                st.session_state.otp_send_error = ""
                st.rerun()
        with c2:
            if st.button("Verify ✓", use_container_width=True):
                if not otp_input or len(otp_input) != 6:
                    st.error("Please enter the 6-digit OTP.")
                else:
                    ok, msg = verify_otp(st.session_state.auth_email, otp_input)
                    if ok:
                        st.session_state.otp_send_error = ""
                        st.session_state.auth_stage = (
                            "register" if st.session_state.auth_is_new else "login"
                        )
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("---")
        st.markdown(
            '<p style="text-align:center;font-size:12px;color:rgba(139,92,246,0.5)">Did not receive the code?</p>',
            unsafe_allow_html=True
        )
        if st.button("Resend OTP", use_container_width=True):
            otp = generate_otp()
            print(f"\n🔑 [CineMatch Dev/Debug] Resent OTP for {st.session_state.auth_email}: {otp}\n")
            save_otp(st.session_state.auth_email, otp)
            ok, msg = send_otp_email(st.session_state.auth_email, otp)
            if ok:
                st.session_state.otp_send_error = ""
                st.success("A new OTP has been sent!")
            else:
                st.session_state.otp_send_error = (
                    f"⚠️ **Email delivery failed:** {msg}\n\n"
                    f"OTP printed to terminal/console as fallback."
                )
                st.warning(st.session_state.otp_send_error)

def page_register():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        auth_logo()
        step_dots(2)
        st.markdown('<div class="auth-step-label">Step 3 of 3 — Create Account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Set up your profile</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="auth-subtitle">Email verified ✅ &nbsp;<strong style="color:#c4b5fd">'
            f'{st.session_state.auth_email}</strong></div>',
            unsafe_allow_html=True
        )

        username  = st.text_input("Username", placeholder="e.g. MovieFan2025")
        dob       = st.date_input(
            "Date of Birth",
            value=date(2000, 1, 1),
            min_value=date(1920, 1, 1),
            max_value=date.today(),
            help="Required for age verification"
        )
        password  = st.text_input("Create Password", type="password", placeholder="Minimum 6 characters")
        password2 = st.text_input("Confirm Password", type="password", placeholder="Repeat your password")

        if dob:
            age = calculate_age(dob.isoformat())
            if age >= 18:
                st.markdown(
                    f'<div class="adult-block"><p>✅ Age: <strong>{age} years</strong> — '
                    f'Full access granted (18+)</p></div>',
                    unsafe_allow_html=True
                )
            elif age >= 13:
                st.markdown(
                    f'<div class="age-block"><h3>⚠️ Age: {age} years</h3>'
                    f'<p>Adult-rated content (R / NC-17) will be hidden on your account.</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="age-block"><h3>🔞 Age: {age} years</h3>'
                    f'<p>Only family-friendly content will be shown.</p></div>',
                    unsafe_allow_html=True
                )

        if st.button("Create Account 🎬", use_container_width=True):
            if not username.strip():
                st.error("Please enter a username.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user_full(
                    username, st.session_state.auth_email,
                    password, dob.isoformat()
                )
                if ok:
                    user = get_user_by_email(st.session_state.auth_email)
                    st.session_state.user       = user["username"]
                    st.session_state.user_data  = dict(user)
                    st.session_state.auth_stage = "app"
                    st.rerun()
                else:
                    st.error(msg)

def page_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        auth_logo()
        step_dots(2)
        st.markdown('<div class="auth-step-label">Step 3 of 3 — Login</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Welcome back! 👋</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="auth-subtitle">Email verified ✅ &nbsp;<strong style="color:#c4b5fd">'
            f'{st.session_state.auth_email}</strong></div>',
            unsafe_allow_html=True
        )

        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Login 🎬", use_container_width=True):
            if not password:
                st.error("Please enter your password.")
            else:
                ok, msg, user = authenticate_user(st.session_state.auth_email, password)
                if ok:
                    st.session_state.user       = user["username"]
                    st.session_state.user_data  = dict(user)
                    st.session_state.auth_stage = "app"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        if st.button("← Use a different email", use_container_width=True):
            st.session_state.auth_stage     = "email"
            st.session_state.otp_send_error = ""
            st.rerun()

# ============================================================
#  SECTION 10 — CARD RENDERER
# ============================================================
def render_card(title, det, show_posters, show_metadata, show_overviews):
    poster    = det.get("poster", PLACEHOLDER_POSTER)
    plot      = det.get("plot", "")
    img_html  = f'<img class="movie-card-img" src="{poster}" alt="{title}"/>' if show_posters else ""
    meta_html = (
        f'<div class="meta-row">📅 {det.get("released","N/A")} • '
        f'⏱ {det.get("runtime","N/A")} • 🎭 {det.get("genre","N/A")}</div>'
        if show_metadata else ""
    )
    st.markdown(f"""
    <div class="movie-card">
      {img_html}
      <div class="movie-card-body">
        <div class="movie-title">{title}</div>
        {meta_html}
        <div class="overview">{(plot[:220]+'...') if len(plot)>220 else plot}</div>
      </div>
    </div>""", unsafe_allow_html=True)
    if show_overviews and plot:
        with st.expander("Read more"):
            st.write(plot)

# ============================================================
#  SECTION 11 — MAIN APP
# ============================================================
def page_app():
    if not RATINGS_CSV.exists():
        download_movielens()

    user     = st.session_state.user
    ud       = st.session_state.user_data or {}
    is_adult = bool(ud.get("is_adult", False))
    user_age = get_user_age(user) if user else None

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        # --- Account block ---
        st.markdown("### 👤 Account")
        st.success(f"**{user}**")
        age_str = f"{user_age} yrs" if user_age else "?"
        st.caption(f"🎂 {age_str}  |  {'✅ Adult (18+)' if is_adult else '🔒 Under 18'}")

        # Logout — clear state THEN rerun
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            keys_to_del = list(st.session_state.keys())
            for k in keys_to_del:
                del st.session_state[k]
            st.rerun()

        st.markdown("---")

        # --- Watchlist block ---
        st.markdown("### 🎬 Your Watchlist")
        wl = get_watchlist(user)
        if wl:
            for m in wl:
                # Two mini-columns: title | remove button
                c_title, c_btn = st.columns([3, 1])
                with c_title:
                    st.markdown(
                        f"<span style='font-size:13px;color:rgba(210,200,255,0.9)'>{m[:30]}</span>",
                        unsafe_allow_html=True
                    )
                with c_btn:
                    rk = "sw_" + hashlib.sha1(m.encode()).hexdigest()
                    st.button("✕", key=rk, on_click=cb_remove, args=(user, m),
                              help=f"Remove {m}")
        else:
            st.info("Your watchlist is empty.")

        st.markdown("---")

        # --- Settings block ---
        st.markdown("### ⚙️ Settings")
        num_recs       = st.slider("Recommendations", 1, 10, 5)
        show_posters   = st.checkbox("Show posters",  value=True)
        show_overviews = st.checkbox("Show overview", value=True)
        show_metadata  = st.checkbox("Show metadata", value=True)

        st.markdown("---")

        # --- Hybrid Settings ---
        st.markdown("### 🔀 Hybrid Settings")
        hybrid_alpha = st.slider(
            "Content ←→ Collaborative",
            min_value=0.0, max_value=1.0,
            value=st.session_state.hybrid_alpha, step=0.1,
            help="1.0 = content-based only | 0.5 = balanced | 0.0 = collaborative only"
        )
        st.session_state.hybrid_alpha = hybrid_alpha
        st.caption(
            f"📄 Content: **{int(hybrid_alpha*100)}%**  |  "
            f"👥 Collab: **{100-int(hybrid_alpha*100)}%**"
        )

        if RATINGS_CSV.exists():
            ml_ratings, _ = load_movielens()
            st.success(f"✅ MovieLens: {len(ml_ratings):,} ratings")
        else:
            st.warning("⏳ MovieLens data not found...")

    # Main content
    st.title("🎬 CineMatch — Movie Recommendation System")

    if not is_adult:
        st.markdown(
            '<div class="restricted-banner">🔒 <strong>Under-18 Mode:</strong> '
            'Adult-rated content (R / NC-17) is hidden on this account.</div>',
            unsafe_allow_html=True
        )

    st.write("Pick a movie **or** describe your mood to get recommendations.")

    options = movies['title'].values if not movies.empty else []
    query   = st.text_input("🔍 Search movie")
    emotion = st.text_input(
        "💬 How are you feeling?",
        placeholder="e.g. I am sad, feeling depressed, want some action..."
    )

    c1, c2 = st.columns(2)
    with c1: rec_btn  = st.button("🎬 Recommend by Movie")
    with c2: mood_btn = st.button("🎭 Recommend by Mood")

    mood_titles = st.session_state.mood_titles
    combined    = (
        list(mood_titles) + [t for t in options if t not in mood_titles]
        if mood_titles else list(options)
    )

    def _update_selected():
        st.session_state.selected_movie = st.session_state._selectbox_val

    if query and len(combined):
        matches = [t for t in combined if query.lower() in t.lower()]
        pool    = matches if matches else combined
        if not matches:
            st.info("No matches found — showing full list.")
        default_idx = (pool.index(st.session_state.selected_movie)
                       if st.session_state.selected_movie in pool else 0)
        selected = st.selectbox(
            "Matched titles" if matches else "Select a movie",
            pool, index=default_idx, key="_selectbox_val", on_change=_update_selected
        )
    else:
        default_idx = (combined.index(st.session_state.selected_movie)
                       if st.session_state.selected_movie in combined else 0)
        selected = st.selectbox(
            "Select a movie", combined, index=default_idx,
            key="_selectbox_val", on_change=_update_selected
        )

    if mood_btn and mood_titles:
        st.session_state.selected_movie = mood_titles[0]
        selected = mood_titles[0]
    elif st.session_state.selected_movie is None and len(combined):
        st.session_state.selected_movie = combined[0]
        selected = combined[0]
    else:
        selected = st.session_state.get("_selectbox_val", selected)

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.subheader("Selected Movie")
        if selected:
            det      = fetch_movie_data(selected)
            min_age  = min_age_for_rating(det.get("rated","N/A"))
            is_restr = not is_adult and min_age >= 17

            if show_posters:
                if is_restr:
                    st.warning(f"Rated {det.get('rated','?')} — restricted for your account")
                    st.image(PLACEHOLDER_POSTER, use_container_width=True)
                else:
                    st.image(det.get("poster", PLACEHOLDER_POSTER), use_container_width=True)

            if show_metadata:
                st.markdown(f"**⭐** {det.get('rating','N/A')} &nbsp;|&nbsp; **📅** {det.get('released','N/A')}")
                st.markdown(f"**⏱** {det.get('runtime','N/A')} &nbsp;|&nbsp; **🎭** {det.get('genre','N/A')}")
                st.markdown(f"**🔞 Rated:** {det.get('rated','N/A')}")

            if show_overviews:
                if is_restr:
                    st.info("Overview hidden (age restriction).")
                else:
                    with st.expander("Overview"):
                        st.write(det.get("plot",""))

            in_wl = selected in get_watchlist(user)
            if in_wl:
                st.button("➖ Remove from Watchlist", on_click=cb_remove, args=(user, selected))
            elif is_restr:
                st.button("🔒 Restricted — Cannot Add", disabled=True)
            else:
                st.button("➕ Add to Watchlist", on_click=cb_add, args=(user, selected))

            if st.session_state.wl_flash:
                action, title_f, ok = st.session_state.wl_flash
                if title_f == selected:
                    (st.success if ok else st.error)(
                        f"{'Added' if action=='added' else 'Removed'} '{title_f}'"
                    )
                st.session_state.wl_flash = None

    with col_main:
        if rec_btn and selected:
            with st.spinner("Finding hybrid recommendations..."):
                names, dets, rec_mode = get_hybrid_recommendations(
                    movie_title=selected,
                    n=num_recs,
                    alpha=st.session_state.hybrid_alpha,
                )
            st.session_state.last_recs      = (names, dets)
            st.session_state.last_mood_info = None
            st.session_state.mood_titles    = []
            st.session_state.rec_mode       = rec_mode

        if mood_btn:
            if not emotion.strip():
                st.warning("Please describe how you are feeling first.")
            else:
                with st.spinner("Analyzing your mood..."):
                    names, dets, mood, genre, conf = recommend_by_emotion(emotion, n=num_recs)
                st.session_state.last_recs      = (names, dets)
                st.session_state.last_mood_info = (mood, genre, conf)
                st.session_state.mood_titles    = names
                st.session_state.rec_mode       = "mood"

        rec_mode = st.session_state.rec_mode
        if rec_mode == "hybrid":
            alpha_pct = int(st.session_state.hybrid_alpha * 100)
            st.markdown(
                f'<div class="hybrid-badge">🔀 Hybrid — '
                f'{alpha_pct}% Content (TF-IDF) + {100-alpha_pct}% Collaborative (MovieLens SVD)'
                f'</div>', unsafe_allow_html=True
            )
        elif rec_mode == "content":
            st.markdown(
                '<div class="content-badge">📄 Content-based only '
                '(movie not found in MovieLens — try adjusting alpha or select another movie)'
                '</div>', unsafe_allow_html=True
            )

        if st.session_state.last_mood_info:
            mood, genre, conf = st.session_state.last_mood_info
            pct = max(int(conf*100), 10)
            st.markdown(f"""
            <div class="mood-badge">
              Detected mood: <strong>{mood_label(mood)}</strong>&nbsp;→&nbsp;<em>{genre}</em>
              &nbsp;&nbsp;<span style="font-size:12px;color:rgba(167,139,250,0.7);">
                Confidence:
                <span class="conf-wrap">
                  <span class="conf-bar" style="width:{pct}%"></span>
                </span>
                &nbsp;{pct}%
              </span>
            </div>""", unsafe_allow_html=True)

        names, dets = st.session_state.last_recs
        if names:
            for title, det in zip(names, dets):
                min_age = min_age_for_rating(det.get("rated","N/A"))
                restr   = not is_adult and min_age >= 17
                if restr:
                    safe = dict(det)
                    safe["poster"] = PLACEHOLDER_POSTER
                    safe["plot"]   = "Overview hidden (age restriction)."
                    render_card(title, safe, show_posters, show_metadata, show_overviews)
                else:
                    render_card(title, det, show_posters, show_metadata, show_overviews)

                btn_cols = st.columns([1, 2])
                sel_key  = "sel_" + hashlib.sha1(title.encode()).hexdigest()

                def _make_select_cb(t):
                    def _cb(): st.session_state.selected_movie = t
                    return _cb

                with btn_cols[0]:
                    st.button(f"🎯 Select — {title[:28]}", key=sel_key,
                              on_click=_make_select_cb(title))

                in_wl   = title in get_watchlist(user)
                btn_key = "rc_" + hashlib.sha1(title.encode()).hexdigest()
                with btn_cols[1]:
                    if in_wl:
                        st.button(f"✅ In Watchlist", key=btn_key,
                                  on_click=cb_remove, args=(user, title))
                    elif not restr:
                        st.button(f"➕ Add — {title[:28]}", key=btn_key,
                                  on_click=cb_add, args=(user, title))
        else:
            st.info("Click **Recommend by Movie** or **Recommend by Mood** to get started.")

    st.markdown("---")
    st.caption(
        "Hybrid Filtering: Content-based (TF-IDF cosine similarity) + "
        "Collaborative (MovieLens 100k SVD item similarity) | Email OTP verified accounts"
    )

# ============================================================
#  SECTION 12 — PAGE ROUTER
# ============================================================
stage    = st.session_state.auth_stage
is_auth  = stage in ["email", "otp", "register", "login"]
inject_theme_css(auth_mode=is_auth)

if stage == "email":
    page_email()
elif stage == "otp":
    page_otp()
elif stage == "register":
    page_register()
elif stage == "login":
    page_login()
elif stage == "app":
    if st.session_state.user:
        page_app()
    else:
        st.session_state.auth_stage = "email"
        st.rerun()
else:
    st.session_state.auth_stage = "email"
    st.rerun()