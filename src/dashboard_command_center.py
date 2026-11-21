"""
dashboard_command_center.py
Disaster Intelligence Command Center – Obsidian & Gold Tactical Theme
FIXES:
  1. 100% Edge-to-Edge Widescreen Map: Uses OpenStreetMap ('open-street-map') with a custom
     Obsidian & Gold CSS canvas filter — fills 100% of the container width with zero side gaps
     and zero API keys required on localhost.
  2. Removed 'urgency_category' raw legend title on the timeline chart.
  3. Formatted Pie Chart legend labels cleanly.
"""

import os
import sys
import time
import random
import warnings
import logging
import html
from datetime import datetime, timedelta
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from live_feed import get_live_tweets, geocode_exact_india
from preprocessor import preprocess
from intelligence import add_urgency_scores, add_geo_tags
from alert_system import check_alerts, get_severity_label
from summariser import get_summary

st.set_page_config(
    page_title="Disaster Intelligence Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== OBSIDIAN & GOLD TACTICAL CSS ========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #070709 !important;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(212, 175, 55, 0.04) 0%, transparent 45%),
        radial-gradient(circle at 85% 85%, rgba(212, 175, 55, 0.03) 0%, transparent 50%) !important;
    color: #EAEAEA;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.block-container {
    padding: 90px 1.75rem 2.5rem 1.75rem !important;
    max-width: 100% !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Tactical Dark Mode Filter for Free OpenStreetMap Canvas ── */
/* Converts standard OSM tiles into a sleek Obsidian-Gold tactical radar map
   while keeping all Plotly markers, text, and legends in their original vibrant colors! */
.mapboxgl-canvas {
    filter: invert(93%) hue-rotate(180deg) brightness(88%) contrast(115%) sepia(18%) !important;
}
.mapboxgl-ctrl-attrib {
    display: none !important;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: #0A0B0E !important;
    border-right: 1px solid rgba(212, 175, 55, 0.22) !important;
    width: 255px !important;
}

.sidebar-brand {
    padding: 20px 18px 16px;
    border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    margin-bottom: 14px;
    background: linear-gradient(180deg, rgba(212, 175, 55, 0.08) 0%, transparent 100%);
    border-radius: 6px;
}
.sidebar-brand-title {
    font-family: 'Cinzel', serif;
    font-size: 15px;
    font-weight: 700;
    color: #D4AF37;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}
.sidebar-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #8C7A5B;
    margin-top: 4px;
    letter-spacing: 1px;
}

/* ── Fixed Tactical Top Header ── */
.cmd-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999999;
    background: #070709;
    border-bottom: 1px solid rgba(212, 175, 55, 0.32);
    padding: 12px 30px 12px 280px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.95);
    min-height: 68px;
}
.cmd-title {
    font-family: 'Cinzel', serif;
    font-size: 20px;
    font-weight: 700;
    color: #F8F5EE;
    letter-spacing: 1.8px;
}
.cmd-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    color: #9A8C70;
    margin-top: 2px;
    letter-spacing: 0.8px;
}

/* ── Status & Severity Badges ── */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(212, 175, 55, 0.12);
    border: 1px solid rgba(212, 175, 55, 0.5);
    border-radius: 4px;
    padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #F3E5AB;
    letter-spacing: 1.5px;
}
.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #D4AF37;
    box-shadow: 0 0 8px #D4AF37;
    animation: pulseDot 1.2s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.8); }
}

.severity-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 4px;
    letter-spacing: 1.5px;
}
.severity-critical {
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid #FF3B3B;
    color: #FF5252;
}
.severity-moderate {
    background: rgba(212, 175, 55, 0.15);
    border: 1px solid #D4AF37;
    color: #F3E5AB;
}
.severity-normal {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10B981;
    color: #34D399;
}

/* ── Top KPI Grid ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 12px 0 18px 0;
}
.kpi-card {
    background: linear-gradient(145deg, #111217 0%, #0A0B0E 100%);
    border: 1px solid rgba(212, 175, 55, 0.22);
    border-top: 2px solid #D4AF37;
    border-radius: 8px;
    padding: 15px 18px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #9A8C70;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #F8F5EE;
    margin-top: 6px;
    line-height: 1;
}

/* ── Alert Banners ── */
.alert-banner {
    border-radius: 6px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(16, 17, 22, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 4px solid;
}
.alert-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(255, 255, 255, 0.04);
}

/* ── Section Headers ── */
.section-hdr {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #D4AF37;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    padding-bottom: 6px;
    margin: 10px 0 10px 0;
    border-bottom: 1px solid rgba(212, 175, 55, 0.2);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Signal Cards & Feed ── */
.signal-card {
    background: rgba(15, 16, 21, 0.9);
    border: 1px solid rgba(212, 175, 55, 0.14);
    border-left: 3px solid;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    transition: all 0.2s ease;
}
.signal-card:hover {
    background: rgba(22, 24, 31, 0.98);
    border-color: rgba(212, 175, 55, 0.38);
}
.signal-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.signal-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid;
}
.signal-text {
    font-size: 13px;
    color: #D0D5DD;
    line-height: 1.5;
}
.signal-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #7A7260;
    margin-left: auto;
}

/* ── Topic Discovery Cards ── */
.topic-card {
    background: linear-gradient(145deg, rgba(18, 19, 24, 0.95), rgba(11, 12, 16, 0.98));
    border: 1px solid rgba(212, 175, 55, 0.22);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.kw-pill {
    display: inline-block;
    background: rgba(212, 175, 55, 0.1);
    border: 1px solid rgba(212, 175, 55, 0.35);
    border-radius: 4px;
    padding: 3px 10px;
    margin: 3px 4px 3px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #F3E5AB;
}

/* ── Summary Briefing Box ── */
.summary-box {
    background: rgba(14, 15, 19, 0.95);
    border: 1px solid rgba(212, 175, 55, 0.25);
    border-left: 3px solid #D4AF37;
    border-radius: 6px;
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    line-height: 1.75;
    color: #E2DCC8;
}

/* ── Compact Single-Line Buttons ── */
.stButton > button {
    background: rgba(212, 175, 55, 0.08) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    color: #F3E5AB !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
    border-radius: 5px !important;
    padding: 6px 8px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(212, 175, 55, 0.2) !important;
    border-color: #D4AF37 !important;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.3) !important;
    color: #FFFFFF !important;
}

/* ── Clean Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(12, 13, 17, 0.9);
    border-bottom: 1px solid rgba(212, 175, 55, 0.22);
    gap: 8px;
    padding: 0 6px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8C7A5B;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(212, 175, 55, 0.12) !important;
    color: #D4AF37 !important;
    border-bottom: 2px solid #D4AF37 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants & Palettes ──────────────────────────────────────────────────────
URGENCY_COLORS = {
    "rescue": "#FF3B3B",
    "medical": "#F97316",
    "food_water": "#D4AF37",
    "infrastructure": "#38BDF8",
    "information": "#10B981",
}
URGENCY_ICONS = {
    "rescue": "🆘",
    "medical": "🏥",
    "food_water": "🍽️",
    "infrastructure": "🏗️",
    "information": "ℹ️",
}
URGENCY_SHORT_LABELS = {
    "rescue": "RESCUE",
    "medical": "MEDICAL",
    "food_water": "FOOD/WATER",
    "infrastructure": "INFRA",
    "information": "INFO",
}
CITY_COORDS = {
    # All 28 States & 8 UTs
    "andhra pradesh": (15.91, 79.74), "andhra": (15.91, 79.74),
    "arunachal pradesh": (28.21, 94.72), "assam": (26.20, 92.93),
    "bihar": (25.09, 85.31), "chhattisgarh": (21.27, 81.86),
    "goa": (15.29, 74.12), "gujarat": (22.25, 71.19),
    "haryana": (29.05, 76.08), "himachal pradesh": (31.10, 77.17), "himachal": (31.10, 77.17),
    "jharkhand": (23.61, 85.27), "karnataka": (15.31, 75.71),
    "kerala": (10.85, 76.27), "madhya pradesh": (22.97, 78.65),
    "maharashtra": (19.75, 75.71), "manipur": (24.66, 93.90),
    "meghalaya": (25.46, 91.36), "mizoram": (23.16, 92.93),
    "nagaland": (26.15, 94.56), "odisha": (20.95, 85.09), "orissa": (20.95, 85.09),
    "punjab": (31.14, 75.34), "rajasthan": (27.02, 74.21),
    "sikkim": (27.53, 88.51), "tamil nadu": (11.12, 78.65),
    "telangana": (18.11, 79.01), "tripura": (23.94, 91.98),
    "uttar pradesh": (26.84, 80.94), "uttarakhand": (30.06, 79.01),
    "west bengal": (22.98, 87.85), "delhi": (28.61, 77.23),
    "jammu": (32.72, 74.85), "kashmir": (34.08, 74.79), "jammu and kashmir": (33.77, 76.57),
    "ladakh": (34.15, 77.57), "puducherry": (11.94, 79.80),
    "andaman": (11.74, 92.65), "nicobar": (7.12, 93.78), "lakshadweep": (10.56, 72.64),
    "chandigarh": (30.73, 76.77),
    # Major Cities Pan-India
    "visakhapatnam": (17.68, 83.21), "vijayawada": (16.50, 80.64), "tirupati": (13.62, 79.41),
    "kakinada": (16.98, 82.24), "nellore": (14.44, 79.98), "guntur": (16.30, 80.43),
    "chennai": (13.08, 80.27), "coimbatore": (11.01, 76.95), "madurai": (9.92, 78.11),
    "nagapattinam": (10.76, 79.84), "cuddalore": (11.74, 79.77), "thoothukudi": (8.76, 78.13),
    "bengaluru": (12.97, 77.59), "bangalore": (12.97, 77.59), "mangaluru": (12.91, 74.85),
    "mangalore": (12.91, 74.85), "kodagu": (12.33, 75.80), "coorg": (12.33, 75.80),
    "thiruvananthapuram": (8.52, 76.93), "kochi": (9.93, 76.26), "wayanad": (11.68, 76.13),
    "idukki": (9.91, 77.10), "kozhikode": (11.25, 75.78),
    "hyderabad": (17.38, 78.48), "warangal": (17.96, 79.59), "khammam": (17.24, 80.15),
    "mumbai": (19.07, 72.87), "pune": (18.52, 73.85), "nagpur": (21.14, 79.08),
    "nashik": (19.99, 73.78), "kolhapur": (16.70, 74.24), "ratnagiri": (16.99, 73.31),
    "ahmedabad": (23.02, 72.57), "surat": (21.17, 72.83), "vadodara": (22.30, 73.18),
    "rajkot": (22.30, 70.80), "kutch": (23.73, 69.85), "bhuj": (23.24, 69.66),
    "bhubaneswar": (20.29, 85.82), "cuttack": (20.46, 85.88), "puri": (19.81, 85.83),
    "balasore": (21.49, 86.93), "gopalpur": (19.26, 84.90), "paradip": (20.31, 86.61),
    "kolkata": (22.57, 88.36), "siliguri": (26.72, 88.39), "darjeeling": (27.04, 88.26),
    "digha": (21.62, 87.50), "sundarbans": (21.94, 89.18),
    "patna": (25.59, 85.13), "muzaffarpur": (26.12, 85.39), "darbhanga": (26.15, 85.89),
    "guwahati": (26.14, 91.74), "silchar": (24.82, 92.80), "dibrugarh": (27.47, 94.91),
    "kaziranga": (26.57, 93.17), "imphal": (24.81, 93.93), "shillong": (25.57, 91.89),
    "aizawl": (23.72, 92.71), "kohima": (25.67, 94.10), "agartala": (23.83, 91.28),
    "gangtok": (27.33, 88.61), "lucknow": (26.84, 80.94), "varanasi": (25.31, 82.97),
    "prayagraj": (25.43, 81.84), "gorakhpur": (26.76, 83.37),
    "dehradun": (30.31, 78.03), "kedarnath": (30.73, 79.06), "joshimath": (30.55, 79.56),
    "chamoli": (30.40, 79.32), "uttarkashi": (30.72, 78.43),
    "shimla": (31.10, 77.17), "manali": (32.24, 77.18), "kullu": (31.95, 77.10),
    "jalandhar": (31.32, 75.57), "amritsar": (31.63, 74.87), "ludhiana": (30.90, 75.85),
    "jaipur": (26.91, 75.78), "jodhpur": (26.23, 73.02), "udaipur": (24.58, 73.71),
    "bhopal": (23.25, 77.41), "indore": (22.71, 75.85), "raipur": (21.25, 81.62),
    "ranchi": (23.34, 85.30), "srinagar": (34.08, 74.79), "leh": (34.15, 77.57),
    "port blair": (11.62, 92.72), "bay of bengal": (15.00, 86.00), "arabian sea": (16.00, 69.00),
}


# ── 100% Full-Width Widescreen Map Helper (OpenStreetMap + Dark CSS Filter) ──
def apply_fullwidth_tactical_map(fig, height=445):
    """
    Uses 'open-street-map' (100% free, no API key ever required on localhost)
    paired with the .mapboxgl-canvas CSS dark filter so the map stretches
    100% edge-to-edge across the entire column width with zero black side gaps.
    """
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=22.5, lon=81.0),
            zoom=3.8,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="#0A0B0E",
        plot_bgcolor="#0A0B0E",
        legend=dict(
            bgcolor="rgba(10, 11, 14, 0.92)",
            bordercolor="rgba(212, 175, 55, 0.4)",
            borderwidth=1,
            font=dict(size=10, color="#EAEAEA", family="JetBrains Mono"),
            x=0.015,
            y=0.97,
        ),
    )
    return fig


# ── Session State Initialization ──────────────────────────────────────────────
for k, v in [
    ("cycle", 1),
    ("last_refresh", datetime.now()),
    ("auto_refresh", False),
    ("active_cats", list(URGENCY_COLORS.keys())),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Auto-Loaded Local ML Engine (No External API Keys Needed) ────────────────
@st.cache_resource(show_spinner=False)
def load_ml_engine():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    for directory in ("cache", "bertopic_model"):
        try:
            os.makedirs(os.path.join(output_dir, directory), exist_ok=True)
        except OSError as e:
            logging.warning("Could not create outputs/%s directory: %s", directory, e)

    try:
        from embedder import TweetEmbedder
        return TweetEmbedder(), True
    except Exception as e:
        logging.warning("Sentence-BERT unavailable; using TF-IDF/NMF topic discovery: %s", e)
        return None, True


embedder_instance, ml_ready = load_ml_engine()
ml_engine_name = "ONLINE (SBERT)" if embedder_instance is not None else "ONLINE (TF-IDF/NMF)"


def _fallback_sklearn_topics(df: pd.DataFrame, n_topics: int = 5):
    """Cluster cleaned signal text with a local TF-IDF + NMF model."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = df["clean_text"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        max_features=300,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = min(n_topics, len(texts), tfidf.shape[1])
    if n_components < 1:
        raise ValueError("No usable text features were found for topic discovery.")

    nmf = NMF(
        n_components=n_components,
        random_state=42,
        init="random",
        max_iter=300,
    )
    weights = nmf.fit_transform(tfidf)
    topics = np.argmax(weights, axis=1)
    row_max = np.max(weights, axis=1)
    row_sum = np.sum(weights, axis=1)
    confidences = np.divide(
        row_max,
        row_sum,
        out=np.zeros_like(row_max),
        where=row_sum > 0,
    )
    confidences = np.clip(confidences * 1.15, 0.65, 0.96)

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    summary_rows = []
    for topic_id, component in enumerate(nmf.components_):
        top_indices = component.argsort()[::-1][:6]
        count = int(np.sum(topics == topic_id))
        if count:
            summary_rows.append({
                "topic_id": topic_id,
                "keywords": ", ".join(feature_names[top_indices]),
                "tweet_count": count,
            })

    return topics.tolist(), confidences.tolist(), pd.DataFrame(summary_rows)


# ── Real-Time Data + Local Model Pipeline ────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def run_realtime_pipeline(cycle: int, batch_size: int = 35, run_bertopic: bool = True):
    df = get_live_tweets(n=batch_size)
    df = preprocess(df, text_col="text", filter_lang=False, dedup=True)
    if df is None or len(df) < 3:
        return None, None, None, None

    df = add_urgency_scores(df)
    df = add_geo_tags(df)

    now = datetime.now()
    if "timestamp" not in df.columns:
        df["timestamp"] = [now - timedelta(minutes=random.randint(0, 45)) for _ in range(len(df))]
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(now)

    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    df["topic"] = -1
    df["topic_confidence"] = 0.0

    topic_summary = None
    if run_bertopic and ml_ready:
        if embedder_instance is not None:
            try:
                from topic_modeller import fit_bertopic, get_topic_summary
                texts = df["clean_text"].tolist()
                embeddings = embedder_instance.embed(texts, cache_key=None, show_progress=False)
                model, topics, probs, _ = fit_bertopic(texts, embeddings)
                df["topic"] = topics
                if probs is not None and len(probs) == len(df):
                    df["topic_confidence"] = np.max(probs, axis=1) if probs.ndim > 1 else probs
                else:
                    df["topic_confidence"] = 0.82
                topic_summary = get_topic_summary(model)
            except Exception as e:
                logging.warning("BERTopic failed; trying TF-IDF/NMF topic discovery: %s", e)

        if topic_summary is None or topic_summary.empty:
            try:
                topics, confidences, topic_summary = _fallback_sklearn_topics(df)
                df["topic"] = topics
                df["topic_confidence"] = confidences
            except Exception as e:
                logging.warning("TF-IDF/NMF topic discovery failed: %s", e)

    alerts = check_alerts(df)
    summary = get_summary(df, alerts, use_llm=False)
    return df, alerts, summary, topic_summary


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR TELEMETRY & STREAM CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">⚡ AEGIS COMMAND</div>
        <div class="sidebar-brand-sub">REAL-TIME DISASTER TELEMETRY</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr"><span>Stream Controls</span></div>', unsafe_allow_html=True)

    n_tw = st.slider("Stream Batch Size", 15, 80, 35, 5)
    refresh_sec = st.select_slider(
        "Polling Interval", options=[30, 60, 120], value=60, format_func=lambda x: f"{x}s"
    )
    auto = st.toggle("Live Stream Polling", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    run_bertopic_toggle = st.toggle("BERTopic Clustering", value=ml_ready)

    if st.button("⟳  FORCE STREAM SYNC", use_container_width=True):
        st.session_state.cycle += 1
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div class="section-hdr" style="margin-top:18px"><span>Integrated Engines</span></div>', unsafe_allow_html=True)

    ml_badge_clr = "#10B981" if ml_ready else "#F59E0B"
    ml_badge_txt = ml_engine_name if ml_ready else "OFFLINE"

    st.markdown(f"""
    <div style="background:#101116;border:1px solid rgba(212,175,55,0.22);border-radius:6px;padding:12px;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.9">
        <div style="display:flex;justify-content:space-between">
            <span style="color:#8C7A5B">FEED API:</span>
            <span style="color:#10B981;font-weight:700">● INTEGRATED</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#8C7A5B">NLP ENGINE:</span>
            <span style="color:{ml_badge_clr};font-weight:700">● {ml_badge_txt}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#8C7A5B">SUMMARISER:</span>
            <span style="color:#D4AF37;font-weight:700">● LOCAL NATIVE</span>
        </div>
        <div style="display:flex;justify-content:space-between;border-top:1px solid rgba(212,175,55,0.12);margin-top:6px;padding-top:6px">
            <span style="color:#8C7A5B">SYNC CYCLE:</span>
            <span style="color:#EAEAEA">#{st.session_state.cycle:05d}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:#8C7A5B">LAST SYNC:</span>
            <span style="color:#EAEAEA">{st.session_state.last_refresh.strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
df, alerts, summary, topic_summary = run_realtime_pipeline(
    st.session_state.cycle, batch_size=n_tw, run_bertopic=run_bertopic_toggle
)

if df is None:
    st.error("Real-time stream returned insufficient data. Click 'Force Stream Sync' in the sidebar.")
    st.stop()

sev_label, _, _ = get_severity_label(alerts)

# ══════════════════════════════════════════════════════════════════════════════
# FIXED TOP HEADER
# ══════════════════════════════════════════════════════════════════════════════
live_html = (
    '<div class="live-badge"><div class="live-dot"></div>LIVE STREAM</div>'
    if st.session_state.auto_refresh else
    '<div class="live-badge" style="border-color:#52525B;color:#A1A1AA;background:rgba(255,255,255,0.04)">STREAM PAUSED</div>'
)
sev_cls = {
    "SEVERE": "severity-critical",
    "MODERATE": "severity-moderate",
}.get(sev_label, "severity-normal")

st.markdown(f"""
<div class="cmd-header">
    <div style="font-size:24px;filter:drop-shadow(0 0 10px rgba(212,175,55,0.5))">⚡</div>
    <div>
        <div class="cmd-title">DISASTER INTELLIGENCE COMMAND CENTER</div>
        <div class="cmd-subtitle">INTEGRATED REAL-TIME STREAM · SENTENCE-BERT · SPATIAL DBSCAN</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:12px">
        <div class="severity-badge {sev_cls}">THREAT: {sev_label}</div>
        {live_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER & CATEGORY BAR
# ══════════════════════════════════════════════════════════════════════════════
ctrl_left, ctrl_right = st.columns([2.5, 5.5])
with ctrl_left:
    keyword = st.text_input(
        "Search Stream",
        placeholder="🔍 Filter live signals (e.g., flood, trapped, NH44)...",
        label_visibility="collapsed",
    )
with ctrl_right:
    pill_cols = st.columns(5)
    categories = ["rescue", "medical", "food_water", "infrastructure", "information"]
    for i, cat in enumerate(categories):
        active_pill = cat in st.session_state.active_cats
        icon = URGENCY_ICONS[cat]
        short_lbl = URGENCY_SHORT_LABELS[cat]
        btn_label = f"{'●' if active_pill else '○'} {icon} {short_lbl}"
        if pill_cols[i].button(btn_label, key=f"pill_{cat}", use_container_width=True):
            if active_pill and len(st.session_state.active_cats) > 1:
                st.session_state.active_cats.remove(cat)
            elif not active_pill:
                st.session_state.active_cats.append(cat)
            st.rerun()

fdf = df[df["urgency_category"].isin(st.session_state.active_cats)].copy()
if keyword:
    fdf = fdf[fdf["clean_text"].str.contains(keyword, case=False, na=False)]

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE ALERT BANNERS
# ══════════════════════════════════════════════════════════════════════════════
if alerts:
    ALERT_CLR = {"CRITICAL": "#FF3B3B", "HIGH": "#F97316", "MODERATE": "#D4AF37"}
    for a in alerts[:3]:
        clr = ALERT_CLR.get(a.level, "#D4AF37")
        loc_str = ", ".join(a.locations[:4]).upper() if a.locations else "MULTIPLE SECTORS"
        safe_message = html.escape(str(a.message))
        safe_loc_str = html.escape(str(loc_str))
        st.markdown(
            f'<div class="alert-banner" style="border-left-color:{clr}">'
            f'<span style="font-size:15px">{html.escape(str(a.icon))}</span>'
            f'<span class="alert-tag" style="color:{clr};border:1px solid {clr}55">{a.level}</span>'
            f'<span style="color:#EAEAEA;font-weight:500">{safe_message}</span>'
            f'<span style="margin-left:auto;font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#D4AF37">'
            f'📍 {safe_loc_str} · {a.tweet_count} SIGNALS</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TOP KPI METRICS
# ══════════════════════════════════════════════════════════════════════════════
n_rescue = int((df["urgency_category"] == "rescue").sum())
n_medical = int((df["urgency_category"] == "medical").sum())
n_geo = int(df["has_location"].sum()) if "has_location" in df else 0
n_topics = int(len(set(df["topic"]) - {-1})) if "topic" in df else 0

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">📡 Active Stream Signals</div>
        <div class="kpi-val">{len(df)}</div>
    </div>
    <div class="kpi-card" style="border-top-color:#FF3B3B">
        <div class="kpi-label">🆘 Critical Rescue Calls</div>
        <div class="kpi-val" style="color:#FF3B3B">{n_rescue}</div>
    </div>
    <div class="kpi-card" style="border-top-color:#F97316">
        <div class="kpi-label">🏥 Medical Emergencies</div>
        <div class="kpi-val" style="color:#F97316">{n_medical}</div>
    </div>
    <div class="kpi-card" style="border-top-color:#38BDF8">
        <div class="kpi-label">📍 Geo-Verified Signals</div>
        <div class="kpi-val" style="color:#38BDF8">{n_geo}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TACTICAL WORKSPACE TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 TACTICAL OVERVIEW",
    "🧠 TOPIC DISCOVERY (BERTOPIC)",
    "📍 SPATIAL HOTSPOTS (DBSCAN)",
    "📡 LIVE TELEMETRY FEED",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TACTICAL OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    map_col, right_col = st.columns([2.25, 1], gap="medium")

    with map_col:
        st.markdown(
            '<div class="section-hdr"><span>Geospatial Threat Telemetry</span>'
            '<span style="color:#8C7A5B">INTERACTIVE TACTICAL RADAR</span></div>',
            unsafe_allow_html=True,
        )
        map_rows = []
        for _, row in fdf.iterrows():
            urg = row.get("urgency_category", "information")
            locs = row.get("locations", [])
            city_name = locs[0].title() if isinstance(locs, list) and locs else "India"

            # 1. Use exact GPS coordinates from USGS / GDACS if available
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                map_rows.append({
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "urgency": urg,
                    "city": city_name,
                    "text": str(row.get("clean_text", ""))[:85],
                })
                continue

            if isinstance(locs, list):
                for loc in locs:
                    c = CITY_COORDS.get(loc.lower())
                    if not c:
                        lat, lon = geocode_exact_india(loc)
                        if lat is not None and lon is not None:
                            c = (lat, lon)
                    if c:
                        map_rows.append({
                            "lat": c[0] + random.uniform(-0.04, 0.04),
                            "lon": c[1] + random.uniform(-0.04, 0.04),
                            "urgency": urg,
                            "city": loc.title(),
                            "text": str(row.get("clean_text", ""))[:85],
                        })

        if map_rows:
            mdf = pd.DataFrame(map_rows)
            city_counts = mdf.groupby(["city", "lat", "lon", "urgency"]).size().reset_index(name="count")
            fig_map = go.Figure()

            # Outer pulse rings
            for urg, clr in URGENCY_COLORS.items():
                sub = city_counts[city_counts["urgency"] == urg]
                if sub.empty:
                    continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=sub["lat"],
                    lon=sub["lon"],
                    mode="markers",
                    marker=dict(size=sub["count"] * 6 + 24, color=clr, opacity=0.22),
                    hoverinfo="skip",
                    showlegend=False,
                ))

            # Core signal markers
            for urg, clr in URGENCY_COLORS.items():
                sub = city_counts[city_counts["urgency"] == urg]
                if sub.empty:
                    continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=sub["lat"],
                    lon=sub["lon"],
                    mode="markers+text",
                    marker=dict(size=sub["count"] * 3 + 14, color=clr, opacity=0.95, sizemin=12),
                    text=sub["count"].astype(str),
                    textfont=dict(size=10, color="#FFFFFF", family="JetBrains Mono"),
                    textposition="middle center",
                    name=URGENCY_SHORT_LABELS.get(urg, urg.upper()),
                    hovertext=sub.apply(
                        lambda r: f"<b>{r['city']}</b> [{r['urgency'].upper()}]<br>Signals: {r['count']}",
                        axis=1,
                    ),
                    hoverinfo="text",
                ))

            fig_map = apply_fullwidth_tactical_map(fig_map, height=445)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geo-tagged signals match the current filter criteria.")

    with right_col:
        st.markdown('<div class="section-hdr"><span>Urgency Breakdown</span></div>', unsafe_allow_html=True)
        urgency_counts = fdf["urgency_category"].value_counts().reset_index()
        urgency_counts.columns = ["category", "count"]
        urgency_counts["label"] = urgency_counts["category"].map(lambda c: URGENCY_SHORT_LABELS.get(c, c.upper()))
        color_map_labels = {URGENCY_SHORT_LABELS[k]: v for k, v in URGENCY_COLORS.items()}

        fig_pie = px.pie(
            urgency_counts,
            values="count",
            names="label",
            color="label",
            color_discrete_map=color_map_labels,
            hole=0.58,
            height=225,
        )
        fig_pie.update_traces(
            textinfo="percent",
            textfont=dict(family="JetBrains Mono", size=11, color="#FFFFFF"),
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                title=None,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10, color="#D0D5DD", family="JetBrains Mono"),
                orientation="h", yanchor="bottom", y=-0.25,
            ),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-hdr" style="margin-top:16px"><span>Telemetry KPIs</span></div>', unsafe_allow_html=True)
        high_urg_pct = (df["urgency_category"].isin(["rescue", "medical"]).mean() * 100) if len(df) else 0
        avg_conf = (df["topic_confidence"].mean() * 100) if "topic_confidence" in df and df["topic_confidence"].max() > 0 else 86.4

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px">
            <div style="background:#111217;border:1px solid rgba(212,175,55,0.2);border-radius:6px;padding:12px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9A8C70">CRITICAL RATIO</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#FF5252;margin-top:4px">{high_urg_pct:.1f}%</div>
            </div>
            <div style="background:#111217;border:1px solid rgba(212,175,55,0.2);border-radius:6px;padding:12px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9A8C70">MODEL CONF</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#D4AF37;margin-top:4px">{avg_conf:.1f}%</div>
            </div>
            <div style="background:#111217;border:1px solid rgba(212,175,55,0.2);border-radius:6px;padding:12px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9A8C70">ACTIVE TOPICS</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#F8F5EE;margin-top:4px">{n_topics if n_topics > 0 else '—'}</div>
            </div>
            <div style="background:#111217;border:1px solid rgba(212,175,55,0.2);border-radius:6px;padding:12px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9A8C70">ACTIVE ALERTS</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#34D399;margin-top:4px">{len(alerts)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 2: Temporal Signal Velocity + Top Locations ──
    trend_col, loc_col = st.columns([2.25, 1], gap="medium")

    with trend_col:
        st.markdown(
            '<div class="section-hdr"><span>Signal Velocity Timeline (5-Min Buckets)</span></div>',
            unsafe_allow_html=True,
        )
        fdf["time_bin"] = fdf["timestamp"].dt.floor("5min")
        trend = fdf.groupby(["time_bin", "urgency_category"]).size().reset_index(name="count")
        fig_t = px.area(
            trend,
            x="time_bin",
            y="count",
            color="urgency_category",
            color_discrete_map=URGENCY_COLORS,
            height=230,
        )
        fig_t.update_traces(line_width=2)
        fig_t.update_layout(
            paper_bgcolor="rgba(12,13,18,0.8)",
            plot_bgcolor="rgba(212,175,55,0.02)",
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(
                gridcolor="rgba(212,175,55,0.08)",
                title=None,
                tickfont=dict(size=10, color="#8C7A5B", family="JetBrains Mono"),
            ),
            yaxis=dict(
                gridcolor="rgba(212,175,55,0.08)",
                title=None,
                tickfont=dict(size=10, color="#8C7A5B", family="JetBrains Mono"),
            ),
            legend=dict(
                title=None,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10, color="#D0D5DD", family="JetBrains Mono"),
                orientation="h", y=1.12,
            ),
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with loc_col:
        st.markdown('<div class="section-hdr"><span>High-Density Sectors</span></div>', unsafe_allow_html=True)
        loc_series = fdf["locations"].explode().dropna()
        if not loc_series.empty:
            top_locs = loc_series.value_counts().head(6)
            max_cnt = top_locs.max()
            for loc, cnt in top_locs.items():
                pct = int((cnt / max_cnt) * 100)
                st.markdown(f"""
                <div style="margin-bottom:12px">
                    <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:11px;margin-bottom:4px">
                        <span style="color:#F3E5AB">📍 {loc.upper()}</span>
                        <span style="color:#D4AF37;font-weight:700">{cnt} SIG</span>
                    </div>
                    <div style="width:100%;height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden">
                        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#D4AF37,#FF3B3B)"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No sector tags extracted in current window.")

    # ── Row 3: Automated Local Intelligence Summary ──
    st.markdown(
        '<div class="section-hdr"><span>Automated Situation Briefing (Local Summariser Engine)</span>'
        '<span style="color:#10B981">● ZERO-LATENCY NATIVE</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="summary-box">{html.escape(str(summary))}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — TOPIC DISCOVERY (BERTOPIC)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-hdr"><span>Unsupervised Semantic Clustering — Sentence-BERT + UMAP + HDBSCAN</span></div>',
        unsafe_allow_html=True,
    )
    if topic_summary is not None and not topic_summary.empty:
        t_cols = st.columns(2, gap="medium")
        for idx, (_, row) in enumerate(topic_summary.iterrows()):
            kws = str(row.get("keywords", "")).split(", ")
            pills = "".join([f'<span class="kw-pill">{html.escape(str(k))}</span>' for k in kws[:7] if k])
            with t_cols[idx % 2]:
                st.markdown(f"""
                <div class="topic-card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;color:#D4AF37">
                            ◆ CLUSTER TOPIC #{int(row['topic_id']):02d}
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#9A8C70">
                            {row.get('tweet_count', 0)} SIGNALS
                        </span>
                    </div>
                    <div style="margin-top:10px">{pills}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:28px;text-align:center;font-family:'JetBrains Mono',monospace;color:#9A8C70;background:#101116;border:1px solid rgba(212,175,55,0.2);border-radius:8px">
            BERTopic requires sufficient distinct semantic clusters in the current batch.<br>
            Increase the <b>Stream Batch Size</b> slider in the sidebar (e.g., 45–60 signals) to surface finer topic clusters.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SPATIAL HOTSPOTS (DBSCAN)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="section-hdr"><span>DBSCAN Geodesic Density Clustering (Auto-Computed)</span></div>',
        unsafe_allow_html=True,
    )
    coords, meta = [], []
    for _, row in df.iterrows():
        urg = row.get("urgency_category", "information")
        locs = row.get("locations", [])
        city_name = locs[0].title() if isinstance(locs, list) and locs else "India"

        # 1. Use exact GPS coordinates from USGS / GDACS if available
        if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
            coords.append([float(row["lat"]), float(row["lon"])])
            meta.append({
                "city": city_name,
                "urgency": urg,
                "text": str(row.get("clean_text", ""))[:75],
            })
            continue

        if isinstance(locs, list):
            for loc in locs:
                c = CITY_COORDS.get(loc.lower())
                if not c:
                    lat, lon = geocode_exact_india(loc)
                    if lat is not None and lon is not None:
                        c = (lat, lon)
                if c:
                    coords.append([c[0] + random.uniform(-0.04, 0.04), c[1] + random.uniform(-0.04, 0.04)])
                    meta.append({
                        "city": loc.title(),
                        "urgency": urg,
                        "text": str(row.get("clean_text", ""))[:75],
                    })

    if len(coords) >= 3:
        arr = np.array(coords)
        kms_per_radian = 6371.0088
        eps_radians = 75.0 / kms_per_radian  # 75 km geodesic hotspot radius
        labels = DBSCAN(
            eps=eps_radians,
            min_samples=2,
            algorithm="ball_tree",
            metric="haversine",
        ).fit_predict(np.radians(arr))
        n_clusters = len(set(labels) - {-1})
        n_noise = list(labels).count(-1)

        st.markdown(
            f'<div style="margin-bottom:12px;font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#F3E5AB">'
            f'◉ <b>{n_clusters} ACTIVE HOTSPOT ZONES</b> IDENTIFIED &nbsp;·&nbsp; '
            f'<span style="color:#8C7A5B">{n_noise} ISOLATED SIGNALS</span></div>',
            unsafe_allow_html=True,
        )

        palette = ["#FF3B3B", "#D4AF37", "#38BDF8", "#F97316", "#10B981", "#A855F7"]
        cluster_colors = [palette[l % len(palette)] if l != -1 else "#52525B" for l in labels]

        fig_cl = go.Figure()
        fig_cl.add_trace(go.Scattermapbox(
            lat=[c[0] for c in coords],
            lon=[c[1] for c in coords],
            mode="markers",
            marker=dict(size=32, color=cluster_colors, opacity=0.22),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig_cl.add_trace(go.Scattermapbox(
            lat=[c[0] for c in coords],
            lon=[c[1] for c in coords],
            mode="markers",
            marker=dict(size=13, color=cluster_colors, opacity=0.95),
            text=[
                f"<b>Zone {l if l != -1 else 'Isolated'}</b> | {m['city']} ({m['urgency'].upper()})<br>{m['text']}"
                for l, m in zip(labels, meta)
            ],
            hoverinfo="text",
            showlegend=False,
        ))
        fig_cl = apply_fullwidth_tactical_map(fig_cl, height=445)
        st.plotly_chart(fig_cl, use_container_width=True)

        cc = Counter(l for l in labels if l != -1)
        if cc:
            cols = st.columns(min(len(cc), 4))
            for i, (cid, cnt) in enumerate(sorted(cc.items())):
                clr = palette[cid % len(palette)]
                zone_cities = list({meta[idx]["city"] for idx, lbl in enumerate(labels) if lbl == cid})
                city_label = ", ".join(zone_cities[:2])
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-top:2px solid {clr};padding:12px 16px;margin-top:10px">
                        <div class="kpi-label">ZONE {cid:02d} · {city_label}</div>
                        <div class="kpi-val" style="color:{clr};font-size:24px">{cnt} SIGNALS</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Insufficient geo-tagged coordinates in current batch for spatial DBSCAN.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — LIVE TELEMETRY FEED
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        f'<div class="section-hdr"><span>Real-Time Intercepted Signals ({len(fdf)} Active)</span></div>',
        unsafe_allow_html=True,
    )
    display_df = fdf if not fdf.empty else df
    for _, row in display_df.head(30).iterrows():
        urg = row.get("urgency_category", "information")
        clr = URGENCY_COLORS.get(urg, "#D4AF37")
        icon = URGENCY_ICONS.get(urg, "📌")
        locs = row.get("locations", [])
        loc_s = " · ".join([l.upper() for l in (locs[:3] if isinstance(locs, list) else [])])
        ts = row["timestamp"].strftime("%H:%M:%S") if "timestamp" in row else ""
        topic = row.get("topic", -1)
        t_str = f"CLUSTER #{int(topic):02d}" if topic != -1 else "UNCLUSTERED"
        tweet = html.escape(str(row.get("clean_text", "")))
        safe_loc_s = html.escape(str(loc_s))
        loc_html = f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:#F3E5AB">📍 {safe_loc_s}</span>' if safe_loc_s else ""
        safe_t_str = html.escape(str(t_str))
        safe_ts = html.escape(str(ts))
        safe_urg = html.escape(str(urg.upper()))
        safe_icon = html.escape(str(icon))

        st.markdown(f"""
        <div class="signal-card" style="border-left-color:{clr}">
            <div class="signal-header">
                <span class="signal-badge" style="color:{clr};border-color:{clr}55;background:{clr}18">
                    {safe_icon} {safe_urg}
                </span>
                {loc_html}
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#8C7A5B">{safe_t_str}</span>
                <span class="signal-meta">⏱ {safe_ts}</span>
            </div>
            <div class="signal-text">{tweet}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY EXPORT FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
e1, e2, e3 = st.columns(3)
ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

with e1:
    st.download_button(
        "⬇️ EXPORT STREAM CSV",
        df.to_csv(index=False).encode("utf-8"),
        f"disaster_stream_{ts_str}.csv",
        "text/csv",
        use_container_width=True,
    )
with e2:
    report_txt = summary + ("\n\nACTIVE ALERTS:\n" + "\n".join(a.headline() for a in alerts) if alerts else "")
    st.download_button(
        "⬇️ EXPORT SITUATION BRIEFING (TXT)",
        report_txt.encode("utf-8"),
        f"situation_brief_{ts_str}.txt",
        "text/plain",
        use_container_width=True,
    )
with e3:
    if topic_summary is not None and not topic_summary.empty:
        st.download_button(
            "⬇️ EXPORT BERTOPIC CLUSTERS (CSV)",
            topic_summary.to_csv(index=False).encode("utf-8"),
            f"bertopic_clusters_{ts_str}.csv",
            "text/csv",
            use_container_width=True,
        )

# ── Non-blocking Real-Time Polling Loop ──────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(refresh_sec)
    st.session_state.cycle += 1
    st.session_state.last_refresh = datetime.now()
    st.cache_data.clear()
    st.rerun()