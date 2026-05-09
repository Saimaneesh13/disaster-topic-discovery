"""
dashboard_command_center.py
Disaster Intelligence Command Center – Cyber Blue Theme
FIXES:
  1. cyber-card::before mask removed (was hiding content inside cards)
  2. cmd-header left:0 + padding-left to avoid sidebar overlap
  3. Plotly charts NOT wrapped in raw HTML divs (Streamlit iframe isolation)
  4. block-container padding corrected
"""

import os, sys, time, random, warnings
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
sys.path.insert(0, os.path.dirname(__file__))
from live_feed    import get_live_tweets
from preprocessor import preprocess
from intelligence import add_urgency_scores, add_geo_tags
from alert_system import check_alerts, get_severity_label
from summariser   import get_summary

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Disaster Intelligence Command Center",
    page_icon="🚨", layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== CYBER BLUE THEME CSS =================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

body { animation: fadeInPage 0.8s ease; }
@keyframes fadeInPage { from { opacity: 0; } to { opacity: 1; } }

html, body, .stApp { background: #06080F !important; color: #CFE3FF; }
.stApp { background: #06080F !important; }

/* FIX 1: Corrected top padding so content clears the fixed header (header ~70px + top-bar ~50px = ~120px) */
.block-container {
    padding: 130px 1.5rem 2rem 1.5rem !important;
    max-width: 100% !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: rgba(8,12,24,0.97) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(77,163,255,0.2) !important;
    width: 240px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 11px 20px; cursor: pointer;
    font-size: 13px; font-weight: 400; color: #7a8bad;
    border-left: 3px solid transparent;
    transition: all 0.2s;
}
.nav-item.active { color: #e0e8ff; background: rgba(77,163,255,0.08); border-left-color: #4DA3FF; }
.nav-item:hover { color: #b0bcda; background: rgba(255,255,255,0.03); }

.sidebar-logo {
    padding: 22px 20px 16px;
    border-bottom: 1px solid rgba(77,163,255,0.15);
    margin-bottom: 8px;
}
.sidebar-logo-text {
    font-family: 'Orbitron', monospace;
    font-size: 11px; font-weight: 500; color: #4DA3FF; letter-spacing: 2px; text-transform: uppercase;
}
.sidebar-sub { font-size: 10px; color: #3a4560; margin-top: 2px; letter-spacing: 1px; }

.alert-critical {
    border-left: 3px solid #FF3B3B;
    background: rgba(255,59,59,0.08);
    border-radius: 6px; padding: 10px 16px; margin: 5px 0; font-size: 13px;
    display: flex; align-items: center; gap: 10px;
    animation: pulseGlow 2s infinite;
}
.alert-high {
    border-left: 3px solid #F97316;
    background: rgba(249,115,22,0.08);
    border-radius: 6px; padding: 10px 16px; margin: 5px 0; font-size: 13px;
    display: flex; align-items: center; gap: 10px;
}
.alert-moderate {
    border-left: 3px solid #FACC15;
    background: rgba(250,204,21,0.08);
    border-radius: 6px; padding: 10px 16px; margin: 5px 0; font-size: 13px;
    display: flex; align-items: center; gap: 10px;
}
.alert-level {
    font-family: 'Orbitron', monospace;
    font-size: 11px; font-weight: 700; letter-spacing: 1px; white-space: nowrap;
}
@keyframes pulseGlow {
    0%   { box-shadow: 0 0 5px rgba(255,59,59,0.4); }
    50%  { box-shadow: 0 0 20px rgba(255,59,59,0.8); }
    100% { box-shadow: 0 0 5px rgba(255,59,59,0.4); }
}

/* FIX 2: Header spans full viewport width; left-padding clears the 240px sidebar */
.cmd-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    backdrop-filter: blur(14px);
    background: rgba(8,12,24,0.95);
    border-bottom: 1px solid rgba(77,163,255,0.15);
    padding: 14px 28px 10px 260px;   /* 260px left indent clears sidebar */
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    min-height: 64px;
}
.cmd-title {
    font-family: 'Orbitron', monospace;
    font-size: 20px; font-weight: 500; color: #E6F0FF; letter-spacing: 1px;
    text-shadow: 0 0 12px rgba(77,163,255,0.6);
}
.cmd-subtitle { font-size: 12px; color: #4a5878; margin-top: 2px; letter-spacing: 0.5px; }

.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(77,163,255,0.1);
    border: 1px solid rgba(77,163,255,0.4);
    border-radius: 20px; padding: 4px 12px;
    font-family: 'Orbitron', monospace; font-size: 10px; font-weight: 700; color: #4DA3FF; letter-spacing: 2px;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #4DA3FF;
    animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }

/* Top control bar — sits just below the fixed header */
.top-bar {
    position: fixed;
    top: 64px;          /* matches cmd-header min-height */
    left: 240px;
    right: 0;
    z-index: 998;
    backdrop-filter: blur(12px);
    background: rgba(6,8,15,0.92);
    border-bottom: 1px solid rgba(77,163,255,0.12);
    padding: 8px 20px;
    display: flex; align-items: center; gap: 12px;
    flex-wrap: wrap;
    min-height: 50px;
}

/* Category pills */
.cat-pill {
    background: rgba(77,163,255,0.08);
    border: 1px solid rgba(77,163,255,0.2);
    color: #a0b4e0;
    font-size: 12px; padding: 4px 14px; border-radius: 20px;
    transition: all 0.2s; white-space: nowrap;
}
.cat-pill.active { background: rgba(77,163,255,0.2); border-color: #4DA3FF; color: white; }
.cat-pill:hover  { background: rgba(77,163,255,0.15); }

/* Metric cards */
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
    padding: 14px 0 0;
}
.metric-card {
    position: relative;
    background: linear-gradient(145deg, rgba(10,18,40,0.9), rgba(5,10,25,0.95));
    border-radius: 14px; padding: 16px; overflow: hidden;
    border: 1px solid rgba(77,163,255,0.25);
    box-shadow: inset 0 0 30px rgba(77,163,255,0.1), 0 0 20px rgba(77,163,255,0.2), 0 10px 40px rgba(0,0,0,0.8);
    backdrop-filter: blur(10px); transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: inset 0 0 40px rgba(77,163,255,0.2), 0 0 30px rgba(77,163,255,0.5), 0 20px 60px rgba(0,0,0,1);
    border-color: rgba(77,163,255,0.5);
}
.metric-num   { font-family: 'Orbitron', monospace; font-size: 28px; font-weight: 700; color: #e8eeff; line-height: 1; }
.metric-label { font-size: 11px; color: #4a5878; margin-top: 4px; letter-spacing: 0.5px; }

/* Signal cards */
.signal-card {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(77,163,255,0.15);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    border-left: 3px solid; transition: all 0.25s ease;
}
.signal-card:hover {
    background: rgba(77,163,255,0.08); transform: translateX(4px);
    box-shadow: 0 0 10px rgba(77,163,255,0.3);
}
.signal-header { display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 500; margin-bottom: 5px; }
.signal-badge {
    font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 1px;
    padding: 2px 6px; border-radius: 3px; border: 1px solid; opacity: 0.9;
}
.signal-text { font-size: 12px; color: #8a96b0; line-height: 1.5; }
.signal-meta { font-size: 10px; color: #3a4560; margin-left: auto; white-space: nowrap; }

/* Topic cards */
.topic-card {
    background: rgba(77,163,255,0.04);
    border: 1px solid rgba(77,163,255,0.12);
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
}
.topic-id { font-family: 'Orbitron', monospace; font-size: 10px; color: #4DA3FF; letter-spacing: 2px; }
.kw-pill {
    display: inline-block;
    background: rgba(77,163,255,0.12);
    border: 1px solid rgba(77,163,255,0.2);
    border-radius: 12px; padding: 2px 8px; margin: 2px 3px; font-size: 11px; color: #a0b4e0;
}

/* Summary */
.summary-box {
    background: rgba(77,163,255,0.03);
    border: 1px solid rgba(77,163,255,0.12);
    border-radius: 10px; padding: 16px 18px;
    font-family: 'Share Tech Mono', monospace; font-size: 12px; line-height: 1.8; color: #8a9ab8;
}

/* Severity badges */
.severity-critical {
    background: rgba(255,59,59,0.15); border: 1px solid rgba(255,59,59,0.3);
    color: #FF3B3B; font-family: 'Orbitron',monospace; font-size: 11px; padding: 4px 12px;
    border-radius: 20px; letter-spacing: 1px;
}
.severity-normal {
    background: rgba(100,200,100,0.1); border: 1px solid rgba(100,200,100,0.2);
    color: #64c864; font-family: 'Orbitron',monospace; font-size: 11px; padding: 4px 12px;
    border-radius: 20px; letter-spacing: 1px;
}
.severity-moderate {
    background: rgba(250,204,21,0.1); border: 1px solid rgba(250,204,21,0.25);
    color: #FACC15; font-family: 'Orbitron',monospace; font-size: 11px; padding: 4px 12px;
    border-radius: 20px; letter-spacing: 1px;
}

.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 11px; font-weight: 500; color: #4a5878; letter-spacing: 2px;
    text-transform: uppercase; padding: 14px 0 8px;
    border-bottom: 1px solid rgba(77,163,255,0.07); margin-bottom: 8px;
}

.center-glow {
    position: fixed; top: 45%; left: 55%;
    width: 300px; height: 300px; transform: translate(-50%,-50%);
    background: radial-gradient(circle, rgba(77,163,255,0.2), transparent 70%);
    filter: blur(40px); z-index: 0; pointer-events: none;
}

/* Streamlit buttons */
.stButton > button {
    background: rgba(77,163,255,0.12) !important;
    border: 1px solid rgba(77,163,255,0.25) !important;
    color: #a0b4e0 !important; border-radius: 6px !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(77,163,255,0.2) !important; border-color: rgba(77,163,255,0.4) !important;
    box-shadow: 0 0 10px rgba(77,163,255,0.5);
}
.stButton > button:active { transform: scale(0.96); }

/* FIX 3: cyber-card — removed ::before mask that was hiding content.
   Replaced with a simple glowing border + subtle top highlight. */
.cyber-card {
    position: relative;
    background: linear-gradient(145deg, rgba(10,18,40,0.92), rgba(6,10,25,0.97));
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(77,163,255,0.28);
    box-shadow:
        0 0 0 1px rgba(77,163,255,0.06),
        0 0 20px rgba(77,163,255,0.2),
        inset 0 0 20px rgba(77,163,255,0.04);
    backdrop-filter: blur(10px);
    overflow: visible;   /* must NOT be hidden — lets Plotly tooltips show */
    animation: cardGlow 4s ease-in-out infinite;
}
@keyframes cardGlow {
    0%,100% { box-shadow: 0 0 16px rgba(77,163,255,0.18), inset 0 0 14px rgba(77,163,255,0.04); }
    50%      { box-shadow: 0 0 30px rgba(77,163,255,0.35), inset 0 0 24px rgba(77,163,255,0.08); }
}
/* Subtle top highlight bar instead of mask trick */
.cyber-card::after {
    content: "";
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    border-radius: 14px 14px 0 0;
    background: linear-gradient(90deg, transparent, rgba(77,163,255,0.6), transparent);
    animation: borderFlow 4s linear infinite;
    pointer-events: none;
}
@keyframes borderFlow { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

/* Make Streamlit metric labels/values readable on dark bg */
[data-testid="stMetric"] label { color: #4a6080 !important; font-size: 11px !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #c8d8f0 !important; font-size: 22px !important; }

/* Style Streamlit tabs to match theme */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,12,24,0.8);
    border-bottom: 1px solid rgba(77,163,255,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a5878;
    font-family: 'Orbitron', monospace;
    font-size: 11px; letter-spacing: 1px;
    border: none; padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: rgba(77,163,255,0.1) !important;
    color: #4DA3FF !important;
    border-bottom: 2px solid #4DA3FF !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
URGENCY_COLORS = {
    "rescue": "#FF3B3B",
    "medical": "#F97316",
    "food_water": "#FACC15",
    "infrastructure": "#4DA3FF",
    "information": "#22C55E",
}
URGENCY_ICONS = {
    "rescue":"🆘","medical":"🏥","food_water":"🍽️",
    "infrastructure":"🏗️","information":"ℹ️",
}
CITY_COORDS = {
    "chennai":(13.08,80.27),"assam":(26.14,91.74),"patna":(25.59,85.13),
    "bihar":(25.09,85.31),"uttarakhand":(30.06,79.01),"mumbai":(19.07,72.87),
    "odisha":(20.29,85.82),"gujarat":(22.25,71.19),"srinagar":(34.08,74.79),
    "silchar":(24.82,92.80),"manipur":(24.66,93.90),"kedarnath":(30.73,79.06),
    "guwahati":(26.14,91.74),"bhubaneswar":(20.29,85.82),"kolkata":(22.57,88.36),
    "delhi":(28.61,77.23),"bangalore":(12.97,77.59),"hyderabad":(17.38,78.48),
    "lucknow":(26.84,80.94),"nagpur":(21.14,79.08),"himachal":(31.10,77.17),
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("cycle", 0), ("last_refresh", datetime.now()), ("auto_refresh", False),
    ("active_tab", "overview"), ("ml_loaded", False), ("embedder", None),
    ("active_cats", list(URGENCY_COLORS.keys())),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=12)
def run_pipeline(cycle: int, n: int = 30, use_ml: bool = False):
    df = get_live_tweets(n=n)
    df = preprocess(df, text_col="text", filter_lang=False, dedup=True)
    if len(df) < 5:
        return None, None, None, None
    df = add_urgency_scores(df)
    df = add_geo_tags(df)
    now = datetime.now()
    df["timestamp"] = [now - timedelta(minutes=random.randint(0, 60)) for _ in range(len(df))]
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    df["topic"] = -1
    df["topic_confidence"] = 0.0

    topic_summary = None
    if use_ml and st.session_state.embedder is not None:
        try:
            from topic_modeller import fit_bertopic, get_topic_summary
            embedder = st.session_state.embedder
            texts = df["clean_text"].tolist()
            embeddings = embedder.embed(texts, cache_key=None, show_progress=False)
            model, topics, probs, _ = fit_bertopic(texts, embeddings)
            df["topic"] = topics
            df["topic_confidence"] = probs.max(axis=1) if probs is not None else [0.5]*len(df)
            topic_summary = get_topic_summary(model)
        except Exception:
            pass

    alerts  = check_alerts(df)
    summary = get_summary(df, alerts)
    return df, alerts, summary, topic_summary

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.auto_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).seconds
    if elapsed >= 10:
        st.session_state.cycle += 1
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">DisasterAI</div>
        <div class="sidebar-sub">Command Center v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    tabs_nav = [
        ("🏠", "overview",  "Overview"),
        ("🔍", "topics",    "Topic Discovery"),
        ("📍", "hotspots",  "Hotspot Clusters"),
        ("📡", "feed",      "Live Feed"),
    ]
    for icon, key, label in tabs_nav:
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.active_tab = key
            st.rerun()
            st.stop()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="padding:8px 4px 4px;font-size:11px;color:#4a5878">Controls</div>',
                unsafe_allow_html=True)

    n_tw = st.slider("Tweets/batch", 15, 60, 30, 5, format="%d", label_visibility="collapsed")
    auto = st.toggle("Auto-Refresh  10s", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    if st.button("⟳  Refresh Now", use_container_width=True):
        st.session_state.cycle += 1
        st.cache_data.clear()
        st.rerun()
        st.stop()

    st.divider()
    use_llm = st.checkbox("Use Gemini API", value=False)
    api_key_input = ""
    if use_llm:
        api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    if not st.session_state.ml_loaded:
        if st.button("⚙️ Load ML Models", use_container_width=True):
            success = False
            with st.spinner("Loading Sentence-BERT…"):
                try:
                    from embedder import TweetEmbedder
                    st.session_state.embedder = TweetEmbedder()
                    st.session_state.ml_loaded = True
                    success = True
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
            if success:
                st.cache_data.clear()
                st.rerun()
    else:
        st.success("✅ ML Active")

    st.markdown(f"""
    <div style="padding:10px 4px 0;font-size:10px;color:#2a3450;font-family:'Share Tech Mono',monospace">
    ◉ Categories: {len(st.session_state.active_cats)}<br>
    ⟳ Cycle: {st.session_state.cycle:05d}<br>
    ⏱ {st.session_state.last_refresh.strftime('%H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df, alerts, summary, topic_summary = run_pipeline(
    st.session_state.cycle, n=n_tw, use_ml=st.session_state.ml_loaded)

if df is None:
    st.error("No data available. Click Refresh.")
    st.stop()

if use_llm:
    api_key = api_key_input or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.warning("⚠️ No GEMINI_API_KEY found. Using rule-based summary.")
    summary = get_summary(df, alerts, use_llm=True, api_key=api_key)

fdf = df[df["urgency_category"].isin(st.session_state.active_cats)]
sev_label, sev_color, _ = get_severity_label(alerts)

# ══════════════════════════════════════════════════════════════════════════════
# FIXED HEADER
# ══════════════════════════════════════════════════════════════════════════════
live_html = (
    '<div class="live-badge"><div class="live-dot"></div>LIVE</div>'
    if st.session_state.auto_refresh else
    '<div style="font-size:11px;color:#3a4560;font-family:Orbitron,monospace;letter-spacing:2px">PAUSED</div>'
)
sev_cls = {"SEVERE": "severity-critical", "MODERATE": "severity-moderate"}.get(sev_label, "severity-normal")

st.markdown(f"""
<div class="cmd-header">
    <div style="font-size:26px">🚨</div>
    <div>
        <div class="cmd-title">Disaster Intelligence Command Center</div>
        <div class="cmd-subtitle">Real-time AI monitoring · Topic discovery · Risk prediction</div>
    </div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:12px">
        <div class="{sev_cls}">{sev_label}</div>
        {live_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TOP CONTROL BAR  (keyword + category pills)
# NOTE: this is rendered as a Streamlit-native row, then visually styled via CSS.
# We do NOT use a raw HTML div wrapper around st.columns because Streamlit
# cannot inject widgets inside raw HTML.
# ══════════════════════════════════════════════════════════════════════════════
ctrl_left, ctrl_right = st.columns([3, 5])
with ctrl_left:
    keyword = st.text_input("🔍", placeholder="Keyword filter: flood, rescue…",
                             label_visibility="collapsed")
with ctrl_right:
    pill_cols = st.columns(5)
    categories = ["rescue", "medical", "food_water", "infrastructure", "information"]
    for i, cat in enumerate(categories):
        active_pill = cat in st.session_state.active_cats
        btn_label = f"{'✓ ' if active_pill else ''}{cat}"
        if pill_cols[i].button(btn_label, key=f"pill_{cat}", help=f"Toggle {cat}",
                               use_container_width=True):
            if active_pill:
                st.session_state.active_cats.remove(cat)
            else:
                st.session_state.active_cats.append(cat)
            st.rerun()

if keyword:
    fdf = fdf[fdf["clean_text"].str.contains(keyword, case=False, na=False)]

# ── Center glow ───────────────────────────────────────────────────────────────
st.markdown('<div class="center-glow"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ALERT BANNERS
# ══════════════════════════════════════════════════════════════════════════════
if alerts:
    ALERT_CLR = {"CRITICAL": "#FF3B3B", "HIGH": "#F97316", "MODERATE": "#FACC15"}
    for a in alerts[:3]:
        clr     = ALERT_CLR.get(a.level, "#FACC15")
        loc_str = ", ".join(a.locations[:3]) if a.locations else "multiple areas"
        st.markdown(
            f'<div class="alert-{a.level.lower()}">'
            f'<span style="font-size:16px">{a.icon}</span>'
            f'<span class="alert-level" style="color:{clr}">{a.level}</span>'
            f'<span style="color:#9aa8c0"> — {a.message} — {loc_str} ({a.tweet_count} signals)</span>'
            f'</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
n_rescue  = len(df[df["urgency_category"] == "rescue"])
n_medical = len(df[df["urgency_category"] == "medical"])
n_geo     = int(df["has_location"].sum()) if "has_location" in df else 0
n_topics  = int(df["topic"].nunique())    if "topic" in df else 0

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-num">{len(df)}</div>
        <div class="metric-label">Total Signals</div>
    </div>
    <div class="metric-card">
        <div class="metric-num" style="color:#FF3B3B">{n_rescue}</div>
        <div class="metric-label">🆘 Rescue</div>
    </div>
    <div class="metric-card">
        <div class="metric-num" style="color:#F97316">{n_medical}</div>
        <div class="metric-label">🏥 Medical</div>
    </div>
    <div class="metric-card">
        <div class="metric-num" style="color:#4DA3FF">{n_geo}</div>
        <div class="metric-label">📍 Geo-Tagged</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🎯 Topic Discovery", "📍 Hotspot Clusters", "📡 Live Feed"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    # ── Row 1: Map  +  Real-time Metrics ─────────────────────────────────────
    map_col, metrics_col = st.columns([2.2, 1])

    with map_col:
        st.markdown('<div class="section-title">Geo Intelligence Map</div>', unsafe_allow_html=True)
        map_rows = []
        for _, row in df.iterrows():
            locs = row.get("locations", [])
            if not isinstance(locs, list):
                continue
            for loc in locs:
                c = CITY_COORDS.get(loc.lower())
                if c:
                    urg = row.get("urgency_category", "information")
                    map_rows.append({
                        "lat":    c[0] + random.uniform(-.25, .25),
                        "lon":    c[1] + random.uniform(-.25, .25),
                        "urgency": urg,
                        "city":   loc.title(),
                        "text":   str(row.get("clean_text", ""))[:80],
                    })
        if map_rows:
            mdf         = pd.DataFrame(map_rows)
            city_counts = mdf.groupby(["city","lat","lon","urgency"]).size().reset_index(name="count")
            fig_map     = go.Figure()

            # glow halos
            for urg, clr in URGENCY_COLORS.items():
                sub = city_counts[city_counts["urgency"] == urg]
                if sub.empty: continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=sub["lat"], lon=sub["lon"], mode="markers",
                    marker=dict(size=sub["count"]*6+20, color=clr, opacity=0.08),
                    hoverinfo="skip", showlegend=False,
                ))
            # main dots
            for urg, clr in URGENCY_COLORS.items():
                sub = city_counts[city_counts["urgency"] == urg]
                if sub.empty: continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=sub["lat"], lon=sub["lon"],
                    mode="markers+text",
                    marker=dict(size=sub["count"]*3+14, color=clr, opacity=0.85, sizemin=12),
                    text=sub["count"].astype(str),
                    textfont=dict(size=10, color="white", family="Orbitron"),
                    textposition="middle center",
                    name=urg.upper(),
                    hovertext=sub.apply(
                        lambda r: f"{r['city']} | {r['urgency'].upper()}<br>{r['count']} signals", axis=1),
                    hoverinfo="text",
                ))
            fig_map.update_layout(
                mapbox=dict(style="carto-darkmatter", center=dict(lat=22, lon=80), zoom=3.8),
                margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    bgcolor="rgba(7,8,15,0.85)", bordercolor="rgba(77,163,255,0.2)",
                    borderwidth=1, font=dict(size=10, color="#7a8bad", family="Inter"),
                    x=0.01, y=0.99),
                showlegend=True,
            )

            core_color = "#4DA3FF"
            if sev_label == "SEVERE":   core_color = "#FF3B3B"
            elif sev_label == "MODERATE": core_color = "#FACC15"

            # Map is rendered inside components.html so it has its own document;
            # the cyber-card wrapper is inside the HTML, which is correct here.
            components.html(f"""
            <div style="
                position:relative; width:100%; height:460px; border-radius:16px; overflow:hidden;
                background:linear-gradient(145deg,rgba(10,18,40,0.9),rgba(5,10,25,0.95));
                border:1px solid rgba(77,163,255,0.25);
                box-shadow:inset 0 0 40px rgba(77,163,255,0.15),0 0 25px rgba(77,163,255,0.25),0 20px 60px rgba(0,0,0,0.9);
            ">
                <div class="map-glow"></div>
                <div style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:0;">
                    {fig_map.to_html(include_plotlyjs='cdn', full_html=False)}
                </div>
                <!-- Radar rings overlay -->
                <div style="position:absolute;bottom:50px;right:40px;width:160px;height:160px;z-index:3;pointer-events:none;">
                    <div style="position:absolute;width:100%;height:100%;border-radius:50%;
                        background:conic-gradient(from 0deg,rgba(77,163,255,0.5),rgba(77,163,255,0.08),transparent 60%);
                        animation:spin 4s linear infinite;box-shadow:0 0 20px rgba(77,163,255,0.5);">
                    </div>
                    <div style="position:absolute;width:100%;height:100%;">
                        <div style="width:70px;height:70px;border:1px solid rgba(77,163,255,0.3);border-radius:50%;
                            position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);animation:ringPulse 3s infinite;"></div>
                        <div style="width:120px;height:120px;border:1px solid rgba(77,163,255,0.2);border-radius:50%;
                            position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);animation:ringPulse 3s infinite 0.5s;"></div>
                        <div style="width:160px;height:160px;border:1px solid rgba(77,163,255,0.25);border-radius:50%;
                            position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);animation:ringPulse 3s infinite 1s;"></div>
                    </div>
                    <div style="position:absolute;top:50%;left:50%;width:10px;height:10px;
                        background:{core_color};border-radius:50%;transform:translate(-50%,-50%);
                        box-shadow:0 0 12px {core_color},0 0 24px {core_color};">
                        <div style="position:absolute;width:10px;height:10px;border-radius:50%;
                            border:2px solid {core_color};animation:pulseWave 1.2s infinite;"></div>
                    </div>
                </div>
                <div style="position:absolute;bottom:10px;right:40px;
                    font-family:'Orbitron',monospace;font-size:10px;letter-spacing:2px;
                    color:{core_color};background:rgba(77,163,255,0.12);border:1px solid {core_color};
                    padding:4px 10px;border-radius:18px;
                    box-shadow:0 0 12px {core_color},0 0 24px {core_color};">
                    ◉ AI CORE ACTIVE
                </div>
            </div>
            <style>
            @keyframes spin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
            @keyframes ringPulse {{
                0%  {{opacity:0.2;transform:translate(-50%,-50%) scale(1)}}
                50% {{opacity:0.6;transform:translate(-50%,-50%) scale(1.05)}}
                100%{{opacity:0.2;transform:translate(-50%,-50%) scale(1)}}
            }}
            @keyframes pulseWave {{
                0%  {{transform:scale(1);opacity:0.8}}
                100%{{transform:scale(5);opacity:0}}
            }}
            .map-glow {{
                position:absolute;top:0;left:-100%;width:50%;height:100%;
                background:linear-gradient(90deg,transparent,rgba(77,163,255,0.2),transparent);
                transform:skewX(-20deg);pointer-events:none;animation:sweep 6s infinite;z-index:3;
            }}
            @keyframes sweep {{0%{{left:-100%}} 100%{{left:150%}}}}
            </style>
            """, height=465)
        else:
            st.info("No geo-tagged data in this batch.")

    with metrics_col:
        st.markdown('<div class="section-title">Real‑time Metrics</div>', unsafe_allow_html=True)
        # FIX: Use st.container with CSS class applied via markdown OUTSIDE the container.
        # Streamlit metrics render correctly when NOT wrapped in raw HTML divs.
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        high_urg = (df["urgency_category"] == "rescue").mean() * 100 if len(df) > 0 else 0
        st.metric("🔴 High Priority",   f"{high_urg:.1f}%")
        st.metric("📍 Locations",
                  df["locations"].apply(lambda x: len(x) if isinstance(x, list) else 0).sum())
        top_urg = df["urgency_category"].value_counts().idxmax() if not df.empty else "—"
        st.metric("🎯 Top Urgency",     top_urg.upper())
        st.metric("🤖 Avg Confidence",  "—")
        st.metric("⏱ Data Points",      len(df))
        st.metric("📊 Alerts Active",   len(alerts))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=False)

    # ── Row 2: Signal Trend  +  System Overview & Urgency Pie ────────────────
    trend_col, sys_col = st.columns([2.2, 1])

    with trend_col:
        st.markdown('<div class="section-title">Signal Trend — Last 60 min</div>', unsafe_allow_html=True)
        df["time_bin"] = df["timestamp"].dt.floor("10min")
        trend = df.groupby(["time_bin","urgency_category"]).size().reset_index(name="count")
        fig_t = px.area(
            trend, x="time_bin", y="count", color="urgency_category",
            color_discrete_map=URGENCY_COLORS, height=220,
        )
        fig_t.update_traces(line_width=1.5)
        fig_t.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(77,163,255,0.03)",
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(gridcolor="rgba(77,163,255,0.07)", showgrid=True, zeroline=False,
                       tickfont=dict(size=10, color="#4a5878")),
            yaxis=dict(gridcolor="rgba(77,163,255,0.07)", showgrid=True, zeroline=False,
                       tickfont=dict(size=10, color="#4a5878")),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#7a8bad")),
            showlegend=True,
        )
        # FIX: plotly_chart is called DIRECTLY (not inside raw HTML div wrapper)
        # The cyber-card wrapper uses st.markdown open/close tags which Streamlit
        # injects as sibling HTML nodes — the chart floats between them visually.
        st.markdown('<div class="cyber-card" style="padding:12px 8px">', unsafe_allow_html=True)
        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with sys_col:
        st.markdown('<div class="cyber-card" style="padding:16px">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Signals", len(df))
            st.metric("Rescue",  n_rescue)
        with m2:
            st.metric("Medical", n_medical)
            st.metric("Topics",  n_topics if n_topics > 1 else "—")

        st.markdown('<div class="section-title" style="margin-top:8px">Urgency Distribution</div>',
                    unsafe_allow_html=True)
        urgency_counts = df["urgency_category"].value_counts().reset_index()
        urgency_counts.columns = ["category", "count"]
        fig_d = px.pie(urgency_counts, values="count", names="category",
                       color="category", color_discrete_map=URGENCY_COLORS, hole=0.4)
        fig_d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7a8bad", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(l=0,r=0,t=0,b=0),
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: AI Summary ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">AI Situation Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    # ── Row 4: Advanced Analytics ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Advanced Analytics</div>', unsafe_allow_html=True)
    ana1, ana2, ana3 = st.columns(3)

    with ana1:
        st.markdown("**URGENCY TIMELINE**")
        urg_counts = df["urgency_category"].value_counts()
        fig_urg = px.bar(
            x=urg_counts.index, y=urg_counts.values,
            color=urg_counts.index, color_discrete_map=URGENCY_COLORS, height=200,
        )
        fig_urg.update_layout(
            margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_urg, use_container_width=True)

    with ana2:
        st.markdown("**TOP LOCATIONS**")
        loc_series = df["locations"].explode().dropna()
        top_locs   = loc_series.value_counts().head(5)
        for loc, cnt in top_locs.items():
            st.markdown(
                f'<div style="padding:4px 0;font-size:12px;color:#8a9ab8">'
                f'<span style="color:#4DA3FF">◉</span> {loc.title()} '
                f'<span style="color:#4a5878;float:right">{cnt}</span></div>',
                unsafe_allow_html=True)

    with ana3:
        st.markdown("**Key Insights**")
        st.metric("SIGNAL VELOCITY",    f"{len(df)}/min")
        dom_cat = df["urgency_category"].value_counts().idxmax() if not df.empty else "—"
        st.metric("DOMINANT CATEGORY",  dom_cat.upper())
        geo_pct = (df["has_location"].sum() / len(df) * 100) if len(df) > 0 else 0
        st.metric("GEO COVERAGE",       f"{geo_pct:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — TOPIC DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">BERTopic Discovery — Sentence-BERT + UMAP + HDBSCAN</div>',
                unsafe_allow_html=True)
    if not st.session_state.ml_loaded:
        st.markdown("""
        <div style="text-align:center;padding:40px;color:#3a4560">
            <div style="font-family:Orbitron,monospace;font-size:14px;color:#4DA3FF;margin-bottom:10px">
                ML MODELS NOT LOADED
            </div>
            <div style="font-size:13px">
                Click "⚙️ Load ML Models" in the sidebar to activate Sentence-BERT topic discovery
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif topic_summary is not None and not topic_summary.empty:
        for _, row in topic_summary.iterrows():
            kws   = str(row.get("keywords", "")).split(", ")
            pills = "".join([f'<span class="kw-pill">{k}</span>' for k in kws[:6]])
            st.markdown(f"""
            <div class="topic-card">
                <div style="display:flex;align-items:center;gap:10px">
                    <span class="topic-id">TOPIC {int(row['topic_id']):03d}</span>
                    <span style="font-size:11px;color:#4a5878">{row.get('tweet_count',0)} signals</span>
                </div>
                <div style="margin-top:6px">{pills}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:20px;color:#4a5878;font-size:13px">
        No topics discovered in current batch. Load ML models and increase tweet count.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOTSPOT CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">DBSCAN Spatial Clustering</div>', unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_clusters = st.button("🔍 Detect Hotspots", type="primary", use_container_width=True)

    if run_clusters:
        coords, meta = [], []
        for _, row in df.iterrows():
            locs = row.get("locations", [])
            if not isinstance(locs, list): continue
            for loc in locs:
                c = CITY_COORDS.get(loc.lower())
                if c:
                    coords.append([c[0]+random.uniform(-.1,.1), c[1]+random.uniform(-.1,.1)])
                    meta.append({"city": loc, "urgency": row.get("urgency_category",""),
                                 "text": str(row.get("clean_text",""))[:70]})
        if len(coords) >= 2:
            arr    = np.array(coords)
            labels = DBSCAN(eps=0.3, min_samples=2).fit_predict(
                StandardScaler().fit_transform(arr))
            n_clusters = len(set(labels) - {-1})
            n_noise    = list(labels).count(-1)
            st.markdown(f"""
            <div style="padding:8px 0 12px;font-family:Orbitron,monospace;font-size:12px;color:#4DA3FF">
            ◉ {n_clusters} HOTSPOT CLUSTERS DETECTED &nbsp;·&nbsp;
            <span style="color:#4a5878">{n_noise} isolated signals</span>
            </div>""", unsafe_allow_html=True)

            colors_list    = ["#FF3B3B","#F97316","#FACC15","#4DA3FF","#9B59B6","#00CED1"]
            cluster_colors = [colors_list[l%len(colors_list)] if l!=-1 else "#2a3450" for l in labels]

            fig_cl = go.Figure()
            fig_cl.add_trace(go.Scattermapbox(
                lat=[c[0] for c in coords], lon=[c[1] for c in coords],
                mode="markers",
                marker=dict(size=30, color=cluster_colors, opacity=0.08),
                hoverinfo="skip", showlegend=False,
            ))
            fig_cl.add_trace(go.Scattermapbox(
                lat=[c[0] for c in coords], lon=[c[1] for c in coords],
                mode="markers",
                marker=dict(size=14, color=cluster_colors, opacity=0.9),
                text=[f"Cluster {l} | {m['city']} | {m['urgency']}<br>{m['text']}"
                      for l, m in zip(labels, meta)],
                hoverinfo="text", showlegend=False,
            ))
            fig_cl.update_layout(
                mapbox=dict(style="carto-darkmatter", center=dict(lat=22, lon=80), zoom=3.8),
                margin=dict(l=0,r=0,t=0,b=0), height=440,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cl, use_container_width=True)

            cc = Counter(l for l in labels if l != -1)
            if cc:
                st.markdown('<div class="section-title">Cluster Breakdown</div>', unsafe_allow_html=True)
                cols = st.columns(min(len(cc), 4))
                for i, (cid, cnt) in enumerate(sorted(cc.items())):
                    with cols[i % len(cols)]:
                        clr = colors_list[cid % len(colors_list)]
                        st.markdown(f"""
                        <div class="metric-card" style="border-top:2px solid {clr}">
                            <div class="metric-num" style="color:{clr};font-size:22px">{cnt}</div>
                            <div class="metric-label">Cluster {cid}</div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.warning("Not enough geo-tagged signals. Try refreshing.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — LIVE FEED
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Real-time Signal Feed</div>', unsafe_allow_html=True)
    if keyword:
        st.markdown(
            f'<div style="font-size:11px;color:#4a5878;padding:4px 0 8px">'
            f'Filtered: "{keyword}" — {len(fdf)} results</div>',
            unsafe_allow_html=True)
    display = fdf if not fdf.empty else df
    for _, row in display.head(25).iterrows():
        urg   = row.get("urgency_category", "information")
        clr   = URGENCY_COLORS.get(urg, "#4DA3FF")
        icon  = URGENCY_ICONS.get(urg, "📌")
        locs  = row.get("locations", [])
        loc_s = " · ".join([l.title() for l in (locs[:2] if isinstance(locs, list) else [])])
        ts    = row["timestamp"].strftime("%H:%M") if "timestamp" in row else ""
        topic = row.get("topic", -1)
        t_str = f"topic {topic}" if topic != -1 else "unassigned"
        tweet = str(row.get("clean_text", ""))[:130]
        loc_html = f"<span style='font-size:11px;color:#4a5878'>📍 {loc_s}</span> " if loc_s else ""
        st.markdown(f"""
        <div class="signal-card" style="border-left-color:{clr}">
            <div class="signal-header">
                <span class="signal-badge" style="color:{clr};border-color:{clr}40;background:{clr}12">{icon} {urg.upper()}</span>
                {loc_html}<span style="font-size:10px;color:#3a4560;font-family:'Share Tech Mono',monospace">{t_str}</span>
                <span class="signal-meta">{ts}</span>
            </div>
            <div class="signal-text">{tweet}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER / EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
e1, e2, e3 = st.columns(3)
ts_str = datetime.now().strftime("%Y%m%d_%H%M")

with e1:
    st.download_button(
        "⬇️ Export Signals CSV",
        df.to_csv(index=False).encode(),
        f"signals_{ts_str}.csv", "text/csv",
        use_container_width=True)
with e2:
    report = summary + (
        "\n\nALERTS:\n" + "\n".join(a.headline() for a in alerts) if alerts else "")
    st.download_button(
        "⬇️ Export Report TXT",
        report.encode(),
        f"report_{ts_str}.txt", "text/plain",
        use_container_width=True)
with e3:
    if topic_summary is not None:
        st.download_button(
            "⬇️ Export Topics CSV",
            topic_summary.to_csv(index=False).encode(),
            f"topics_{ts_str}.csv", "text/csv",
            use_container_width=True)

st.markdown(f"""
<div style="padding:8px 0 16px;font-family:'Share Tech Mono',monospace;
     font-size:10px;color:#2a3450;letter-spacing:1px">
◉ DISASTER INTELLIGENCE COMMAND CENTER &nbsp;·&nbsp;
SENTENCE-BERT + BERTOPIC &nbsp;·&nbsp;
CYCLE {st.session_state.cycle:05d} &nbsp;·&nbsp;
{len(df)} SIGNALS ACTIVE
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh loop (must be last) ─────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(10)
    st.session_state.cycle += 1
    st.session_state.last_refresh = datetime.now()
    st.cache_data.clear()
    st.rerun()
    st.stop()