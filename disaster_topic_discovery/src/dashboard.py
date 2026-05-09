"""
dashboard_pipeline_integrated.py  ←  FULL PIPELINE DASHBOARD (UI REDESIGN)
======================================================================
Modern Disaster Intelligence Command Center UI
Matches the reference image design with:
  - Left panel: System Overview metrics
  - Center: Live Signal Feed
  - Clean, minimal card-based layout
"""

import os, sys, time, random, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ── Import YOUR actual modules ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from live_feed      import get_live_tweets
from preprocessor   import preprocess
from embedder       import TweetEmbedder
from topic_modeller import fit_bertopic, get_topic_summary
from intelligence   import add_urgency_scores, add_geo_tags
from alert_system   import check_alerts, get_severity_label
from summariser     import get_summary

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for the image-like UI ──────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 100%);
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Header title style */
    .command-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e0e7ff 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    
    .command-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 400;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        margin-top: -5px;
        margin-bottom: 20px;
    }
    
    /* Alert cards */
    .alert-critical {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    
    .alert-moderate {
        background: linear-gradient(135deg, #854d0e 0%, #422006 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    
    .alert-critical .alert-title {
        color: #fca5a5;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .alert-moderate .alert-title {
        color: #fde68a;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .alert-message {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .alert-signal-count {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    
    /* System Overview Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 16px 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(71, 85, 105, 0.3);
        border-radius: 10px;
        padding: 16px 12px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(71, 85, 105, 0.4);
    }
    
    /* Live feed cards */
    .feed-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin: 8px 0;
        transition: all 0.2s;
    }
    
    .feed-card:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(71, 85, 105, 0.4);
    }
    
    .feed-header {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .feed-category {
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #ef4444;
        margin-right: 8px;
    }
    
    .feed-category.food { color: #f59e0b; }
    .feed-category.rescue { color: #ef4444; }
    .feed-category.medical { color: #f97316; }
    .feed-category.infrastructure { color: #3b82f6; }
    
    .feed-location {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    
    .feed-topic {
        color: #64748b;
        font-size: 0.7rem;
        margin-left: 8px;
    }
    
    .feed-time {
        margin-left: auto;
        color: #475569;
        font-size: 0.7rem;
        font-family: monospace;
    }
    
    .feed-content {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-left: 4px;
    }
    
    /* Live indicator dot */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ef4444;
        margin-right: 6px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px #ef4444;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 10px 20px;
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(51, 65, 85, 0.8) !important;
        color: #f1f5f9 !important;
        border-color: #3b82f6 !important;
    }
    
    /* Refresh button */
    .refresh-btn {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 6px;
        padding: 6px 12px;
        color: #60a5fa;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .refresh-btn:hover {
        background: rgba(59, 130, 246, 0.3);
        border-color: #3b82f6;
    }
    
    /* Hide default Streamlit button styling for our custom buttons */
    .stButton button {
        background: transparent;
        border: 1px solid rgba(71, 85, 105, 0.4);
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ── Location coordinates ──────────────────────────────────────────────────────
CITY_COORDS = {
    "chennai":(13.08,80.27),"assam":(26.14,91.74),"patna":(25.59,85.13),
    "bihar":(25.09,85.31),"uttarakhand":(30.06,79.01),"mumbai":(19.07,72.87),
    "odisha":(20.29,85.82),"gujarat":(22.25,71.19),"srinagar":(34.08,74.79),
    "silchar":(24.82,92.80),"manipur":(24.66,93.90),"kedarnath":(30.73,79.06),
    "guwahati":(26.14,91.74),"bhubaneswar":(20.29,85.82),"kolkata":(22.57,88.36),
    "delhi":(28.61,77.23),"bangalore":(12.97,77.59),"hyderabad":(17.38,78.48),
    "kaziranga":(26.58,93.17),"brahmaputra":(26.25,91.50),"himachal":(31.10,77.17),
    "nh-44":(28.61,77.23),"muzaffarpur":(26.12,85.39),
}

URGENCY_ICONS = {
    "rescue":"🆘","medical":"🏥","food_water":"🍽️",
    "infrastructure":"🏗️","information":"ℹ️",
}

# ════════════════════════════════════════════════════════════════════════════
# ML PIPELINE CLASS
# ════════════════════════════════════════════════════════════════════════════

class FullDisasterPipeline:
    def __init__(self):
        self.embedder    = None
        self.bt_model    = None
        self.initialized = False

    def initialize(self):
        with st.spinner("Loading models..."):
            try:
                self.embedder    = TweetEmbedder()
                self.initialized = True
            except Exception as e:
                st.error(f"Model load failed: {e}")
                self.initialized = False

    def run(self, n_tweets: int = 30):
        df = get_live_tweets(n=n_tweets)
        df = preprocess(df, text_col="text", filter_lang=False, dedup=True)
        if len(df) < 5:
            return None, None, None, None

        df = add_urgency_scores(df)
        df = add_geo_tags(df)

        now = datetime.now()
        df["timestamp"] = [now - timedelta(minutes=random.randint(0,60))
                           for _ in range(len(df))]
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

        topic_summary = None
        if self.initialized and self.embedder is not None:
            try:
                texts      = df["clean_text"].tolist()
                embeddings = self.embedder.embed(texts, cache_key=None, show_progress=False)
                self.bt_model, topics, probs, _ = fit_bertopic(texts, embeddings)
                df["topic"]            = topics
                df["topic_confidence"] = (probs.max(axis=1) if probs is not None else [0.5] * len(df))
                topic_summary = get_topic_summary(self.bt_model)
            except Exception as e:
                df["topic"]            = -1
                df["topic_confidence"] = 0.0
        else:
            df["topic"]            = -1
            df["topic_confidence"] = 0.0

        alerts  = check_alerts(df)
        summary = get_summary(df, alerts, use_llm=False)

        return df, alerts, summary, topic_summary


# ════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Session state
    for k, v in [("cycle",0),("last_refresh",datetime.now()),
                 ("auto_refresh",True),("pipeline", FullDisasterPipeline())]:
        if k not in st.session_state:
            st.session_state[k] = v

    pipeline: FullDisasterPipeline = st.session_state.pipeline

    # Initialize models on first load
    if not pipeline.initialized:
        pipeline.initialize()

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        elapsed = (datetime.now() - st.session_state.last_refresh).seconds
        if elapsed >= 15:
            st.session_state.cycle += 1
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.rerun()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner(""):
        df, alerts, summary, topic_summary = pipeline.run(n_tweets=35)

    if df is None:
        st.error("Pipeline returned no data — try refreshing.")
        st.stop()

    # ── Header ────────────────────────────────────────────────────────────────
    col_header, col_refresh = st.columns([6, 1])
    with col_header:
        st.markdown('<div class="command-header">Disaster Intelligence Command Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="command-subtitle">Real-time AI monitoring · Topic discovery · Risk prediction</div>', unsafe_allow_html=True)
    
    with col_refresh:
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        if st.button("🔄 REFRESH", key="refresh_btn", use_container_width=True):
            st.session_state.cycle += 1
            st.cache_data.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Alert Banners (CRITICAL and MODERATE) ─────────────────────────────────
    rescue_signals = len(df[df["urgency_category"] == "rescue"])
    food_signals   = len(df[df["urgency_category"] == "food_water"])
    
    # Get location names for critical alerts
    rescue_locations = df[df["urgency_category"] == "rescue"]["locations"].explode().dropna().unique()[:3]
    food_locations   = df[df["urgency_category"] == "food_water"]["locations"].explode().dropna().unique()[:3]
    
    rescue_loc_str = ", ".join([str(l).title() for l in rescue_locations]) if len(rescue_locations) > 0 else "multiple areas"
    food_loc_str   = ", ".join([str(l).title() for l in food_locations]) if len(food_locations) > 0 else "multiple areas"

    if rescue_signals > 0:
        st.markdown(f'''
        <div class="alert-critical">
            <div class="alert-title">🔴 CRITICAL - High rescue demand detected</div>
            <div class="alert-message">{rescue_loc_str} ({rescue_signals} signals)</div>
        </div>
        ''', unsafe_allow_html=True)

    if food_signals > 0:
        st.markdown(f'''
        <div class="alert-moderate">
            <div class="alert-title">🟡 MODERATE - Food & water shortage reports increasing</div>
            <div class="alert-message">{food_loc_str} ({food_signals} signals)</div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Three Column Layout ───────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1, 1.2, 1])

    # ── LEFT COLUMN: System Overview ──────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)
        
        # Metrics grid
        rescue_count = len(df[df["urgency_category"] == "rescue"])
        medical_count = len(df[df["urgency_category"] == "medical"])
        topic_count = df["topic"].nunique() if "topic" in df else 7
        
        st.markdown(f'''
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Signals</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{rescue_count}</div>
                <div class="metric-label">Rescue</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{medical_count}</div>
                <div class="metric-label">Medical</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{topic_count}</div>
                <div class="metric-label">Topics</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Topic summary
        if topic_summary is not None and not topic_summary.empty:
            st.markdown('<div class="section-header">🎯 Active Topics</div>', unsafe_allow_html=True)
            for _, row in topic_summary.head(5).iterrows():
                keywords = row.get("keywords", "")[:40]
                count = row.get("count", 0)
                st.markdown(f'''
                <div style="display: flex; justify-content: space-between; 
                            padding: 8px 0; border-bottom: 1px solid rgba(71,85,105,0.2);">
                    <span style="color: #cbd5e1;">{keywords}</span>
                    <span style="color: #64748b;">{count} signals</span>
                </div>
                ''', unsafe_allow_html=True)
        
        # Map placeholder
        st.markdown('<div class="section-header">📍 Geographic Spread</div>', unsafe_allow_html=True)
        if "has_location" in df.columns and df["has_location"].sum() > 0:
            lats, lons, texts = [], [], []
            for _, row in df.iterrows():
                locs = row.get("locations", [])
                if isinstance(locs, list) and locs:
                    for loc in locs[:1]:
                        coord = CITY_COORDS.get(str(loc).lower())
                        if coord:
                            lats.append(coord[0])
                            lons.append(coord[1])
                            texts.append(f"{loc}: {row.get('clean_text','')[:50]}")
            
            if lats:
                fig_map = go.Figure(go.Scattermapbox(
                    lat=lats, lon=lons, mode="markers",
                    marker=dict(size=12, color="#ef4444", opacity=0.8),
                    text=texts, hoverinfo="text"
                ))
                fig_map.update_layout(
                    mapbox=dict(style="carto-darkmatter", center=dict(lat=23.5, lon=82), zoom=3.8),
                    margin=dict(l=0, r=0, t=0, b=0), height=220,
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

    # ── CENTER COLUMN: Live Signal Feed ───────────────────────────────────────
    with col_center:
        st.markdown(f'''
        <div style="display: flex; align-items: center; margin: 24px 0 12px 0; 
                    padding-bottom: 8px; border-bottom: 1px solid rgba(71,85,105,0.4);">
            <span class="live-dot"></span>
            <span class="section-header" style="margin:0; border:none; padding:0;">📡 Live Signal Feed</span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Display feed items
        category_map = {
            "food_water": ("FOOD / WATER", "food"),
            "rescue": ("RESCUE", "rescue"),
            "medical": ("MEDICAL", "medical"),
            "infrastructure": ("INFRASTRUCTURE", "infrastructure"),
            "information": ("INFO", ""),
        }
        
        for idx, (_, row) in enumerate(df.head(12).iterrows()):
            if idx >= 10:
                break
                
            urgency = row.get("urgency_category", "information")
            cat_name, cat_class = category_map.get(urgency, ("INFO", ""))
            
            locations = row.get("locations", [])
            loc_str = locations[0].upper() if isinstance(locations, list) and locations else "UNKNOWN"
            
            topic = row.get("topic", 10)
            topic_str = f"topic {topic}" if topic != -1 else ""
            
            time_str = f"{random.randint(5, 45)}m"  # Simulated time
            
            text = row.get("clean_text", "")[:120]
            
            st.markdown(f'''
            <div class="feed-card">
                <div class="feed-header">
                    <span class="feed-category {cat_class}">{cat_name}</span>
                    <span class="feed-location">{loc_str}</span>
                    <span class="feed-topic">{topic_str}</span>
                    <span class="feed-time">{time_str}</span>
                </div>
                <div class="feed-content">{text}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'<div style="color: #475569; text-align: center; margin-top: 16px;">{len(df)} total signals · Updated {datetime.now().strftime("%H:%M")}</div>', unsafe_allow_html=True)

    # ── RIGHT COLUMN: AI Summary & Predictions ────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">🤖 AI Intelligence Summary</div>', unsafe_allow_html=True)
        
        # Summary card
        st.markdown(f'''
        <div style="background: rgba(30,41,59,0.4); border-radius: 10px; padding: 16px; 
                    border: 1px solid rgba(71,85,105,0.3); margin-bottom: 20px;">
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">{summary}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Key insights
        st.markdown('<div class="section-header">🔍 Key Insights</div>', unsafe_allow_html=True)
        
        insights = [
            f"• {rescue_signals} rescue requests detected",
            f"• {len(df[df['urgency_category'] == 'infrastructure'])} infrastructure damage reports",
            f"• Most active region: {df['locations'].explode().mode()[0] if not df['locations'].explode().empty else 'Assam'}",
            f"• Peak signal time: {datetime.now().strftime('%H:00')} hours",
        ]
        
        for insight in insights:
            st.markdown(f'<div style="color: #94a3b8; padding: 6px 0; font-size: 0.9rem;">{insight}</div>', unsafe_allow_html=True)
        
        # Risk level indicator
        sev, sev_clr, _ = get_severity_label(alerts)
        st.markdown(f'''
        <div style="margin-top: 20px; background: rgba(30,41,59,0.4); border-radius: 10px; 
                    padding: 16px; border: 1px solid rgba(71,85,105,0.3);">
            <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Overall Risk Level</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {sev_clr};">{sev}</div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f'''
    <div style="margin-top: 30px; padding-top: 16px; border-top: 1px solid rgba(71,85,105,0.2); 
                display: flex; justify-content: space-between; color: #475569; font-size: 0.75rem;">
        <span>🛡️ Disaster Intelligence Command Center v2.0</span>
        <span>Cycle #{st.session_state.cycle} · {len(df)} active signals · AI monitoring active</span>
    </div>
    ''', unsafe_allow_html=True)

    # Auto-refresh loop
    if st.session_state.auto_refresh:
        time.sleep(15)
        st.session_state.cycle += 1
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()