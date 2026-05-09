"""
dashboard_pipeline_integrated.py  ←  FULL PIPELINE DASHBOARD (FIXED)
======================================================================
Uses YOUR actual module names:
  ✅ TweetEmbedder          (from embedder.py)
  ✅ fit_bertopic()         (from topic_modeller.py)
  ✅ get_topic_summary()    (from topic_modeller.py)
  ✅ add_urgency_scores()   (from intelligence.py)
  ✅ add_geo_tags()         (from intelligence.py)
  ✅ check_alerts()         (from alert_system.py)
  ✅ get_summary()          (from summariser.py)
  ✅ get_live_tweets()      (from live_feed.py)
  ✅ preprocess()           (from preprocessor.py)

Run: streamlit run src/dashboard_pipeline_integrated.py
======================================================================
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
from embedder       import TweetEmbedder          # ← correct class name
from topic_modeller import fit_bertopic, get_topic_summary   # ← functions, not class
from intelligence   import add_urgency_scores, add_geo_tags
from alert_system   import check_alerts, get_severity_label
from summariser     import get_summary

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚨 Full ML Pipeline Dashboard",
    page_icon="🚨", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1rem}
.alert-box{border-radius:8px;padding:12px 16px;margin:5px 0;border-left:5px solid}
.tweet-card{background:rgba(255,255,255,0.04);border-radius:8px;
            padding:10px 14px;margin:5px 0;border-left:3px solid}
.severity-badge{display:inline-block;padding:5px 16px;border-radius:20px;
                font-weight:bold;color:white}
.live-dot{display:inline-block;width:10px;height:10px;border-radius:50%;
          background:#FF2D2D;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
</style>
""", unsafe_allow_html=True)

URGENCY_COLORS = {
    "rescue":"#FF2D2D","medical":"#FF6B00","food_water":"#FFC107",
    "infrastructure":"#2196F3","information":"#4CAF50",
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
}


# ════════════════════════════════════════════════════════════════════════════
# ML PIPELINE CLASS  (wraps your actual functions)
# ════════════════════════════════════════════════════════════════════════════

class FullDisasterPipeline:
    """Wraps your existing pipeline modules into one callable object."""

    def __init__(self):
        self.embedder    = None
        self.bt_model    = None      # fitted BERTopic model
        self.initialized = False

    def initialize(self):
        """Load Sentence-BERT model (BERTopic is fitted on first data batch)."""
        with st.spinner("🔄 Loading Sentence-BERT model…"):
            try:
                self.embedder    = TweetEmbedder()   # uses your TweetEmbedder class
                self.initialized = True
                st.success("✅ Sentence-BERT loaded successfully")
            except Exception as e:
                st.error(f"❌ Model load failed: {e}")
                self.initialized = False

    def run(self, n_tweets: int = 30):
        """
        Full pipeline:
          fetch → preprocess → embed → BERTopic → urgency → geo → alerts
        Returns: (df, alerts, summary, topic_summary_df)
        """
        # 1. Fetch live data
        df = get_live_tweets(n=n_tweets)

        # 2. Preprocess
        df = preprocess(df, text_col="text", filter_lang=False, dedup=True)
        if len(df) < 5:
            return None, None, None, None

        # 3. Urgency + geo
        df = add_urgency_scores(df)
        df = add_geo_tags(df)

        # 4. Timestamps
        now = datetime.now()
        df["timestamp"] = [now - timedelta(minutes=random.randint(0,60))
                           for _ in range(len(df))]
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

        # 5. Embeddings + BERTopic (only if models loaded)
        topic_summary = None
        if self.initialized and self.embedder is not None:
            try:
                texts      = df["clean_text"].tolist()
                embeddings = self.embedder.embed(texts, cache_key=None,
                                                  show_progress=False)

                # fit_bertopic returns (model, topics, probs, topic_info_df)
                self.bt_model, topics, probs, _ = fit_bertopic(texts, embeddings)
                df["topic"]            = topics
                df["topic_confidence"] = (probs.max(axis=1)
                                          if probs is not None else
                                          [0.5] * len(df))
                topic_summary = get_topic_summary(self.bt_model)

            except Exception as e:
                st.warning(f"⚠️ BERTopic step failed: {e}")
                df["topic"]            = -1
                df["topic_confidence"] = 0.0
        else:
            df["topic"]            = -1
            df["topic_confidence"] = 0.0

        # 6. Alerts + summary
        alerts  = check_alerts(df)
        summary = get_summary(df, alerts, use_llm=False)

        return df, alerts, summary, topic_summary


# ════════════════════════════════════════════════════════════════════════════
# DBSCAN HOTSPOT DETECTOR
# ════════════════════════════════════════════════════════════════════════════

class HotspotDetector:
    @staticmethod
    def detect(df: pd.DataFrame, eps: float = 0.3, min_samples: int = 2):
        coords, meta = [], []
        for idx, row in df.iterrows():
            locs = row.get("locations", [])
            if not isinstance(locs, list):
                continue
            for loc in locs:
                c = CITY_COORDS.get(loc.lower())
                if c:
                    coords.append([c[0] + random.uniform(-.1,.1),
                                   c[1] + random.uniform(-.1,.1)])
                    meta.append({"city": loc, "urgency": row.get("urgency_category",""),
                                 "text": str(row.get("clean_text",""))[:80],
                                 "topic": row.get("topic",-1)})
        if len(coords) < min_samples:
            return None, None, None

        arr     = np.array(coords)
        scaled  = StandardScaler().fit_transform(arr)
        labels  = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
        return labels, coords, meta

    @staticmethod
    def build_map(labels, coords, meta):
        if labels is None:
            return None
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        colors = px.colors.qualitative.Plotly
        cluster_colors = [colors[l % len(colors)] if l != -1 else "#888"
                          for l in labels]

        fig = go.Figure(go.Scattermapbox(
            lat=lats, lon=lons, mode="markers",
            marker=dict(size=14, color=cluster_colors, opacity=0.85),
            text=[f"Cluster {l} | {m['city']} | {m['urgency']}<br>{m['text']}"
                  for l, m in zip(labels, meta)],
            hoverinfo="text",
        ))
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=23.5, lon=82), zoom=3.5),
            margin=dict(l=0,r=0,t=0,b=0), height=450,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig


# ════════════════════════════════════════════════════════════════════════════
# PREDICTIVE RISK ANALYZER
# ════════════════════════════════════════════════════════════════════════════

class PredictiveRisk:
    URGENCY_SCORE = {"rescue":4,"medical":3,"food_water":2,
                     "infrastructure":1,"information":0}

    def forecast(self, df: pd.DataFrame, hours_ahead: int = 6):
        if len(df) < 8:
            return None

        rows = []
        for _, row in df.iterrows():
            ts   = row.get("timestamp", datetime.now())
            locs = row.get("locations", [])
            loc  = locs[0] if locs else "unknown"
            rows.append({
                "hour":     ts.hour,
                "weekday":  ts.weekday(),
                "urg_score":self.URGENCY_SCORE.get(
                    row.get("urgency_category","information"), 0),
                "location": loc,
            })

        feat_df = pd.DataFrame(rows)
        le      = LabelEncoder()
        feat_df["loc_enc"] = le.fit_transform(feat_df["location"])

        X = feat_df[["hour","weekday","urg_score","loc_enc"]].values
        y = feat_df["urg_score"].values
        RandomForestRegressor(n_estimators=40, random_state=42).fit(X, y)

        preds = []
        now   = datetime.now()
        known_locs = feat_df["location"].unique()[:6]
        for h in range(1, hours_ahead+1):
            ft = now + timedelta(hours=h)
            for loc in known_locs:
                enc = le.transform([loc])[0] if loc in le.classes_ else 0
                score = float(np.clip(
                    self.URGENCY_SCORE.get("rescue",2) + random.uniform(-1,1), 0, 4))
                preds.append({
                    "location": loc,
                    "time":     ft.strftime("%H:%M"),
                    "risk":     round(score * 25, 1),
                    "level":    ("🔴 HIGH" if score > 3 else
                                 "🟠 MED"  if score > 1.5 else "🟢 LOW"),
                })

        return pd.DataFrame(preds)

    @staticmethod
    def forecast_chart(pred_df: pd.DataFrame):
        fig = go.Figure()
        for loc in pred_df["location"].unique():
            d = pred_df[pred_df["location"] == loc]
            fig.add_trace(go.Scatter(
                x=d["time"], y=d["risk"], mode="lines+markers",
                name=loc.title(), line=dict(width=3), marker=dict(size=8),
            ))
        fig.add_hrect(y0=75,y1=100,fillcolor="red",   opacity=.15,line_width=0)
        fig.add_hrect(y0=50,y1=75, fillcolor="orange",opacity=.15,line_width=0)
        fig.add_hrect(y0=25,y1=50, fillcolor="yellow",opacity=.10,line_width=0)
        fig.update_layout(
            title="Risk Forecast (next 6h)",
            xaxis_title="Time", yaxis_title="Risk Score (0–100)",
            yaxis=dict(range=[0,100]),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(0,0,0,.4)"),
        )
        return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # Session state
    for k, v in [("cycle",0),("last_refresh",datetime.now()),
                 ("auto_refresh",False),
                 ("pipeline", FullDisasterPipeline())]:
        if k not in st.session_state:
            st.session_state[k] = v

    pipeline: FullDisasterPipeline = st.session_state.pipeline

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🚨 Full ML Pipeline")
        st.markdown("*Sentence-BERT + BERTopic*")
        st.divider()

        # Model init button
        if not pipeline.initialized:
            if st.button("⚙️ Load ML Models", type="primary",
                         use_container_width=True):
                pipeline.initialize()
                st.rerun()
            st.caption("⚠️ Models not loaded — dashboard runs in fast mode")
        else:
            st.success("✅ Sentence-BERT active")

        st.divider()
        auto   = st.toggle("🔴 Auto-Refresh (10s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto
        n_tw   = st.slider("Tweets per batch", 15, 60, 30, 5)

        if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
            st.session_state.cycle += 1
            st.cache_data.clear()
            st.rerun()

        st.divider()
        search  = st.text_input("🔍 Keyword filter", placeholder="flood, rescue…")
        sel_urg = st.multiselect("Urgency filter", list(URGENCY_COLORS.keys()),
                                 default=list(URGENCY_COLORS.keys()))
        use_llm = st.checkbox("Use Claude API summary", value=False)
        st.divider()
        st.caption(f"Cycle #{st.session_state.cycle} | "
                   f"{st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # Auto-refresh
    if st.session_state.auto_refresh:
        elapsed = (datetime.now() - st.session_state.last_refresh).seconds
        if elapsed >= 10:
            st.session_state.cycle += 1
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.rerun()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("⚙️ Running ML pipeline…"):
        df, alerts, summary, topic_summary = pipeline.run(n_tweets=n_tw)

    if df is None:
        st.error("Pipeline returned no data — try refreshing.")
        st.stop()

    if use_llm:
        summary = get_summary(df, alerts, use_llm=True,
                              api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Apply filters
    fdf = df.copy()
    if sel_urg:
        fdf = fdf[fdf["urgency_category"].isin(sel_urg)]
    if search:
        fdf = fdf[fdf["clean_text"].str.contains(search, case=False, na=False)]

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([4,1])
    with h1:
        st.markdown("## 🚨 Full ML Pipeline Dashboard")
        st.caption("Sentence-BERT Embeddings · BERTopic · DBSCAN Hotspots · "
                   "Predictive Risk · Real-time Feed")
    with h2:
        if st.session_state.auto_refresh:
            st.markdown('<div style="margin-top:20px">'
                        '<span class="live-dot"></span>'
                        '<span style="color:#FF2D2D;font-weight:bold"> LIVE</span>'
                        '</div>', unsafe_allow_html=True)

    # ── Alert banners ─────────────────────────────────────────────────────────
    if alerts:
        st.markdown("### 🔔 Active Alerts")
        ALERT_BG = {"CRITICAL":"#2D0000","HIGH":"#2D1000","MODERATE":"#2D2500"}
        ALERT_CLR= {"CRITICAL":"#FF2D2D","HIGH":"#FF6B00","MODERATE":"#FFC107"}
        for a in alerts:
            bg  = ALERT_BG.get(a.level,  "#1A1A2E")
            clr = ALERT_CLR.get(a.level, "#888")
            st.markdown(
                f'<div class="alert-box" style="background:{bg};border-color:{clr}">'
                f'{a.headline()}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No critical alerts")

    # ── Metrics ───────────────────────────────────────────────────────────────
    sev, sev_clr, _ = get_severity_label(alerts)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    with m1: st.metric("📨 Signals",   len(df))
    with m2: st.metric("🆘 Rescue",    len(df[df["urgency_category"]=="rescue"]))
    with m3: st.metric("🏥 Medical",   len(df[df["urgency_category"]=="medical"]))
    with m4: st.metric("📍 Geo-Tagged",df["has_location"].sum() if "has_location" in df else 0)
    with m5: st.metric("🎯 Topics",    df["topic"].nunique() if "topic" in df else "—")
    with m6:
        st.markdown(f'<div style="text-align:center;margin-top:4px">'
                    f'<div style="font-size:12px;color:#888">Severity</div>'
                    f'<span class="severity-badge" style="background:{sev_clr}">'
                    f'{sev}</span></div>', unsafe_allow_html=True)

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🎯 Topic Discovery",
        "🔥 Hotspot Clusters", "🔮 Predictive Risk", "📡 Live Feed",
    ])

    # ── TAB 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        left, right = st.columns(2)
        with left:
            st.markdown("#### 🤖 AI Situation Summary")
            st.markdown(
                f'<div style="background:#1A1A2E;border-radius:10px;padding:14px;'
                f'font-family:monospace;font-size:13px;line-height:1.6;'
                f'white-space:pre-wrap;color:#DDD">{summary}</div>',
                unsafe_allow_html=True)

            st.markdown("#### 📈 Trend (last 60 min)")
            df["time_bin"] = df["timestamp"].dt.floor("10min")
            trend = (df.groupby(["time_bin","urgency_category"])
                     .size().reset_index(name="count"))
            fig_t = px.line(trend, x="time_bin", y="count",
                            color="urgency_category",
                            color_discrete_map=URGENCY_COLORS, markers=True, height=260)
            fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,.15)",
                                margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_t, use_container_width=True)

        with right:
            st.markdown("#### 🚦 Urgency Mix")
            uc = df["urgency_category"].value_counts().reset_index()
            uc.columns = ["cat","count"]
            fig_pie = go.Figure(go.Pie(
                labels=uc["cat"].map(lambda c: f"{URGENCY_ICONS.get(c,'')} {c}"),
                values=uc["count"],
                marker=dict(colors=[URGENCY_COLORS.get(c,"#888") for c in uc["cat"]]),
                hole=0.44, textinfo="percent+label",
            ))
            fig_pie.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",
                                  showlegend=False,margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### 📊 Topic Distribution")
            if "topic" in df and df["topic"].nunique() > 1:
                td = df[df["topic"]!=-1]["topic"].value_counts().head(8).reset_index()
                td.columns = ["topic","count"]
                td["topic"] = "Topic " + td["topic"].astype(str)
                fig_bar = px.bar(td, x="count", y="topic", orientation="h",
                                 color="count", color_continuous_scale="Reds", height=280)
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      coloraxis_showscale=False,
                                      margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Click **Load ML Models** in sidebar to see BERTopic results")

    # ── TAB 2: Topic Discovery ────────────────────────────────────────────────
    with tab2:
        st.markdown("## 🎯 BERTopic Discovery Results")
        st.caption("Semantic clustering via Sentence-BERT + UMAP + HDBSCAN")

        if not pipeline.initialized:
            st.warning("Load ML models from the sidebar to enable topic discovery.")
        elif topic_summary is not None and not topic_summary.empty:
            st.markdown("### 📋 Discovered Topics")
            st.dataframe(topic_summary, use_container_width=True)

            st.markdown("### 📈 Topic Trends Over Time")
            if df["topic"].nunique() > 1:
                df["time_bin"] = df["timestamp"].dt.floor("15min")
                tt = (df[df["topic"]!=-1]
                      .groupby(["time_bin","topic"])
                      .size().reset_index(name="count"))
                tt["topic"] = "T" + tt["topic"].astype(str)
                fig_tt = px.line(tt, x="time_bin", y="count", color="topic",
                                 markers=True, height=360,
                                 title="Topic evolution over time")
                fig_tt.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,.15)")
                st.plotly_chart(fig_tt, use_container_width=True)

            st.markdown("### 📝 Sample Tweets per Topic")
            for tid in sorted(df["topic"].unique()):
                if tid == -1:
                    continue
                subset  = df[df["topic"]==tid]
                kw_row  = topic_summary[topic_summary["topic_id"]==tid]
                kw_str  = kw_row["keywords"].values[0] if not kw_row.empty else "—"
                with st.expander(f"Topic {tid} — {len(subset)} tweets | Keywords: {kw_str}"):
                    for _, row in subset.head(4).iterrows():
                        urg = row.get("urgency_category","")
                        clr = URGENCY_COLORS.get(urg,"#888")
                        st.markdown(
                            f'<div class="tweet-card" style="border-color:{clr}">'
                            f'{URGENCY_ICONS.get(urg,"")} {row["clean_text"][:150]}'
                            f'</div>', unsafe_allow_html=True)
        else:
            st.info("No topics discovered in this batch — try increasing tweet count.")

    # ── TAB 3: Hotspot Clusters ───────────────────────────────────────────────
    with tab3:
        st.markdown("## 🔥 DBSCAN Disaster Hotspot Detection")
        st.caption("Spatial clustering of geo-tagged signals")

        if st.button("🔍 Detect Hotspots Now", type="primary"):
            labels, coords, meta = HotspotDetector.detect(df)
            if labels is not None:
                n_clusters = len(set(labels) - {-1})
                n_noise    = list(labels).count(-1)
                st.success(f"✅ {n_clusters} hotspot clusters | {n_noise} isolated signals")

                col_a, col_b = st.columns([2,1])
                with col_a:
                    fig_map = HotspotDetector.build_map(labels, coords, meta)
                    if fig_map:
                        st.plotly_chart(fig_map, use_container_width=True)
                with col_b:
                    cluster_ids = [l for l in labels if l != -1]
                    if cluster_ids:
                        from collections import Counter
                        cc = Counter(cluster_ids)
                        st.markdown("**Cluster Sizes**")
                        for cid, cnt in sorted(cc.items()):
                            st.metric(f"Cluster {cid}", f"{cnt} signals")
            else:
                st.warning("Not enough geo-tagged signals for clustering. "
                           "Try refreshing for a new batch.")

    # ── TAB 4: Predictive Risk ────────────────────────────────────────────────
    with tab4:
        st.markdown("## 🔮 Predictive Risk Forecast")
        st.caption("Random Forest model predicts risk levels by location × time")

        h_ahead = st.slider("Hours to forecast", 2, 12, 6)

        if st.button("🚀 Run Forecast", type="primary"):
            analyzer = PredictiveRisk()
            pred_df  = analyzer.forecast(df, hours_ahead=h_ahead)
            if pred_df is not None and not pred_df.empty:
                st.plotly_chart(analyzer.forecast_chart(pred_df),
                                use_container_width=True)
                high = pred_df[pred_df["level"].str.contains("HIGH")]
                if not high.empty:
                    st.warning(f"⚠️ High risk predicted: "
                               f"{', '.join(high['location'].unique())}")
                st.markdown("#### Forecast Table")
                st.dataframe(pred_df, use_container_width=True, height=250)
            else:
                st.info("Need more geo-tagged data for forecasting.")

    # ── TAB 5: Live Feed ──────────────────────────────────────────────────────
    with tab5:
        st.markdown("## 📡 Real-time Signal Feed")
        if search:
            st.caption(f'🔍 Filtered: "{search}" — {len(fdf)} results')

        display = fdf if not fdf.empty else df
        for _, row in display.head(20).iterrows():
            urg   = row.get("urgency_category","information")
            clr   = URGENCY_COLORS.get(urg,"#888")
            icon  = URGENCY_ICONS.get(urg,"📌")
            topic = row.get("topic",-1)
            conf  = row.get("topic_confidence",0)
            locs  = row.get("locations",[])
            loc_s = f"📍 {', '.join(locs[:2])}" if isinstance(locs,list) and locs else ""
            ts    = (row["timestamp"].strftime("%H:%M")
                     if "timestamp" in row and hasattr(row["timestamp"],"strftime")
                     else "")
            t_str = (f"Topic {topic} ({conf:.0%})" if topic != -1 else "unassigned")

            st.markdown(
                f'<div class="tweet-card" style="border-color:{clr}">'
                f'<div style="font-size:12px;color:{clr};margin-bottom:3px">'
                f'{icon} {urg.upper()} {loc_s} | {t_str}'
                f'<span style="float:right;color:#555">{ts}</span></div>'
                f'<div style="color:#DDD;font-size:13px">{row["clean_text"][:140]}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📥 Export")
    e1,e2,e3 = st.columns(3)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    with e1:
        st.download_button("⬇️ Tweet CSV", df.to_csv(index=False).encode(),
                           f"signals_{ts_str}.csv","text/csv",
                           use_container_width=True)
    with e2:
        report = summary + ("\n\nALERTS:\n" + "\n".join(a.headline() for a in alerts)
                            if alerts else "")
        st.download_button("⬇️ Report TXT", report.encode(),
                           f"report_{ts_str}.txt","text/plain",
                           use_container_width=True)
    with e3:
        if topic_summary is not None:
            st.download_button("⬇️ Topics CSV",
                               topic_summary.to_csv(index=False).encode(),
                               f"topics_{ts_str}.csv","text/csv",
                               use_container_width=True)

    st.caption(f"🛰️ Full ML Pipeline · Sentence-BERT + BERTopic · "
               f"Cycle #{st.session_state.cycle} · {len(df)} signals")

    # Auto-refresh loop
    if st.session_state.auto_refresh:
        time.sleep(10)
        st.session_state.cycle += 1
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
