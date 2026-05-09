"""
dashboard.py
------------
Streamlit dashboard for disaster topic intelligence.
Run: streamlit run dashboard.py

Shows:
- Topic overview cards with urgency badges
- Tweet count per topic (bar chart)
- Urgency distribution (pie chart)
- Per-topic tweet browser
- Geo-location mentions
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Topic Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.topic-card {
    background: #1E1E2E;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 5px solid;
}
.urgency-badge {
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: bold;
    color: white;
}
.metric-card {
    background: #16213E;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

URGENCY_COLORS = {
    "rescue":         "#FF2D2D",
    "medical":        "#FF6B00",
    "food_water":     "#FFC107",
    "infrastructure": "#2196F3",
    "information":    "#4CAF50",
}

URGENCY_ICONS = {
    "rescue":         "🆘",
    "medical":        "🏥",
    "food_water":     "🍽️",
    "infrastructure": "🏗️",
    "information":    "ℹ️",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    tweets_path = os.path.join(OUTPUT_DIR, "tweets_with_topics.csv")
    intel_path  = os.path.join(OUTPUT_DIR, "topic_intelligence.csv")
    topic_path  = os.path.join(OUTPUT_DIR, "topic_summary.csv")

    if not os.path.exists(tweets_path):
        return None, None, None

    df    = pd.read_csv(tweets_path)
    intel = pd.read_csv(intel_path)
    topics = pd.read_csv(topic_path)

    # Parse locations stored as string
    if "locations" in df.columns:
        df["locations"] = df["locations"].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )

    return df, intel, topics


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/sos.png", width=80)
    st.title("🛰️ Disaster Intel")
    st.caption("Social Media Topic Discovery")
    st.divider()

    if st.button("🔄 Re-run Pipeline", use_container_width=True):
        import subprocess
        with st.spinner("Running pipeline... (~2 min)"):
            subprocess.run(
                ["python", os.path.join(os.path.dirname(__file__), "pipeline.py")],
                capture_output=True
            )
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**How to use**")
    st.markdown("1. Run `pipeline.py` first")
    st.markdown("2. Refresh this dashboard")
    st.markdown("3. Explore topics & urgency")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🚨 Disaster Response Topic Intelligence Dashboard")
st.caption("Unsupervised NLP • BERTopic • Real-time Social Media Analysis")

df, intel, topics = load_data()

if df is None:
    st.warning("⚠️ No output data found. Please run `python src/pipeline.py` first.")
    st.code("cd disaster_topic_discovery\npython src/pipeline.py", language="bash")
    st.stop()

# ── Top metrics row ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

n_tweets  = len(df)
n_topics  = len(intel)
n_rescue  = len(df[df.get("urgency_category", pd.Series()) == "rescue"]) if "urgency_category" in df else 0
n_geo     = df["has_location"].sum() if "has_location" in df else 0
n_medical = len(df[df.get("urgency_category", pd.Series()) == "medical"]) if "urgency_category" in df else 0

with col1:
    st.metric("📨 Total Tweets", f"{n_tweets:,}")
with col2:
    st.metric("🗂️ Topics Found", n_topics)
with col3:
    st.metric("🆘 Rescue Signals", n_rescue)
with col4:
    st.metric("🏥 Medical Alerts", n_medical)
with col5:
    st.metric("📍 Geo-Tagged", n_geo)

st.divider()

# ── Layout: charts + topics ───────────────────────────────────────────────────
left, right = st.columns([1.5, 1])

with left:
    st.subheader("📊 Topics by Tweet Volume")
    if not topics.empty:
        fig = px.bar(
            topics.sort_values("tweet_count", ascending=True).tail(12),
            x="tweet_count",
            y="topic_name",
            orientation="h",
            color="tweet_count",
            color_continuous_scale="Reds",
            labels={"tweet_count": "Tweets", "topic_name": "Topic"},
        )
        fig.update_layout(
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🚦 Urgency Distribution")
    if "urgency_category" in df.columns:
        urgency_counts = df["urgency_category"].value_counts().reset_index()
        urgency_counts.columns = ["category", "count"]
        urgency_counts["icon"] = urgency_counts["category"].map(URGENCY_ICONS)
        urgency_counts["label"] = urgency_counts["icon"] + " " + urgency_counts["category"]

        colors = [URGENCY_COLORS.get(c, "#888") for c in urgency_counts["category"]]
        fig2 = go.Figure(go.Pie(
            labels=urgency_counts["label"],
            values=urgency_counts["count"],
            marker=dict(colors=colors),
            hole=0.4,
        ))
        fig2.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Topic intelligence cards ──────────────────────────────────────────────────
st.subheader("🎯 Topic Intelligence Cards (Ranked by Urgency)")

if not intel.empty:
    for _, row in intel.iterrows():
        urgency = row.get("dominant_urgency", "information")
        color   = URGENCY_COLORS.get(urgency, "#888")
        icon    = URGENCY_ICONS.get(urgency, "ℹ️")

        with st.expander(
            f"{icon} Topic {int(row['topic_id'])}  |  "
            f"{row['tweet_count']} tweets  |  "
            f"Urgency: {urgency.upper()}",
            expanded=False,
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Dominant Urgency:** `{urgency}`")
                st.markdown(f"**Avg Urgency Score:** `{row['avg_urgency_score']}`")
                locs = row.get("top_locations", "[]")
                if isinstance(locs, str):
                    locs = eval(locs)
                st.markdown(f"**Top Locations:** {', '.join(locs) if locs else 'None detected'}")

            with c2:
                # Sample tweets for this topic
                sample = df[df["topic"] == row["topic_id"]]["clean_text"].head(3)
                st.markdown("**Sample Tweets:**")
                for tweet in sample:
                    st.markdown(f"- _{tweet}_")

st.divider()

# ── Tweet browser ─────────────────────────────────────────────────────────────
st.subheader("🔍 Browse Tweets by Topic")
topic_ids = sorted(df["topic"].unique())
topic_ids = [t for t in topic_ids if t != -1]
selected_topic = st.selectbox("Select Topic ID", topic_ids)

filtered = df[df["topic"] == selected_topic][
    ["clean_text", "urgency_category", "urgency_score", "locations"]
].reset_index(drop=True)

st.dataframe(filtered, use_container_width=True, height=300)

st.caption("Built for SIH · Powered by BERTopic + Sentence-BERT · Unsupervised NLP")
