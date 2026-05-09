"""
dashboard_v2.py  ←  UPGRADED DASHBOARD (use this one)
----------------------------------------------------------
Features:
  ✅ Live auto-refreshing tweet feed (10s cycle)
  ✅ Alert system (CRITICAL / HIGH / MODERATE banners)
  ✅ Interactive geo map (urgency-coloured markers)
  ✅ AI situation summary (rule-based + optional LLM)
  ✅ Time-based trend chart (topic drift over time)
  ✅ Language detection display (Hindi / English)
  ✅ Keyword search filter
  ✅ Severity score gauge
  ✅ Export CSV + report TXT

Run: streamlit run src/dashboard_v2.py
"""

import os, sys, time, random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from live_feed    import get_live_tweets
from preprocessor import preprocess
from intelligence import add_urgency_scores, add_geo_tags
from alert_system import check_alerts, get_severity_label
from summariser   import get_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚨 Disaster Intel Dashboard",
    page_icon="🚨", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1rem}
.alert-critical{background:#2D0000;border-left:5px solid #FF2D2D;border-radius:8px;padding:12px 16px;margin:5px 0}
.alert-high    {background:#2D1000;border-left:5px solid #FF6B00;border-radius:8px;padding:12px 16px;margin:5px 0}
.alert-moderate{background:#2D2500;border-left:5px solid #FFC107;border-radius:8px;padding:12px 16px;margin:5px 0}
.severity-badge{display:inline-block;padding:5px 16px;border-radius:20px;font-weight:bold;font-size:14px;color:white}
.live-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#FF2D2D;margin-right:6px;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.tweet-card{background:#1A1A2E;border-radius:8px;padding:10px 14px;margin:4px 0;border-left:3px solid}
</style>
""", unsafe_allow_html=True)

URGENCY_COLORS = {
    "rescue":"#FF2D2D","medical":"#FF6B00","food_water":"#FFC107",
    "infrastructure":"#2196F3","information":"#4CAF50",
}
URGENCY_ICONS  = {
    "rescue":"🆘","medical":"🏥","food_water":"🍽️",
    "infrastructure":"🏗️","information":"ℹ️",
}
CITY_COORDS = {
    "chennai":(13.08,80.27),"assam":(26.14,91.74),"patna":(25.59,85.13),
    "bihar":(25.09,85.31),"uttarakhand":(30.06,79.01),"mumbai":(19.07,72.87),
    "odisha":(20.29,85.82),"gujarat":(22.25,71.19),"srinagar":(34.08,74.79),
    "silchar":(24.82,92.80),"manipur":(24.66,93.90),"kedarnath":(30.73,79.06),
    "guwahati":(26.14,91.74),"bhubaneswar":(20.29,85.82),"kolkata":(22.57,88.36),
    "himachal":(31.10,77.17),"rajasthan":(27.02,74.21),"lucknow":(26.84,80.94),
    "varanasi":(25.31,82.97),"nagpur":(21.14,79.08),
}

# Session state
for key, val in [("cycle",0),("last_refresh",datetime.now()),("auto_refresh",False)]:
    if key not in st.session_state:
        st.session_state[key] = val


@st.cache_data(ttl=15)
def run_pipeline(cycle: int, n: int = 25):
    df = get_live_tweets(n=n)
    df = preprocess(df, text_col="text", filter_lang=False, dedup=True)
    if len(df) < 5:
        return None, None, None
    df = add_urgency_scores(df)
    df = add_geo_tags(df)
    now = datetime.now()
    df["timestamp"] = [now - timedelta(minutes=random.randint(0,60)) for _ in range(len(df))]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["topic_label"] = df["urgency_category"].map({
        "rescue":"🆘 Emergency Rescue","medical":"🏥 Medical Crisis",
        "food_water":"🍽️ Food & Water","infrastructure":"🏗️ Infrastructure",
        "information":"ℹ️ Official Updates",
    })
    alerts  = check_alerts(df)
    summary = get_summary(df, alerts, use_llm=False)
    return df, alerts, summary


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ Disaster Intel")
    st.caption("Real-time Social Media NLP")
    st.divider()

    st.markdown("### ⚙️ Feed Controls")
    auto = st.toggle("🔴 Auto-Refresh (10s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto
    n_tweets = st.slider("Tweets per batch", 10, 50, 25, 5)

    if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
        st.session_state.cycle += 1
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 🔍 Search Tweets")
    search_query = st.text_input("Filter by keyword", placeholder="flood, rescue, Mumbai…")

    st.divider()
    st.markdown("### 🎯 Filter Urgency")
    selected_urgencies = st.multiselect(
        "Show categories", options=list(URGENCY_COLORS.keys()),
        default=list(URGENCY_COLORS.keys()),
    )

    st.divider()
    show_lang = st.checkbox("Show language detection", value=True)

    st.divider()
    use_llm   = st.checkbox("Use Claude API summary", value=False)
    st.caption(f"Cycle #{st.session_state.cycle} | "
               f"Last: {st.session_state.last_refresh.strftime('%H:%M:%S')}")


# Auto-refresh gate
if st.session_state.auto_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).seconds
    if elapsed >= 10:
        st.session_state.cycle += 1
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()

# Load data
df, alerts, summary = run_pipeline(st.session_state.cycle, n=n_tweets)
if df is None:
    st.error("Pipeline returned no data. Try refreshing.")
    st.stop()

# Apply filters
fdf = df.copy()
if selected_urgencies:
    fdf = fdf[fdf["urgency_category"].isin(selected_urgencies)]
if search_query:
    fdf = fdf[fdf["clean_text"].str.contains(search_query, case=False, na=False)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
h1, h2 = st.columns([4,1])
with h1:
    st.markdown("## 🚨 Disaster Response Intelligence Dashboard")
    st.caption("Unsupervised NLP · BERTopic · Sentence-BERT · Real-time Feed")
with h2:
    if st.session_state.auto_refresh:
        st.markdown('<div style="margin-top:20px"><span class="live-dot"></span>'
                    '<span style="color:#FF2D2D;font-weight:bold">LIVE</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-top:20px;color:#888">⏸ Paused</div>',
                    unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALERT BANNERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if alerts:
    st.markdown("### 🔔 Active Alerts")
    for a in alerts:
        css = "alert-critical" if a.level=="CRITICAL" else f"alert-{a.level.lower()}"
        st.markdown(f'<div class="{css}">{a.headline()}</div>', unsafe_allow_html=True)
else:
    st.success("✅ No critical alerts — situation appears stable")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS ROW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sev_label, sev_color, _ = get_severity_label(alerts)
m1,m2,m3,m4,m5,m6 = st.columns(6)
with m1: st.metric("📨 Signals",  len(df))
with m2: st.metric("🆘 Rescue",   len(df[df["urgency_category"]=="rescue"]))
with m3: st.metric("🏥 Medical",  len(df[df["urgency_category"]=="medical"]))
with m4: st.metric("🍽️ Food/Water",len(df[df["urgency_category"]=="food_water"]))
with m5: st.metric("📍 Geo-Tagged",df["has_location"].sum() if "has_location" in df else 0)
with m6:
    st.markdown(f'<div style="text-align:center;margin-top:4px">'
                f'<div style="font-size:12px;color:#888;margin-bottom:4px">Severity</div>'
                f'<span class="severity-badge" style="background:{sev_color}">{sev_label}</span>'
                f'</div>', unsafe_allow_html=True)

st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAP + PIE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
map_col, pie_col = st.columns([2,1])

with map_col:
    st.markdown("### 📍 Geo-Intelligence Map")
    map_rows = []
    for _, row in df.iterrows():
        locs = row.get("locations",[])
        if not isinstance(locs, list): continue
        for loc in locs:
            c = CITY_COORDS.get(loc.lower())
            if c:
                map_rows.append({
                    "city": loc.title(),
                    "lat":  c[0]+random.uniform(-.3,.3),
                    "lon":  c[1]+random.uniform(-.3,.3),
                    "urgency": row.get("urgency_category","information"),
                    "tweet": str(row.get("clean_text",""))[:80],
                })
    if map_rows:
        mdf = pd.DataFrame(map_rows)
        fig_map = px.scatter_mapbox(
            mdf, lat="lat", lon="lon", color="urgency",
            color_discrete_map=URGENCY_COLORS,
            hover_name="city", hover_data={"tweet":True,"urgency":True,"lat":False,"lon":False},
            zoom=4, center={"lat":23.5,"lon":82},
            mapbox_style="carto-darkmatter", height=380,
        )
        fig_map.update_traces(marker=dict(size=12,opacity=0.85))
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(font=dict(size=11),bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No geo-tagged tweets in this batch")

with pie_col:
    st.markdown("### 🚦 Urgency Mix")
    uc = df["urgency_category"].value_counts().reset_index()
    uc.columns = ["category","count"]
    uc["label"] = uc["category"].map(lambda c: f"{URGENCY_ICONS.get(c,'')} {c}")
    colors = [URGENCY_COLORS.get(c,"#888") for c in uc["category"]]
    fig_pie = go.Figure(go.Pie(
        labels=uc["label"], values=uc["count"],
        marker=dict(colors=colors), hole=0.45, textinfo="percent+label",
    ))
    fig_pie.update_layout(height=380,paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False,margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TREND CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("### 📈 Topic Trend Over Time (last 60 min)")
df["time_bin"] = df["timestamp"].dt.floor("10min")
trend = df.groupby(["time_bin","urgency_category"]).size().reset_index(name="count")
if not trend.empty:
    fig_t = px.line(trend, x="time_bin", y="count", color="urgency_category",
                    color_discrete_map=URGENCY_COLORS, markers=True, height=260,
                    labels={"time_bin":"Time","count":"Signals","urgency_category":"Category"})
    fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,.15)",
                        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.08)"),
                        legend=dict(bgcolor="rgba(0,0,0,.4)"),
                        margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig_t, use_container_width=True)

st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI SUMMARY + LIVE FEED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sc, fc = st.columns([1,1])

with sc:
    st.markdown("### 🤖 AI Situation Summary")
    if use_llm:
        api_key = os.getenv("ANTHROPIC_API_KEY","")
        live_summary = get_summary(df, alerts, use_llm=True, api_key=api_key or None)
    else:
        live_summary = summary
    st.markdown(
        f'<div style="background:#1A1A2E;border-radius:10px;padding:16px;'
        f'font-family:monospace;font-size:13px;line-height:1.7;'
        f'white-space:pre-wrap;color:#E0E0E0">{live_summary}</div>',
        unsafe_allow_html=True,
    )

with fc:
    st.markdown("### 📡 Live Tweet Feed")
    if search_query:
        st.caption(f'🔍 "{search_query}" — {len(fdf)} results')
    display = fdf if not fdf.empty else df
    lang_map = {"en":"🇬🇧 EN","hi":"🇮🇳 HI","unknown":"❓"}
    for _, row in display.head(8).iterrows():
        urg   = row.get("urgency_category","information")
        color = URGENCY_COLORS.get(urg,"#888")
        icon  = URGENCY_ICONS.get(urg,"ℹ️")
        tweet = str(row.get("clean_text",""))[:120]
        locs  = row.get("locations",[])
        loc_s = f"📍 {', '.join(locs[:2])}" if isinstance(locs,list) and locs else ""
        lang  = lang_map.get(row.get("lang","unknown"),"") if show_lang else ""
        t_str = row["timestamp"].strftime("%H:%M") if "timestamp" in row else ""
        st.markdown(
            f'<div class="tweet-card" style="border-color:{color}">'
            f'<div style="font-size:12px;color:{color};margin-bottom:3px">'
            f'{icon} {urg.upper()} {loc_s} {lang}'
            f'<span style="float:right;color:#666">{t_str}</span></div>'
            f'<div style="color:#DDD;font-size:13px">{tweet}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("### 📥 Export")
e1,e2,e3 = st.columns(3)
ts = datetime.now().strftime("%Y%m%d_%H%M")

with e1:
    st.download_button("⬇️ Tweet CSV", df.to_csv(index=False).encode(),
                       f"signals_{ts}.csv","text/csv",use_container_width=True)
with e2:
    report = summary + ("\n\nALERTS:\n" + "\n".join(a.headline() for a in alerts) if alerts else "")
    st.download_button("⬇️ Report TXT", report.encode(),
                       f"report_{ts}.txt","text/plain",use_container_width=True)
with e3:
    tc = df["urgency_category"].value_counts().reset_index()
    tc.columns = ["category","count"]
    st.download_button("⬇️ Topic Summary CSV", tc.to_csv(index=False).encode(),
                       f"topics_{ts}.csv","text/csv",use_container_width=True)

st.caption(f"🛰️ SIH Project · Disaster Topic Discovery · "
           f"BERTopic + Sentence-BERT · Cycle #{st.session_state.cycle}")

# Auto-refresh
if st.session_state.auto_refresh:
    time.sleep(10)
    st.session_state.cycle += 1
    st.session_state.last_refresh = datetime.now()
    st.cache_data.clear()
    st.rerun()
