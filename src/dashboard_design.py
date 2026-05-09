"""
dashboard_design.py
====================
Disaster Intelligence Command Center - Complete Design
Integrated HTML components with real-time data processing
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import json

sys.path.insert(0, os.path.dirname(__file__))

# ======================== PAGE CONFIG =================================
st.set_page_config(
    page_title="Disaster Topic Discovery",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================== CUSTOM CSS THEME =================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root styling */
html, body, .stApp {
    background: #0a0e1a !important;
    color: #e0e8ff !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide default elements */
#MainMenu, footer, header { display: none !important; }
.stDeployButton { display: none !important; }

/* Main container */
.main { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Top bar styling */
.top-bar {
    background: linear-gradient(180deg, rgba(15, 20, 40, 0.98), rgba(10, 14, 28, 0.95));
    border-bottom: 1px solid rgba(77, 163, 255, 0.15);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(10px);
}

.title-section {
    display: flex;
    align-items: center;
    gap: 16px;
}

.title-icon {
    font-size: 32px;
}

.title-text h1 {
    font-size: 24px;
    font-weight: 600;
    margin: 0;
    color: #e0e8ff;
    letter-spacing: 0.5px;
}

.title-text p {
    font-size: 12px;
    color: #7a8bad;
    margin: 4px 0 0 0;
}

.controls-section {
    display: flex;
    align-items: center;
    gap: 16px;
}

.date-selector {
    background: rgba(77, 163, 255, 0.08);
    border: 1px solid rgba(77, 163, 255, 0.2);
    padding: 8px 14px;
    border-radius: 6px;
    color: #7a8bad;
    font-size: 12px;
    cursor: pointer;
}

.export-btn {
    background: linear-gradient(135deg, #4da3ff 0%, #3587e0 100%);
    border: none;
    color: white;
    padding: 8px 20px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.3s;
}

.export-btn:hover {
    box-shadow: 0 0 20px rgba(77, 163, 255, 0.5);
    transform: translateY(-2px);
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 24px 32px;
    background: linear-gradient(180deg, rgba(10, 14, 28, 0.5), rgba(8, 12, 24, 0.8));
}

.metric-card {
    background: linear-gradient(135deg, rgba(15, 24, 50, 0.9), rgba(8, 14, 32, 0.8));
    border: 1px solid rgba(77, 163, 255, 0.2);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.metric-card:hover {
    border-color: rgba(77, 163, 255, 0.4);
    box-shadow: 0 12px 40px rgba(77, 163, 255, 0.2);
    transform: translateY(-4px);
}

.metric-icon {
    font-size: 24px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 32px;
    font-weight: 600;
    color: #e0e8ff;
    margin: 4px 0;
}

.metric-label {
    font-size: 11px;
    color: #7a8bad;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-change {
    font-size: 11px;
    color: #22c55e;
    margin-top: 8px;
}

/* Main content area */
.content-area {
    display: grid;
    grid-template-columns: 1.8fr 1.2fr;
    gap: 24px;
    padding: 24px 32px;
}

.section {
    background: linear-gradient(135deg, rgba(15, 24, 50, 0.9), rgba(8, 14, 32, 0.8));
    border: 1px solid rgba(77, 163, 255, 0.2);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
}

.section-title {
    font-size: 14px;
    font-weight: 600;
    color: #e0e8ff;
    margin: 0 0 16px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid rgba(77, 163, 255, 0.1);
    padding-bottom: 12px;
}

/* Chart styling */
.chart-container {
    width: 100%;
    height: 300px;
    margin-bottom: 16px;
}

/* Alert cards */
.alert-item {
    background: rgba(255, 59, 59, 0.08);
    border-left: 3px solid #ff3b3b;
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

.alert-item.high {
    background: rgba(249, 115, 22, 0.08);
    border-left-color: #f97316;
}

.alert-item.medium {
    background: rgba(250, 204, 21, 0.08);
    border-left-color: #facc15;
}

.alert-item.low {
    background: rgba(34, 197, 94, 0.08);
    border-left-color: #22c55e;
}

.alert-icon {
    font-size: 18px;
    flex-shrink: 0;
    margin-top: 2px;
}

.alert-content {
    flex: 1;
}

.alert-time {
    font-size: 10px;
    color: #4a5878;
    margin-left: auto;
    white-space: nowrap;
}

.alert-title {
    font-size: 12px;
    font-weight: 500;
    color: #e0e8ff;
    margin-bottom: 2px;
}

.alert-desc {
    font-size: 11px;
    color: #7a8bad;
    line-height: 1.4;
}

/* Location list */
.location-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(77, 163, 255, 0.08);
}

.location-name {
    font-size: 12px;
    color: #e0e8ff;
}

.location-count {
    font-size: 12px;
    font-weight: 600;
    color: #4da3ff;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(77, 163, 255, 0.3);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(77, 163, 255, 0.5);
}
</style>
""", unsafe_allow_html=True)

# ======================== SAMPLE DATA GENERATION =================================
def generate_sample_data():
    """Generate realistic sample disaster data"""
    now = datetime.now()
    
    # Sample topics
    topics = {
        0: "Flood",
        1: "Earthquake", 
        2: "Heatwave",
        3: "Cyclone",
        4: "Landslide",
        5: "Others"
    }
    
    # Sample locations
    locations = ["Mumbai", "Chennai", "Kolkata", "Hyderabad", "Bangalore", "Delhi"]
    
    # Generate data points
    data = []
    for i in range(100):
        data.append({
            'topic': topics[random.randint(0, 5)],
            'location': random.choice(locations),
            'mentions': random.randint(10, 100),
            'timestamp': now - timedelta(hours=random.randint(0, 24)),
        })
    
    return pd.DataFrame(data)

# ======================== MAIN DASHBOARD =================================

# Top Bar
st.markdown("""
<div class="top-bar">
    <div class="title-section">
        <div class="title-icon">🚨</div>
        <div class="title-text">
            <h1>DISASTER TOPIC DISCOVERY</h1>
            <p>Real-time insights. Smarter decisions.</p>
        </div>
    </div>
    <div class="controls-section">
        <div class="date-selector">📅 May 12 – May 18, 2024</div>
        <button class="export-btn">⬇️ Export Report</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Load sample data
df = generate_sample_data()

# Calculate metrics
total_sources = 12
total_mentions = len(df)
alerts_count = 23
topics_detected = len(df['topic'].unique())

# Metrics Grid
st.markdown("""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-icon">💾</div>
        <div class="metric-value">12</div>
        <div class="metric-label">Data Sources</div>
        <div class="metric-change">▲ 20% This Week</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">💬</div>
        <div class="metric-value">8,456</div>
        <div class="metric-label">Live Mentions</div>
        <div class="metric-change">▲ 18% Last 24h</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">🔔</div>
        <div class="metric-value">23</div>
        <div class="metric-label">Alerts</div>
        <div class="metric-change">▲ 5% This Week</div>
    </div>
    <div class="metric-card">
        <div class="metric-icon">👥</div>
        <div class="metric-value">36</div>
        <div class="metric-label">Topics Detected</div>
        <div class="metric-change">▲ 12% This Week</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Content Grid
st.markdown('<div class="content-area">', unsafe_allow_html=True)

# Left Column - Charts
col_left = st.container()

with col_left:
    # Top Discovered Topics (Pie Chart)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Discovered Topics</div>', unsafe_allow_html=True)
    
    topic_counts = df['topic'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=topic_counts.index,
        values=topic_counts.values,
        marker=dict(colors=['#9d4edd', '#3a86ff', '#06ffa5', '#ff006e', '#ffbe0b', '#8338ec']),
        textposition='inside',
        textinfo='label+value',
        hovertemplate='%{label}: %{value} mentions<extra></extra>',
    )])
    
    fig_pie.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=11, color='#e0e8ff'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(77, 163, 255, 0.08)',
            bordercolor='rgba(77, 163, 255, 0.2)',
            borderwidth=1,
            font=dict(size=10, color='#7a8bad'),
        ),
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mentions Over Time
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Mentions Over Time</div>', unsafe_allow_html=True)
    
    df['date'] = df['timestamp'].dt.date
    timeline = df.groupby('date').size().reset_index(name='count')
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=timeline['date'],
        y=timeline['count'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#4da3ff', width=3),
        fillcolor='rgba(77, 163, 255, 0.2)',
        hovertemplate='%{x}: %{y} mentions<extra></extra>',
    ))
    
    fig_line.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(77, 163, 255, 0.03)',
        xaxis=dict(
            gridcolor='rgba(77, 163, 255, 0.07)',
            showgrid=True,
            tickfont=dict(size=10, color='#4a5878'),
        ),
        yaxis=dict(
            gridcolor='rgba(77, 163, 255, 0.07)',
            showgrid=True,
            tickfont=dict(size=10, color='#4a5878'),
        ),
        font=dict(family='Inter', color='#e0e8ff'),
        showlegend=False,
    )
    
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Alerts and Locations
st.markdown("""
<style>
.content-area > div:nth-child(2) {
    grid-column: 2;
    grid-row: 1 / 3;
}
</style>
""", unsafe_allow_html=True)

col_right = st.container()

with col_right:
    # Alerts Summary
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Alerts Summary</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-item">
        <div class="alert-icon">🔴</div>
        <div class="alert-content">
            <div class="alert-title">Critical</div>
            <div class="alert-desc">Heavy rainfall causes flooding in multiple areas.</div>
        </div>
        <div class="alert-time">10:24 AM</div>
    </div>
    
    <div class="alert-item">
        <div class="alert-icon">🟠</div>
        <div class="alert-content">
            <div class="alert-title">Magnitude 5.3 Earthquake</div>
            <div class="alert-desc">Earthquake reported near the coast.</div>
        </div>
        <div class="alert-time">10:21 AM</div>
    </div>
    
    <div class="alert-item medium">
        <div class="alert-icon">🟡</div>
        <div class="alert-content">
            <div class="alert-title">Heatwave Alert</div>
            <div class="alert-desc">Heatwave alert issued for 5 districts.</div>
        </div>
        <div class="alert-time">10:18 AM</div>
    </div>
    
    <div class="alert-item">
        <div class="alert-icon">🟣</div>
        <div class="alert-content">
            <div class="alert-title">Cyclone Update</div>
            <div class="alert-desc">Cyclone update: Landfall expected tomorrow evening.</div>
        </div>
        <div class="alert-time">10:15 AM</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Top Locations
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Locations</div>', unsafe_allow_html=True)
    
    location_counts = df['location'].value_counts().head(6)
    
    for location, count in location_counts.items():
        st.markdown(f"""
        <div class="location-item">
            <span class="location-name">📍 {location}</span>
            <span class="location-count">{count}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 24px; color: #4a5878; font-size: 11px; border-top: 1px solid rgba(77, 163, 255, 0.1); margin-top: 32px;">
    © 2024 Disaster Topic Discovery. Real-time intelligence powered by AI.
</div>
""", unsafe_allow_html=True)
