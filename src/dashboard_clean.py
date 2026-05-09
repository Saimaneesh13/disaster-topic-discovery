"""
dashboard_clean.py
==================
DisasterAI Command Center v2.0 - Clean Layout Design
Streamlit wrapper for HTML dashboard with optimized layout
"""

import streamlit as st
import streamlit.components.v1 as components

# ======================== PAGE CONFIG =================================
st.set_page_config(
    page_title="DisasterAI · Command Center v2.0",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit UI
st.markdown("""
<style>
    .stApp { background: transparent !important; }
    #MainMenu {display: none}
    footer {display: none}
    .stDeployButton {display: none}
    header {display: none}
    [data-testid="stToolbar"] {display: none}
    .block-container { padding: 0 !important; margin: 0 !important; }
    body { margin: 0 !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ======================== MAIN HTML COMPONENT =====================================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DisasterAI · Command Center v2.0</title>
  <!-- Tailwind -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Leaflet CSS + JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {
      background: #0a0f1e;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .glass-panel {
      background: rgba(17, 24, 39, 0.7);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 1rem;
    }
    .sidebar-item {
      transition: all 0.15s ease;
    }
    .sidebar-item:hover, .sidebar-item.active {
      background: rgba(99, 102, 241, 0.15);
      border-left: 3px solid #6366f1;
    }
    @keyframes pulse-red {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .category-btn.active {
      background: #6366f1;
      color: white;
    }
    #map {
      z-index: 1;
    }
  </style>
</head>
<body class="text-gray-200 p-4 h-screen flex gap-4 overflow-hidden">

  <!-- SIDEBAR -->
  <aside class="w-64 flex-shrink-0 glass-panel p-4 flex flex-col">
    <div class="mb-8">
      <h1 class="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
        DisasterAI
      </h1>
      <p class="text-xs text-gray-500">Command Center v2.0</p>
    </div>
    <nav class="flex-1 space-y-2">
      <div class="sidebar-item active px-3 py-2 rounded-r-lg cursor-pointer flex items-center gap-3">
        <span>📊</span> Overview
      </div>
      <div class="sidebar-item px-3 py-2 rounded-r-lg cursor-pointer flex items-center gap-3">
        <span>🎯</span> Topic Discovery
      </div>
      <div class="sidebar-item px-3 py-2 rounded-r-lg cursor-pointer flex items-center gap-3">
        <span>📍</span> Hotspot Clusters
      </div>
      <div class="sidebar-item px-3 py-2 rounded-r-lg cursor-pointer flex items-center gap-3">
        <span>📡</span> Live Feed
      </div>
    </nav>
    <div class="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-500 flex items-center gap-2">
      <span class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span> System Online
    </div>
  </aside>

  <!-- MAIN CONTENT -->
  <main class="flex-1 flex flex-col gap-4 overflow-y-auto pr-1">

    <!-- TOP CONTROL BAR -->
    <div class="glass-panel p-4 flex flex-wrap items-center gap-4">
      <div class="flex items-center gap-2 flex-1 min-w-[200px]">
        <span>🔍</span>
        <input type="text" placeholder="Keyword filter: flood, rescue..." 
               class="bg-gray-900/50 rounded-lg px-3 py-1.5 text-sm w-full outline-none border border-gray-700 focus:border-indigo-500" 
               value="flood, rescue">
      </div>
      <div class="flex gap-2 flex-wrap">
        <button class="category-btn active px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700">rescue</button>
        <button class="category-btn px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700">medical</button>
        <button class="category-btn px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700">food_water</button>
        <button class="category-btn px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700">infrastructure</button>
        <button class="category-btn px-3 py-1 text-xs rounded-full bg-gray-800 hover:bg-gray-700">information</button>
      </div>
      <div class="flex items-center gap-4 text-sm text-gray-400">
        <span>◉ 5 active</span>
        <span>⟳ 00000</span>
        <span>⏱ 21:19:11</span>
        <button class="bg-yellow-600/20 text-yellow-400 px-3 py-1 rounded-lg text-xs flex items-center gap-1">
          ⏸️ PAUSED
        </button>
      </div>
    </div>

    <!-- ALERT BANNERS -->
    <div class="grid grid-cols-2 gap-4">
      <div class="bg-red-900/30 border border-red-700/40 rounded-xl p-3 flex items-start gap-3">
        <span class="text-2xl">🆘</span>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-bold text-red-400">CRITICAL</span>
            <span class="text-xs bg-red-700/50 px-2 py-0.5 rounded">9 signals</span>
          </div>
          <p class="text-sm text-gray-300 mt-1">High rescue demand — bhubaneswar, silchar, brahmaputra</p>
        </div>
      </div>
      <div class="bg-orange-900/30 border border-orange-700/40 rounded-xl p-3 flex items-start gap-3">
        <span class="text-2xl">🍽️</span>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-bold text-orange-400">MODERATE</span>
            <span class="text-xs bg-orange-700/50 px-2 py-0.5 rounded">13 signals</span>
          </div>
          <p class="text-sm text-gray-300 mt-1">Food & water shortage — brahmaputra, chennai, himachal</p>
        </div>
      </div>
    </div>

    <!-- METRICS ROW -->
    <div class="grid grid-cols-4 gap-4">
      <div class="glass-panel p-4">
        <div class="text-gray-400 text-xs uppercase">Total Signals</div>
        <div class="text-3xl font-bold mt-1">29</div>
      </div>
      <div class="glass-panel p-4">
        <div class="text-gray-400 text-xs uppercase">🆘 Rescue</div>
        <div class="text-3xl font-bold mt-1 text-red-400">9</div>
      </div>
      <div class="glass-panel p-4">
        <div class="text-gray-400 text-xs uppercase">🏥 Medical</div>
        <div class="text-3xl font-bold mt-1 text-blue-400">0</div>
      </div>
      <div class="glass-panel p-4">
        <div class="text-gray-400 text-xs uppercase">📍 Geo-Tagged</div>
        <div class="text-3xl font-bold mt-1 text-green-400">26</div>
      </div>
    </div>

    <!-- MAP + TREND + METRICS -->
    <div class="grid grid-cols-3 gap-4">
      <!-- Real-time Metrics Dashboard -->
      <div class="glass-panel p-4 space-y-3">
        <h3 class="font-bold text-sm flex items-center gap-2">📊 Real-time Metrics</h3>
        <div class="flex justify-between"><span class="text-gray-400">🔴 High Priority</span><span class="font-mono">31.0%</span></div>
        <div class="flex justify-between"><span class="text-gray-400">📍 Locations</span><span class="font-mono">18</span></div>
        <div class="flex justify-between"><span class="text-gray-400">🎯 Top Urgency</span><span class="font-mono text-orange-400">FOOD_WATER</span></div>
        <div class="flex justify-between"><span class="text-gray-400">🤖 Avg Confidence</span><span class="font-mono">—</span></div>
        <div class="flex justify-between"><span class="text-gray-400">⏱ Data Points</span><span class="font-mono">29</span></div>
        <div class="flex justify-between"><span class="text-gray-400">📊 Alerts Active</span><span class="font-mono text-red-400">2</span></div>
      </div>

      <!-- Geo Intelligence Map -->
      <div class="glass-panel p-4 col-span-2">
        <h3 class="font-bold text-sm mb-2 flex items-center gap-2">🌍 Geo Intelligence Map</h3>
        <div id="map" class="w-full h-64 rounded-lg overflow-hidden border border-gray-700"></div>
      </div>
    </div>

    <!-- SIGNAL TREND + SYSTEM OVERVIEW -->
    <div class="grid grid-cols-3 gap-4">
      <!-- Signal Trend -->
      <div class="glass-panel p-4 col-span-2">
        <h3 class="font-bold text-sm mb-2">📈 Signal Trend — Last 60 min</h3>
        <canvas id="trendChart" height="80"></canvas>
      </div>
      <!-- System Overview -->
      <div class="glass-panel p-4 space-y-2">
        <h3 class="font-bold text-sm">⚙️ System Overview</h3>
        <div class="flex justify-between"><span>Signals</span><span class="font-mono">29</span></div>
        <div class="flex justify-between"><span>Rescue</span><span class="font-mono text-red-400">9</span></div>
        <div class="flex justify-between"><span>Medical</span><span class="font-mono">0</span></div>
        <div class="flex justify-between"><span>Topics</span><span class="font-mono">—</span></div>
        <div class="mt-3">
          <div class="text-xs text-gray-400 mb-1">Urgency Distribution</div>
          <div class="w-full bg-gray-800 rounded-full h-2 flex">
            <div class="bg-red-500 h-2 rounded-l-full" style="width:31%"></div>
            <div class="bg-orange-500 h-2" style="width:45%"></div>
            <div class="bg-yellow-500 h-2" style="width:17%"></div>
            <div class="bg-gray-600 h-2 rounded-r-full" style="width:7%"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- AI SITUATION SUMMARY -->
    <div class="glass-panel p-4">
      <h3 class="font-bold text-sm mb-2 flex items-center gap-2">📋 AI Situation Summary</h3>
      <div class="text-sm text-gray-300 space-y-1">
        <p><span class="text-red-400 font-semibold">🚨 CRITICAL:</span> Mass rescue operations required — 9 distress signals detected.</p>
        <p><span class="text-orange-400 font-semibold">🍽️ Food & water crisis:</span> 13 shortage reports from relief camps.</p>
        <p><span class="text-yellow-400 font-semibold">🏗️ Infrastructure:</span> 3 reports of road/bridge/power damage.</p>
        <p class="mt-2 text-gray-400">📍 Most affected: brahmaputra, assam, bhubaneswar, silchar, himachal</p>
        <p>🔔 Active alerts: 2 (CRITICAL, MODERATE) · 📊 Total signals analysed: 29</p>
        <div class="mt-2 p-2 bg-gray-800/50 rounded-lg">
          <p class="font-semibold text-white">Recommended Actions:</p>
          <ul class="list-disc list-inside text-gray-400 text-xs">
            <li>Deploy NDRF rescue teams to distress hotspots immediately</li>
            <li>Accelerate food & water distribution to relief camps</li>
            <li>Dispatch engineering teams to assess and restore connectivity</li>
            <li>Continue monitoring social media streams for emerging signals</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- ADVANCED ANALYTICS -->
    <div class="grid grid-cols-3 gap-4 pb-4">
      <!-- Urgency Timeline -->
      <div class="glass-panel p-4">
        <h3 class="font-bold text-sm mb-2">⏳ URGENCY TIMELINE</h3>
        <canvas id="urgencyChart" height="100"></canvas>
      </div>
      <!-- Top Locations -->
      <div class="glass-panel p-4">
        <h3 class="font-bold text-sm mb-2">📍 TOP LOCATIONS</h3>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between"><span>brahmaputra</span><span class="font-mono">14</span></div>
          <div class="flex justify-between"><span>bhubaneswar</span><span class="font-mono">9</span></div>
          <div class="flex justify-between"><span>silchar</span><span class="font-mono">7</span></div>
          <div class="flex justify-between"><span>chennai</span><span class="font-mono">6</span></div>
          <div class="flex justify-between"><span>himachal</span><span class="font-mono">4</span></div>
        </div>
      </div>
      <!-- Key Insights -->
      <div class="glass-panel p-4 space-y-3">
        <h3 class="font-bold text-sm">🔎 Key Insights</h3>
        <div>
          <div class="text-xs text-gray-400">SIGNAL VELOCITY</div>
          <div class="text-2xl font-bold text-indigo-400">29<span class="text-sm">/min</span></div>
        </div>
        <div>
          <div class="text-xs text-gray-400">DOMINANT CATEGORY</div>
          <div class="text-2xl font-bold text-orange-400">13 <span class="text-sm">FOOD_WATER</span></div>
        </div>
        <div>
          <div class="text-xs text-gray-400">GEO COVERAGE</div>
          <div class="text-2xl font-bold text-green-400">89.7%</div>
        </div>
      </div>
    </div>

  </main>

  <!-- SCRIPTS -->
  <script>
    // --- LEAFLET MAP ---
    setTimeout(() => {
      const map = L.map('map', { zoomControl: false }).setView([21.5, 83.5], 5);
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap'
      }).addTo(map);

      const hotspots = [
        {name: 'Brahmaputra', lat: 26.2, lng: 91.0, type: 'rescue'},
        {name: 'Bhubaneswar', lat: 20.3, lng: 85.8, type: 'rescue'},
        {name: 'Silchar', lat: 24.8, lng: 92.8, type: 'rescue'},
        {name: 'Chennai', lat: 13.1, lng: 80.3, type: 'food'},
        {name: 'Himachal', lat: 31.1, lng: 77.2, type: 'food'},
        {name: 'Assam', lat: 26.4, lng: 92.8, type: 'infra'}
      ];

      hotspots.forEach(h => {
        const color = h.type === 'rescue' ? 'red' : h.type === 'food' ? 'orange' : 'yellow';
        L.circleMarker([h.lat, h.lng], {
          radius: 8,
          color: color,
          fillColor: color,
          fillOpacity: 0.6
        }).addTo(map).bindPopup('<b>' + h.name + '</b><br>' + h.type + ' signal');
      });
    }, 500);

    // --- CHART.JS: Signal Trend ---
    setTimeout(() => {
      const trendCtx = document.getElementById('trendChart');
      if (trendCtx) {
        new Chart(trendCtx.getContext('2d'), {
          type: 'line',
          data: {
            labels: ['-60','-50','-40','-30','-20','-10','0'],
            datasets: [{
              label: 'Signal count',
              data: [2,5,4,7,9,6,11],
              borderColor: '#6366f1',
              backgroundColor: 'rgba(99,102,241,0.2)',
              tension: 0.4,
              fill: true
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#9ca3af' } },
              x: { grid: { color: '#1e293b' }, ticks: { color: '#9ca3af' } }
            }
          }
        });
      }
    }, 500);

    // --- CHART.JS: Urgency Timeline ---
    setTimeout(() => {
      const urgencyCtx = document.getElementById('urgencyChart');
      if (urgencyCtx) {
        new Chart(urgencyCtx.getContext('2d'), {
          type: 'bar',
          data: {
            labels: ['Rescue','Food','Medical','Infra'],
            datasets: [{
              data: [9, 13, 3, 4],
              backgroundColor: ['#ef4444','#f97316','#3b82f6','#eab308']
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#9ca3af' } },
              x: { grid: { color: '#1e293b' }, ticks: { color: '#9ca3af' } }
            }
          }
        });
      }
    }, 500);

    // --- Category buttons toggle ---
    document.querySelectorAll('.category-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        const active = document.querySelectorAll('.category-btn.active').length;
        const activeSpan = Array.from(document.querySelectorAll('.text-sm.text-gray-400')).find(el => el.textContent.includes('◉'));
        if (activeSpan) activeSpan.textContent = '◉ ' + active + ' active';
      });
    });

    // Update clock
    function updateClock() {
      const now = new Date();
      const timeStr = now.toTimeString().split(' ')[0];
      const timeSpan = Array.from(document.querySelectorAll('.text-sm.text-gray-400')).find(el => el.textContent.includes('⏱'));
      if (timeSpan) timeSpan.textContent = '⏱ ' + timeStr;
    }
    setInterval(updateClock, 1000);
    updateClock();
  </script>
</body>
</html>
"""

# ======================== RENDER DASHBOARD =================================
components.html(html_content, height=1400, scrolling=True)
