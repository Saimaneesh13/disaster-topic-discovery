"""
summariser.py
-------------
Generates human-readable disaster intelligence summaries.
Two modes:
  1. Rule-based (default, works offline, instant)
  2. LLM-powered via Anthropic API (optional, richer output)
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List


# ── Rule-based summary ────────────────────────────────────────────────────────
def rule_based_summary(df: pd.DataFrame, alerts) -> str:
    """
    Generates a structured text summary from tweet data + alerts.
    No external API needed.
    """
    if df.empty:
        return "No data available for summary."

    total = len(df)
    timestamp = datetime.now().strftime("%d %b %Y, %H:%M")

    # Count by urgency
    counts = df["urgency_category"].value_counts().to_dict() if "urgency_category" in df else {}

    # Top locations
    all_locs = []
    if "locations" in df.columns:
        for loc_list in df["locations"]:
            if isinstance(loc_list, list):
                all_locs.extend(loc_list)
    top_locs = pd.Series(all_locs).value_counts().head(5).index.tolist() if all_locs else []

    # Build summary text
    lines = [f"📋 DISASTER SITUATION REPORT — {timestamp}", ""]

    # Overall severity
    rescue_count  = counts.get("rescue", 0)
    medical_count = counts.get("medical", 0)
    food_count    = counts.get("food_water", 0)
    infra_count   = counts.get("infrastructure", 0)

    if rescue_count >= 4:
        lines.append(f"🚨 CRITICAL: Mass rescue operations required — {rescue_count} distress signals detected.")
    elif rescue_count >= 2:
        lines.append(f"⚠️  URGENT: {rescue_count} rescue requests active across affected areas.")

    if medical_count >= 3:
        lines.append(f"🏥 Medical emergency: {medical_count} signals indicate healthcare system under stress.")

    if food_count >= 3:
        lines.append(f"🍽️  Food & water crisis: {food_count} shortage reports from relief camps.")

    if infra_count >= 2:
        lines.append(f"🏗️  Infrastructure: {infra_count} reports of road/bridge/power damage.")

    lines.append("")

    if top_locs:
        lines.append(f"📍 Most affected areas: {', '.join(top_locs)}")

    if alerts:
        lines.append(f"🔔 Active alerts: {len(alerts)} ({', '.join(a.level for a in alerts[:3])})")

    lines.append(f"📊 Total signals analysed: {total}")
    lines.append("")
    lines.append("Recommended Actions:")

    if rescue_count >= 2:
        lines.append("  • Deploy NDRF rescue teams to distress hotspots immediately")
    if medical_count >= 2:
        lines.append("  • Activate field medical units and ensure medicine supply chains")
    if food_count >= 2:
        lines.append("  • Accelerate food & water distribution to relief camps")
    if infra_count >= 2:
        lines.append("  • Dispatch engineering teams to assess and restore connectivity")

    lines.append("  • Continue monitoring social media streams for emerging signals")

    return "\n".join(lines)


# ── LLM-powered summary (optional) ──────────────────────────────────────────
def llm_summary(df: pd.DataFrame, alerts, api_key: str = None) -> str:
    """
    Generates a richer summary using Gemini API.
    Falls back to rule-based if API call fails.
    """
    try:
        import google.generativeai as genai

        # Build context from data
        counts = df["urgency_category"].value_counts().to_dict() if "urgency_category" in df else {}

        all_locs = []
        if "locations" in df.columns:
            for loc_list in df["locations"]:
                if isinstance(loc_list, list):
                    all_locs.extend(loc_list)
        top_locs = pd.Series(all_locs).value_counts().head(5).index.tolist() if all_locs else []

        sample_tweets = df["clean_text"].head(8).tolist() if "clean_text" in df.columns else []

        alert_lines = [a.headline() for a in alerts] if alerts else ["None"]

        prompt = f"""You are a disaster response intelligence analyst. 
Based on the following social media signal data, write a concise 3-4 sentence 
situation report for emergency response coordinators. 
Be factual, urgent where needed, and end with 1-2 actionable recommendations.

SIGNAL COUNTS:
{json.dumps(counts, indent=2)}

TOP AFFECTED LOCATIONS:
{', '.join(top_locs) if top_locs else 'Unknown'}

ACTIVE ALERTS:
{chr(10).join(alert_lines)}

SAMPLE TWEETS:
{chr(10).join(f'- {t}' for t in sample_tweets[:5])}

Write the situation report now:"""

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return "🤖 AI-Generated Summary\n\n" + response.text

    except Exception as e:
        print(f"⚠️  LLM summary failed ({e}), using rule-based fallback")
        return rule_based_summary(df, alerts)


def get_summary(df: pd.DataFrame, alerts, use_llm: bool = False, api_key: str = None) -> str:
    """
    Master summary function.
    use_llm=True requires GEMINI_API_KEY in environment or passed as api_key.
    """
    if use_llm:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if key:
            return llm_summary(df, alerts, api_key=key)
        # Missing API key warning is handled in the UI to prevent console spam

    return rule_based_summary(df, alerts)


if __name__ == "__main__":
    from alert_system import check_alerts

    test_df = pd.DataFrame({
        "clean_text": [
            "rescue needed flood assam", "trapped rooftop patna",
            "hospital medicines shortage", "food water not available",
            "road blocked bridge washed", "electricity down cyclone",
        ],
        "urgency_category": ["rescue", "rescue", "medical", "food_water", "infrastructure", "infrastructure"],
        "locations": [["Assam"], ["Patna"], ["Bihar"], ["Gujarat"], ["Uttarakhand"], ["Odisha"]],
    })

    alerts = check_alerts(test_df)
    summary = get_summary(test_df, alerts, use_llm=False)
    print(summary)
