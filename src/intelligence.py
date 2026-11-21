"""
intelligence.py
---------------
Post-processing layer that turns raw topics into actionable intelligence.
- Urgency scoring (rescue > medical > food/water > infrastructure > info)
- Geo-entity extraction using spaCy NER
- Auto-labelling topics by dominant urgency category
"""

import re
import pandas as pd
import numpy as np


# ── Urgency keyword dictionary ───────────────────────────────────────────────
URGENCY_KEYWORDS = {
    "rescue": {
        "keywords": [
            "rescue", "trapped", "stranded", "help", "sos", "save", "stuck",
            "drowning", "missing", "urgent", "emergency", "danger", "rooftop",
        ],
        "score": 5,
        "color": "#FF2D2D",
        "icon": "🆘",
    },
    "medical": {
        "keywords": [
            "hospital", "injured", "medical", "medicine", "ambulance", "doctor",
            "treatment", "dead", "death", "casualty", "wound", "bleeding",
            "sick", "disease", "epidemic",
        ],
        "score": 4,
        "color": "#FF6B00",
        "icon": "🏥",
    },
    "food_water": {
        "keywords": [
            "food", "water", "hungry", "starvation", "drinking", "supply",
            "shortage", "relief", "ration", "grain", "aid", "distribute",
        ],
        "score": 3,
        "color": "#FFC107",
        "icon": "🍽️",
    },
    "infrastructure": {
        "keywords": [
            "road", "bridge", "electricity", "power", "flood", "damage",
            "building", "collapse", "highway", "blocked", "broken", "landslide",
        ],
        "score": 2,
        "color": "#2196F3",
        "icon": "🏗️",
    },
    "information": {
        "keywords": [
            "update", "alert", "warning", "news", "report", "confirm",
            "official", "government", "ndma", "imd", "notice",
        ],
        "score": 1,
        "color": "#4CAF50",
        "icon": "ℹ️",
    },
}


RESOLVED_OR_DRILL_PHRASES = (
    "mock drill", "rescue completed", "safely rescued", "all safe",
    "no casualties", "no damage", "alert lifted", "averted", "under control",
    "normalcy restored", "reopened"
)

def score_urgency(text: str) -> dict:
    """
    Returns urgency category + numeric score with negation/drill filtering.
    """
    text_lower = text.lower()

    # Downgrade resolved events or drills to 'information'
    if any(phrase in text_lower for phrase in RESOLVED_OR_DRILL_PHRASES):
        cat = URGENCY_KEYWORDS["information"]
        return {
            "urgency_category": "information",
            "urgency_score": cat["score"],
            "urgency_color": cat["color"],
            "urgency_icon": cat["icon"],
        }

    best_category = "information"
    best_score = 0

    for category, config in URGENCY_KEYWORDS.items():
        matches = sum(1 for kw in config["keywords"] if kw in text_lower)
        weighted = matches * config["score"]
        if weighted > best_score:
            best_score = weighted
            best_category = category

    cat = URGENCY_KEYWORDS[best_category]
    return {
        "urgency_category": best_category,
        "urgency_score": cat["score"],
        "urgency_color": cat["color"],
        "urgency_icon": cat["icon"],
    }


def add_urgency_scores(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Adds urgency columns to the DataFrame."""
    print("🚨 Scoring urgency levels...")
    urgency_data = df[text_col].apply(score_urgency)
    urgency_df = pd.DataFrame(list(urgency_data))
    df = pd.concat([df, urgency_df], axis=1)
    print(f"✅ Urgency scored. Distribution:")
    print(df["urgency_category"].value_counts().to_string())
    return df


# ── Geo extraction (Pan-India registry) ──────────────────────────────────────
try:
    from live_feed import PAN_INDIA_PLACES
except ImportError:
    from src.live_feed import PAN_INDIA_PLACES

INDIA_LOCATIONS = PAN_INDIA_PLACES

LOCATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(loc) for loc in INDIA_LOCATIONS) + r")\b",
    re.IGNORECASE
)


def extract_locations(text: str) -> list:
    """Extract location mentions from text using Pan-India regex."""
    matches = LOCATION_PATTERN.findall(text.lower())
    return list(set(matches))


def add_geo_tags(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Adds extracted location list to DataFrame, merging feed location hints."""
    df["locations"] = df[text_col].apply(extract_locations)
    if "location_hint" in df.columns:
        for idx, row in df.iterrows():
            hint = str(row.get("location_hint", "")).strip().lower()
            if hint and hint not in ("none", "nan", "") and hint not in row["locations"]:
                df.at[idx, "locations"] = row["locations"] + [hint]
    df["has_location"] = df["locations"].apply(lambda x: len(x) > 0)
    return df


# ── Topic-level aggregation ──────────────────────────────────────────────────
def aggregate_topic_intelligence(df: pd.DataFrame,
                                  topic_col: str = "topic") -> pd.DataFrame:
    """
    For each topic, compute:
    - dominant urgency category
    - average urgency score
    - top locations mentioned
    - tweet count
    """
    if topic_col not in df.columns:
        print("⚠️  No topic column found. Run topic modelling first.")
        return df

    records = []
    for topic_id, group in df.groupby(topic_col):
        if topic_id == -1:
            continue  # skip noise cluster

        dom_urgency = group["urgency_category"].mode()[0] if "urgency_category" in group else "unknown"
        avg_score = group["urgency_score"].mean() if "urgency_score" in group else 0

        all_locs = []
        if "locations" in group:
            for loc_list in group["locations"]:
                all_locs.extend(loc_list)
        top_locs = pd.Series(all_locs).value_counts().head(3).index.tolist()

        sample_tweets = group["clean_text"].head(3).tolist() if "clean_text" in group else []

        records.append({
            "topic_id": topic_id,
            "tweet_count": len(group),
            "dominant_urgency": dom_urgency,
            "avg_urgency_score": round(avg_score, 2),
            "top_locations": top_locs,
            "sample_tweets": sample_tweets,
        })

    result = pd.DataFrame(records).sort_values("avg_urgency_score", ascending=False)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "clean_text": [
            "rescue needed flood assam people trapped rooftop",
            "hospital medicine shortage in Bihar flood",
            "road blocked near kedarnath landslide uttarakhand",
            "drinking water not available Gujarat relief camp",
        ]
    })

    test_df = add_urgency_scores(test_df)
    test_df = add_geo_tags(test_df)
    print("\nFinal DataFrame:")
    print(test_df[["clean_text", "urgency_category", "urgency_score", "locations"]].to_string())
