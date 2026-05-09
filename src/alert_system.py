"""
alert_system.py
---------------
Monitors topic + urgency data and fires alerts when thresholds are crossed.
Used by the dashboard to show live warning banners.
"""

import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import List


# ── Alert config ──────────────────────────────────────────────────────────────
ALERT_RULES = {
    "rescue": {
        "threshold": 3,          # fire alert if ≥ 3 rescue tweets in batch
        "level": "CRITICAL",
        "color": "#FF2D2D",
        "icon": "🆘",
        "message": "High rescue demand detected",
    },
    "medical": {
        "threshold": 3,
        "level": "HIGH",
        "color": "#FF6B00",
        "icon": "🏥",
        "message": "Medical emergency signals rising",
    },
    "food_water": {
        "threshold": 4,
        "level": "MODERATE",
        "color": "#FFC107",
        "icon": "🍽️",
        "message": "Food & water shortage reports increasing",
    },
    "infrastructure": {
        "threshold": 4,
        "level": "MODERATE",
        "color": "#2196F3",
        "icon": "🏗️",
        "message": "Infrastructure damage reports detected",
    },
}

LEVEL_ORDER = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}


@dataclass
class Alert:
    level: str
    category: str
    message: str
    locations: List[str]
    tweet_count: int
    color: str
    icon: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return self.__dict__

    def headline(self) -> str:
        loc_str = ", ".join(self.locations[:3]) if self.locations else "multiple areas"
        return f"{self.icon} [{self.level}] {self.message} — {loc_str} ({self.tweet_count} signals)"


def check_alerts(df: pd.DataFrame) -> List[Alert]:
    """
    Scans the tweet DataFrame and returns active Alert objects.
    df must have: urgency_category, locations columns.
    """
    alerts = []

    if "urgency_category" not in df.columns:
        return alerts

    for category, rule in ALERT_RULES.items():
        subset = df[df["urgency_category"] == category]
        count = len(subset)

        if count >= rule["threshold"]:
            # Collect top mentioned locations
            all_locs = []
            if "locations" in subset.columns:
                for loc_list in subset["locations"]:
                    if isinstance(loc_list, list):
                        all_locs.extend(loc_list)
            top_locs = (
                pd.Series(all_locs).value_counts().head(3).index.tolist()
                if all_locs else []
            )

            alerts.append(Alert(
                level=rule["level"],
                category=category,
                message=rule["message"],
                locations=top_locs,
                tweet_count=count,
                color=rule["color"],
                icon=rule["icon"],
            ))

    # Sort: CRITICAL first
    alerts.sort(key=lambda a: LEVEL_ORDER.get(a.level, 0), reverse=True)
    return alerts


def get_severity_label(alerts: List[Alert]) -> tuple:
    """
    Returns overall disaster severity label + color based on active alerts.
    Returns: (label, color, description)
    """
    if not alerts:
        return ("NORMAL", "#4CAF50", "No significant distress signals detected")

    top_level = alerts[0].level
    if top_level == "CRITICAL":
        return ("SEVERE", "#FF2D2D", "Multiple critical rescue signals active")
    elif top_level == "HIGH":
        return ("HIGH", "#FF6B00", "High-urgency signals detected across categories")
    elif top_level == "MODERATE":
        return ("MODERATE", "#FFC107", "Moderate distress activity in affected areas")
    else:
        return ("LOW", "#4CAF50", "Low-level distress signals, monitoring active")


if __name__ == "__main__":
    # Test
    test_df = pd.DataFrame({
        "urgency_category": [
            "rescue", "rescue", "rescue", "rescue",
            "medical", "medical",
            "food_water", "infrastructure",
        ],
        "locations": [
            ["Chennai"], ["Assam"], ["Bihar"], ["Patna"],
            ["Mumbai"], ["Odisha"],
            ["Gujarat"], ["Uttarakhand"],
        ],
    })

    alerts = check_alerts(test_df)
    for a in alerts:
        print(a.headline())

    label, color, desc = get_severity_label(alerts)
    print(f"\nOverall Severity: {label} — {desc}")
