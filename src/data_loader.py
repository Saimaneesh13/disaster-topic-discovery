"""
data_loader.py
--------------
Loads disaster tweet datasets.
Supports: HumAID (HuggingFace), CrisisLex CSV, or your own CSV.
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ── Sample fallback data (works with zero internet / API keys) ──────────────
SAMPLE_TWEETS = [
    "Flood in Chennai, need rescue immediately, water level rising fast #flood",
    "3 people trapped in building collapse in Odisha, please send help urgently",
    "Hospital running out of medicines in Assam flood area, need medical supplies",
    "Roads blocked between Patna and Muzaffarpur due to flooding, no vehicles passing",
    "Electricity down since 2 days in entire Sylhet district after cyclone",
    "Relief camp setup at school near Brahmaputra, 500 families displaced",
    "Drinking water not available in flood areas, urgent need for water tankers",
    "Rescue teams working in Kaziranga but many areas still unreachable by boat",
    "Earthquake felt across Delhi NCR, magnitude 5.8, minor damage reported",
    "Landslide blocks NH-44 in Himachal Pradesh, hundreds stuck on highway",
    "Cyclone Biparjoy making landfall in Gujarat, evacuation ongoing",
    "Fire breaks out in slum area Mumbai, 6 fire engines deployed",
    "Volunteers needed at relief camp in Agartala, food distribution starting",
    "Bridge washed away in Uttarakhand, 3 villages cut off from supplies",
    "People stranded on rooftops in Srinagar, army deployed for rescue ops",
    "Flood waters receding in Bihar plains, damage assessment underway",
    "School building collapsed in UP after heavy rains, students evacuated safely",
    "Power restoration in progress after cyclone hit Andhra coast",
    "Missing person report: elderly man missing since yesterday in flood area",
    "Donation drive for flood victims, drop clothes and food at community centre",
    "Helicopter rescue ops underway in Kedarnath valley, 200 pilgrims airlifted",
    "IMD issues red alert for heavy rainfall in Maharashtra and Goa next 48 hours",
    "Crocodile spotted in residential area due to floodwater in Vadodara",
    "Army column reaches flood hit Silchar after 3 days, supplies being distributed",
    "NDRF teams deployed in 12 districts of Assam as flood situation worsens",
]


def load_humaid(split: str = "train", max_samples: int = 5000) -> pd.DataFrame:
    """Load HumAID dataset from HuggingFace Hub."""
    print("📥 Loading HumAID dataset from HuggingFace...")
    try:
        ds = load_dataset("QCRI/HumAID", split=split)
        df = ds.to_pandas()[["tweet_text", "event_type"]].dropna()
        df = df.rename(columns={"tweet_text": "text", "event_type": "event"})
        df = df.head(max_samples)
        print(f"✅ Loaded {len(df)} tweets from HumAID")
        return df
    except Exception as e:
        print(f"⚠️  HumAID load failed: {e}")
        return None


def load_csv(filepath: str) -> pd.DataFrame:
    """Load tweets from a local CSV. Must have a 'text' column."""
    print(f"📥 Loading from {filepath}...")
    df = pd.read_csv(filepath)
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    df = df.dropna(subset=["text"])
    print(f"✅ Loaded {len(df)} tweets from CSV")
    return df


def load_sample_data() -> pd.DataFrame:
    """Returns built-in sample data — works offline, no API needed."""
    print("📥 Using built-in sample disaster tweets (offline mode)...")
    df = pd.DataFrame({"text": SAMPLE_TWEETS, "event": ["sample"] * len(SAMPLE_TWEETS)})
    print(f"✅ {len(df)} sample tweets loaded")
    return df


def get_data(source: str = "sample", filepath: str = None, max_samples: int = 5000) -> pd.DataFrame:
    """
    Master loader. source options:
      'humaid'   → HuggingFace HumAID dataset
      'csv'      → local CSV (provide filepath)
      'combined' → combined dataset (outputs/combined_dataset.csv)
      'sample'   → built-in sample tweets
    """

    if source == "humaid":
        df = load_humaid(max_samples=max_samples)
        if df is None:
            print("↩️  Falling back to sample data")
            df = load_sample_data()

    elif source == "csv":
        df = load_csv(filepath)

    # ✅ ADD THIS BLOCK
    elif source == "combined":
        print("📦 Loading combined dataset...")

        path = os.path.join(os.path.dirname(__file__), "..", "outputs", "combined_dataset.csv")

        if not os.path.exists(path):
            print("❌ Combined dataset not found. Run dataset_combiner.py first.")
            return load_sample_data()

        df = pd.read_csv(path)

        if "text" not in df.columns:
            raise ValueError("combined_dataset.csv must contain 'text' column")

        print(f"✅ Loaded {len(df)} tweets from combined dataset")

    else:
        df = load_sample_data()

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = get_data(source="sample")
    print(df.head())
