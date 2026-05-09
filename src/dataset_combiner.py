"""
dataset_combiner.py
-------------------
Combines multiple disaster NLP datasets into one unified CSV.

Supported datasets (add whichever you download):
  1. HumAID        — 77k tweets, 10 labels, 22 disasters
  2. CrisisLex T26 — 28k tweets, 26 crisis events
  3. CrisisLex T6  — 60k tweets, 6 major disasters
  4. TREC-IS       — 25k tweets, information types
  5. COVID-CHS     — crisis tweets during COVID
  6. build_dataset — our local generated set (always included)

HOW TO GET EACH DATASET — see DATASETS_DOWNLOAD.md
Run: python dataset_combiner.py
"""

import os, glob, pandas as pd
from build_dataset import get_or_build

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT   = os.path.join(os.path.dirname(__file__), "..", "outputs", "combined_dataset.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Unified urgency label map ─────────────────────────────────────────────────
URGENCY_MAP = {
    # HumAID labels
    "rescue_volunteering_or_donation_effort": "rescue",
    "requests_or_urgent_needs":               "rescue",
    "affected_individuals":                   "rescue",
    "displaced_people_and_evacuations":       "rescue",
    "missing_or_found_people":                "rescue",
    "injured_or_dead_people":                 "medical",
    "infrastructure_and_utility_damage":      "infrastructure",
    "caution_and_advice":                     "information",
    "other_relevant_information":             "information",
    "sympathy_and_support":                   "information",
    "not_humanitarian":                       "information",

    # CrisisLex labels
    "on-topic":                               "rescue",
    "off-topic":                              "information",
    "related and informative":                "information",
    "related - but not informative":          "information",
    "not related":                            "information",
    "cant judge":                             "information",

    # TREC-IS labels
    "CallToAction-Donations":                 "rescue",
    "CallToAction-MovePeople":                "rescue",
    "CallToAction-Volunteer":                 "rescue",
    "Report-EmergingThreats":                 "rescue",
    "Report-NewSubEvent":                     "information",
    "Report-Official":                        "information",
    "Report-Weather":                         "information",
    "Report-Infrastructure":                  "infrastructure",
    "Report-Casualties":                      "medical",
    "Report-MultimediaShare":                 "information",
    "Sentiment-Sympathy":                     "information",
    "Other-Irrelevant":                       "information",

    # Generic
    "flood":          "rescue",
    "earthquake":     "rescue",
    "hurricane":      "rescue",
    "wildfire":       "rescue",
    "tornado":        "rescue",
    "typhoon":        "rescue",
    "rescue":         "rescue",
    "medical":        "medical",
    "infrastructure": "infrastructure",
    "information":    "information",
}

def map_urgency(label: str) -> str:
    if not isinstance(label, str):
        return "information"
    return URGENCY_MAP.get(label.strip().lower(),
           URGENCY_MAP.get(label.strip(), "information"))


# ── Loaders for each dataset format ──────────────────────────────────────────

def load_humaid_csv(folder: str) -> pd.DataFrame:
    """
    HumAID: TSV files with columns: tweet_id, tweet_text, class_label
    Expected location: data/humaid/
    """
    pattern = os.path.join(folder, "**", "*.tsv")
    files   = glob.glob(pattern, recursive=True)
    if not files:
        pattern = os.path.join(folder, "*.tsv")
        files   = glob.glob(pattern)
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep="\t")
            text_col  = next((c for c in df.columns if "text"  in c.lower()), None)
            label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)
            if text_col is None:
                continue
            row = pd.DataFrame({"text": df[text_col]})
            if label_col:
                row["label"] = df[label_col]
            row["source"] = "humaid"
            row["event"]  = os.path.basename(f).replace(".tsv","")
            dfs.append(row)
        except Exception as e:
            print(f"  ⚠️  Skipping {f}: {e}")

    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True)
    print(f"  📂 HumAID: {len(result)} tweets from {len(files)} files")
    return result


def load_crisislex_t26(folder: str) -> pd.DataFrame:
    """
    CrisisLex T26: CSV files, columns vary by file
    Expected location: data/crisislex_t26/
    Download: https://github.com/sajao/CrisisLex  (CrisisLexT26 folder)
    """
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
            # CrisisLex T26 columns: tweet_id, tweet_text, label
            text_col  = next((c for c in df.columns if "text"  in c.lower()), None)
            label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)
            if text_col is None:
                continue
            row = pd.DataFrame({"text": df[text_col]})
            if label_col:
                row["label"] = df[label_col]
            row["source"] = "crisislex_t26"
            row["event"]  = os.path.basename(os.path.dirname(f))
            dfs.append(row)
        except Exception as e:
            print(f"  ⚠️  Skipping {f}: {e}")

    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True)
    print(f"  📂 CrisisLex T26: {len(result)} tweets from {len(files)} files")
    return result


def load_crisislex_t6(folder: str) -> pd.DataFrame:
    """
    CrisisLex T6: CSV files per event
    Expected location: data/crisislex_t6/
    Download: https://github.com/sajao/CrisisLex  (CrisisLexT6 folder)
    """
    return load_crisislex_t26(folder)   # same format, reuse loader


def load_trec_is(folder: str) -> pd.DataFrame:
    """
    TREC Incident Streams: JSON files
    Expected location: data/trec_is/
    Download: https://www.dcs.gla.ac.uk/~richardm/TREC_IS/
    """
    files = glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder, "*.json"))
    if not files:
        return None

    import json
    dfs = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            # TREC-IS format: list of {postID, text, informationTypes, ...}
            if isinstance(data, list):
                rows = []
                for item in data:
                    text  = item.get("text", item.get("tweet", {}).get("text", ""))
                    label = ""
                    info_types = item.get("informationTypes", [])
                    if info_types:
                        label = info_types[0] if isinstance(info_types[0], str) else str(info_types[0])
                    if text:
                        rows.append({"text": text, "label": label,
                                     "source": "trec_is", "event": os.path.basename(f)})
                if rows:
                    dfs.append(pd.DataFrame(rows))
        except Exception as e:
            print(f"  ⚠️  Skipping {f}: {e}")

    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True)
    print(f"  📂 TREC-IS: {len(result)} tweets")
    return result


def load_generic_csv(folder: str, source_name: str) -> pd.DataFrame:
    """
    Generic loader for any CSV with a text column.
    Put any other datasets in data/<folder_name>/
    """
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
            text_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ["text","tweet","content","message"])),
                None
            )
            if text_col is None:
                continue
            label_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ["label","class","category","type"])),
                None
            )
            row = pd.DataFrame({"text": df[text_col]})
            if label_col:
                row["label"] = df[label_col]
            row["source"] = source_name
            row["event"]  = os.path.basename(f).replace(".csv","")
            dfs.append(row)
        except Exception:
            continue

    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True)
    print(f"  📂 {source_name}: {len(result)} rows")
    return result


# ── MASTER COMBINER ───────────────────────────────────────────────────────────

DATASET_LOADERS = [
    ("humaid",        os.path.join(DATA_DIR, "humaid"),        load_humaid_csv),
    ("crisislex_t26", os.path.join(DATA_DIR, "crisislex_t26"), load_crisislex_t26),
    ("crisislex_t6",  os.path.join(DATA_DIR, "crisislex_t6"),  load_crisislex_t6),
    ("trec_is",       os.path.join(DATA_DIR, "trec_is"),        load_trec_is),
]


def combine_all(include_local: bool = True,
                min_text_length: int = 15,
                dedup: bool = True,
                save: bool = True) -> pd.DataFrame:
    """
    Loads all available datasets from data/ folder, combines them.
    Always includes the local generated dataset as base.
    Skips any dataset folder that doesn't exist yet.
    """
    all_parts = []

    # Always include local generated dataset
    if include_local:
        print("🔨 Loading local generated dataset...")
        local = get_or_build()
        local = local.rename(columns={"tweet_text": "text", "event_type": "event"})
        local["source"] = "local_generated"
        all_parts.append(local[["text", "event", "source", "urgency"]])
        print(f"  ✅ Local: {len(local)} tweets")

    # Try each external dataset
    print("\n📦 Scanning data/ folder for external datasets...")
    found_any = False
    for name, folder, loader_fn in DATASET_LOADERS:
        if not os.path.isdir(folder):
            print(f"  ⏭️  {name}: folder not found (data/{name}/) — skipping")
            continue
        if not os.listdir(folder):
            print(f"  ⏭️  {name}: folder is empty — skipping")
            continue

        print(f"  Loading {name}...")
        df = loader_fn(folder)
        if df is None:
            print(f"  ⚠️  {name}: no valid files found")
            continue

        # Normalize columns
        df = df[df["text"].notna()].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() >= min_text_length]

        # Map urgency
        if "label" in df.columns:
            df["urgency"] = df["label"].apply(map_urgency)
        elif "urgency" not in df.columns:
            df["urgency"] = "information"

        if "event" not in df.columns:
            df["event"] = name

        all_parts.append(df[["text", "event", "source", "urgency"]])
        found_any = True

    # Also scan for any generic CSV in data/other/
    other_dir = os.path.join(DATA_DIR, "other")
    if os.path.isdir(other_dir) and os.listdir(other_dir):
        print("  Loading data/other/ (generic CSVs)...")
        generic = load_generic_csv(other_dir, "custom")
        if generic is not None:
            if "urgency" not in generic.columns:
                generic["urgency"] = "information"
            all_parts.append(generic[["text", "event", "source", "urgency"]])
            found_any = True

    if not found_any:
        print("\n  💡 No external datasets found yet.")
        print("     See DATASETS_DOWNLOAD.md for download instructions.")
        print("     Using local generated dataset only.\n")

    # Combine
    combined = pd.concat(all_parts, ignore_index=True)
    combined["text"] = combined["text"].str.strip()
    combined = combined[combined["text"].str.len() >= min_text_length]

    # Dedup
    if dedup:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
        print(f"\n🧹 Deduplication: {before} → {len(combined)} tweets")

    # Stats
    print(f"\n{'='*50}")
    print(f"✅ COMBINED DATASET: {len(combined):,} tweets")
    print(f"   Sources:  {combined['source'].value_counts().to_dict()}")
    print(f"   Urgency:  {combined['urgency'].value_counts().to_dict()}")
    print(f"   Events:   {combined['event'].nunique()} unique events")
    print(f"{'='*50}")

    if save:
        combined.to_csv(OUTPUT, index=False)
        print(f"💾 Saved to {OUTPUT}")

    return combined


if __name__ == "__main__":
    df = combine_all()
    print(f"\nSample rows:")
    for _, row in df.sample(min(5, len(df)), random_state=42).iterrows():
        print(f"  [{row['source']:20}] [{row['urgency']:15}] {row['text'][:75]}")
