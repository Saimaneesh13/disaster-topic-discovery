"""
preprocessor.py
---------------
Cleans raw tweet text for embedding + topic modelling.
Steps: URL removal → mention/hashtag cleaning → emoji conversion
       → lowercasing → language detection → deduplication
"""

import re
import pandas as pd
import emoji
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

# Make langdetect deterministic
DetectorFactory.seed = 42


# ── Regex patterns ────────────────────────────────────────────────────────────
URL_RE      = re.compile(r"http\S+|www\.\S+")
MENTION_RE  = re.compile(r"@\w+")
HASHTAG_RE  = re.compile(r"#(\w+)")   # keep word, drop #
WHITESPACE  = re.compile(r"\s+")
SPECIAL_RE  = re.compile(r"[^\w\s]")


def remove_urls(text: str) -> str:
    return URL_RE.sub("", text)

def clean_mentions(text: str) -> str:
    return MENTION_RE.sub("", text)

def clean_hashtags(text: str) -> str:
    """Remove # but keep the word — hashtag words are informative."""
    return HASHTAG_RE.sub(r"\1", text)

def convert_emojis(text: str) -> str:
    """Convert emoji to text description: 🔥 → fire, 🆘 → SOS."""
    return emoji.demojize(text, delimiters=(" ", " "))

def normalize_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()

def clean_tweet(text: str) -> str:
    """Full cleaning pipeline for a single tweet."""
    text = remove_urls(text)
    text = clean_mentions(text)
    text = clean_hashtags(text)
    text = convert_emojis(text)
    text = normalize_whitespace(text)
    return text


def detect_language(text: str) -> str:
    """Returns ISO 639-1 language code ('en', 'hi', etc.) or 'unknown'."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def filter_languages(df: pd.DataFrame, keep: list = ["en", "hi"]) -> pd.DataFrame:
    """Keep only tweets in specified languages."""
    print(f"🌐 Detecting languages (keeping: {keep})...")
    tqdm.pandas()
    df["lang"] = df["clean_text"].progress_apply(detect_language)
    before = len(df)
    df = df[df["lang"].isin(keep)].reset_index(drop=True)
    print(f"✅ Language filter: {before} → {len(df)} tweets")
    return df


def deduplicate(df: pd.DataFrame, col: str = "clean_text", threshold: float = 0.95) -> pd.DataFrame:
    """Simple exact + near-duplicate removal using hash."""
    before = len(df)
    # Exact duplicates
    df = df.drop_duplicates(subset=[col])
    print(f"🧹 Dedup: {before} → {len(df)} tweets (removed {before - len(df)} exact duplicates)")
    return df.reset_index(drop=True)


def preprocess(df: pd.DataFrame,
               text_col: str = "text",
               filter_lang: bool = True,
               keep_langs: list = ["en", "hi"],
               dedup: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Returns df with added 'clean_text' and 'lang' columns.
    """
    print("\n🔧 Starting preprocessing...")

    # Step 1: Clean text
    tqdm.pandas(desc="Cleaning tweets")
    df["clean_text"] = df[text_col].astype(str).progress_apply(clean_tweet)

    # Step 2: Drop empty rows
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    # Step 3: Language detection
    if filter_lang:
        df = filter_languages(df, keep=keep_langs)
    else:
        df["lang"] = "unknown"

    # Step 4: Deduplication
    if dedup:
        df = deduplicate(df)

    print(f"✅ Preprocessing complete. {len(df)} tweets ready.\n")
    return df


if __name__ == "__main__":
    # Quick test
    sample = pd.DataFrame({
        "text": [
            "Flood in Chennai! Need rescue now 🆘 http://t.co/abc #flood @NDMA",
            "बाढ़ में फंसे लोग, मदद चाहिए #Assam",
            "RT @user: same tweet repeated",
            "same tweet repeated",
        ]
    })
    result = preprocess(sample, filter_lang=False)
    print(result[["text", "clean_text", "lang"]])
