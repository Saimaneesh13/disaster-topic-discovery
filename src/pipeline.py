"""
pipeline.py
-----------
Master pipeline that runs all stages end-to-end:
  1. Load data
  2. Preprocess
  3. Embed
  4. Topic model (BERTopic + LDA baseline)
  5. Add intelligence (urgency + geo)
  6. Save outputs
  
Run: python pipeline.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── Local modules ─────────────────────────────────────────────────────────────
from data_loader   import get_data
from preprocessor  import preprocess
from embedder      import TweetEmbedder
from topic_modeller import fit_bertopic, fit_lda, get_topic_summary
from intelligence  import add_urgency_scores, add_geo_tags, aggregate_topic_intelligence

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pipeline(
    data_source: str = "sample",        # 'sample' | 'humaid' | 'csv'
    csv_path: str = None,               # path if data_source='csv'
    max_samples: int = 2000,
    run_lda: bool = True,
    filter_lang: bool = False,          # False for sample data (all English)
) -> dict:

    print("=" * 60)
    print("  DISASTER TOPIC DISCOVERY PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Stage 1: Load ──────────────────────────────────────────────────────
    print("\n[1/5] DATA LOADING")
    df = get_data(source=data_source, filepath=csv_path, max_samples=max_samples)

    df = df.sample(n=min(5000, len(df)), random_state=42)

    # ── Stage 2: Preprocess ────────────────────────────────────────────────
    print("\n[2/5] PREPROCESSING")
    df = preprocess(df, filter_lang=filter_lang)

    if len(df) < 5:
        print("❌ Not enough data after preprocessing. Exiting.")
        return {}

    texts = df["clean_text"].tolist()

    # ── Stage 3: Embed ─────────────────────────────────────────────────────
    print("\n[3/5] EMBEDDING")
    embedder = TweetEmbedder()
    embeddings = embedder.embed(texts, cache_key=None)

    # ── Stage 4: Topic Modelling ───────────────────────────────────────────
    print("\n[4/5] TOPIC MODELLING")

    # BERTopic (main)
    bt_model, topics, probs, topic_info = fit_bertopic(texts, embeddings)
    df["topic"] = topics

    topic_summary = get_topic_summary(bt_model)
    print("\n📋 Discovered Topics:")
    print(topic_summary.to_string(index=False))

    # LDA baseline
    lda_topics_df = None
    if run_lda:
        _, _, _, lda_topics_df = fit_lda(texts, num_topics=min(8, len(df) // 3))

    # ── Stage 5: Intelligence layer ────────────────────────────────────────
    print("\n[5/5] INTELLIGENCE SCORING")
    df = add_urgency_scores(df)
    df = add_geo_tags(df)

    intel = aggregate_topic_intelligence(df)
    print("\n🎯 Topic Intelligence Summary:")
    print(intel[["topic_id", "tweet_count", "dominant_urgency",
                  "avg_urgency_score", "top_locations"]].to_string(index=False))

    # ── Save outputs ───────────────────────────────────────────────────────
    print("\n💾 Saving outputs...")

    df.to_csv(os.path.join(OUTPUT_DIR, "tweets_with_topics.csv"), index=False)
    topic_summary.to_csv(os.path.join(OUTPUT_DIR, "topic_summary.csv"), index=False)
    intel.to_csv(os.path.join(OUTPUT_DIR, "topic_intelligence.csv"), index=False)

    if lda_topics_df is not None:
        lda_topics_df.to_csv(os.path.join(OUTPUT_DIR, "lda_topics.csv"), index=False)

    # Save BERTopic model
    model_path = os.path.join(OUTPUT_DIR, "bertopic_model")
    bt_model.save(model_path, serialization="safetensors", save_ctfidf=True)
    print(f"✅ BERTopic model saved to {model_path}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE ✅")
    print(f"  Outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    return {
        "df": df,
        "model": bt_model,
        "topic_summary": topic_summary,
        "intelligence": intel,
        "embeddings": embeddings,
        "lda_topics": lda_topics_df,
    }


if __name__ == "__main__":
    results = run_pipeline(
        data_source="combined",
        max_samples=5000,   # 🔥 CHANGE THIS
        run_lda=False,      # 🔥 optional (faster)
        filter_lang=False,
    )