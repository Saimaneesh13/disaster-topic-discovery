"""
topic_modeller.py
-----------------
Fits BERTopic (main) and LDA (baseline) on tweet embeddings.
Also handles dynamic topic modelling for time-based drift detection.
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
import warnings
warnings.filterwarnings("ignore")


# ── BERTopic configuration ─────────────────────────────────────────────────
def build_bertopic_model(
    n_neighbors: int = 10,
    n_components: int = 5,
    min_cluster_size: int = 3,
    min_topic_size: int = 3,
) -> BERTopic:
    """
    Builds a configured BERTopic model.
    Parameters tuned for disaster tweet data (short, noisy text).
    """
    # UMAP: reduce 384-dim embeddings to manageable space
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN: density-based clustering (handles noise well)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorizer: ignore very common/rare words
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),   # unigrams + bigrams
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )

    return topic_model


# ── Main BERTopic fitter ────────────────────────────────────────────────────
def fit_bertopic(texts: list, embeddings: np.ndarray) -> tuple:
    """
    Fit BERTopic on texts + precomputed embeddings.
    Returns: (model, topics, probabilities, topic_info_df)
    """
    print("\n🔍 Fitting BERTopic model...")
    model = build_bertopic_model()
    topics, probs = model.fit_transform(texts, embeddings)

    topic_info = model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topics else 0

    print(f"\n✅ BERTopic complete:")
    print(f"   Topics found : {n_topics}")
    print(f"   Outlier docs : {n_outliers} (Topic -1 = noise, normal for HDBSCAN)")

    return model, topics, probs, topic_info


# ── Topic info formatter ────────────────────────────────────────────────────
def get_topic_summary(model: BERTopic) -> pd.DataFrame:
    """Returns a clean summary DataFrame of all discovered topics."""
    info = model.get_topic_info()
    info = info[info["Topic"] != -1].copy()

    # Get top keywords per topic
    keywords = []
    for topic_id in info["Topic"]:
        words = model.get_topic(topic_id)
        if words:
            kw = ", ".join([w for w, _ in words[:5]])
        else:
            kw = "N/A"
        keywords.append(kw)

    info["keywords"] = keywords
    info = info.rename(columns={"Topic": "topic_id", "Count": "tweet_count", "Name": "topic_name"})
    return info[["topic_id", "topic_name", "tweet_count", "keywords"]]


# ── LDA Baseline ────────────────────────────────────────────────────────────
def fit_lda(texts: list, num_topics: int = 8) -> tuple:
    """
    Baseline LDA topic model using Gensim.
    Returns: (lda_model, corpus, dictionary, topics_df)
    """
    print(f"\n📚 Fitting LDA baseline ({num_topics} topics)...")

    # Tokenize and remove stopwords
    tokenized = [
        [w for w in doc.lower().split() if w not in STOPWORDS and len(w) > 2]
        for doc in texts
    ]
    tokenized = [doc for doc in tokenized if doc]  # remove empty

    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = models.LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10,
        random_state=42,
    )

    # Format topics
    topics = []
    for i in range(num_topics):
        words = lda.show_topic(i, topn=5)
        topics.append({
            "topic_id": i,
            "keywords": ", ".join([w for w, _ in words]),
            "weights": [round(p, 3) for _, p in words],
        })

    topics_df = pd.DataFrame(topics)
    print(f"✅ LDA complete. {num_topics} topics extracted.")
    return lda, corpus, dictionary, topics_df


# ── Dynamic (time-windowed) topic modelling ─────────────────────────────────
def fit_dynamic_topics(model: BERTopic,
                       texts: list,
                       timestamps: list,
                       nr_bins: int = 5) -> object:
    """
    Fits dynamic topic model to track how topics evolve over time.
    timestamps: list of datetime objects aligned with texts.
    """
    print(f"\n📈 Fitting dynamic topics across {nr_bins} time bins...")
    topics_over_time = model.topics_over_time(
        texts,
        timestamps,
        nr_bins=nr_bins,
        evolution_tuning=True,
        global_tuning=True,
    )
    print("✅ Dynamic topic modelling complete.")
    return topics_over_time


if __name__ == "__main__":
    # Quick smoke test with tiny data
    sample_texts = [
        "flood rescue needed immediately water rising",
        "help needed flood water level high",
        "earthquake damage buildings collapsed rescue",
        "earthquake tremors felt relief needed",
        "hospital needs medicine supplies shortage",
        "medical aid required flood victims",
        "power outage electricity down cyclone",
        "roads blocked bridge washed away",
    ]
    # Fake embeddings for test
    fake_embeddings = np.random.rand(len(sample_texts), 384).astype(np.float32)

    model, topics, probs, info = fit_bertopic(sample_texts, fake_embeddings)
    summary = get_topic_summary(model)
    print("\nTopic Summary:")
    print(summary.to_string(index=False))
