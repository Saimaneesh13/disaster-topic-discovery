"""
embedder.py
-----------
Generates sentence embeddings using Sentence-BERT.
Uses multilingual model so Hindi + English both work.
Caches embeddings to disk so you don't recompute every run.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# Multilingual model — works for Hindi + English tweets
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Fallback: English-only but slightly better quality
ENGLISH_MODEL = "all-MiniLM-L6-v2"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class TweetEmbedder:
    """
    Wraps SentenceTransformer with caching + batch processing.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"🤖 Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed(self,
              texts: list,
              batch_size: int = 64,
              cache_key: str = None,
              show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of strings into dense vectors.
        If cache_key provided, saves/loads from disk.
        """
        # ── Try loading from cache ──
        if cache_key:
            cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
            if os.path.exists(cache_path):
                print(f"💾 Loading cached embeddings from {cache_path}")
                return np.load(cache_path)

        # ── Generate embeddings ──
        print(f"⚙️  Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity friendly
        )

        # ── Save to cache ──
        if cache_key:
            np.save(cache_path, embeddings)
            print(f"💾 Embeddings cached to {cache_path}")

        print(f"✅ Embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_dataframe(self,
                        df: pd.DataFrame,
                        text_col: str = "clean_text",
                        cache_key: str = "embeddings") -> np.ndarray:
        """Convenience method — embeds text column of a DataFrame."""
        texts = df[text_col].tolist()
        return self.embed(texts, cache_key=cache_key)


if __name__ == "__main__":
    embedder = TweetEmbedder()
    texts = [
        "Flood rescue needed in Chennai immediately",
        "बाढ़ में फंसे लोग मदद चाहिए",
        "Earthquake in Delhi, buildings damaged",
    ]
    vecs = embedder.embed(texts, cache_key=None)
    print(f"Shape: {vecs.shape}")  # (3, 384)
    # Quick similarity check
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(vecs)
    print("Cosine similarity matrix:")
    print(sim.round(3))
