"""
Text features.

Two modes:
  - 'tfidf' : sklearn TfidfVectorizer (uni+bi grams), used by the LR baseline.
  - 'bert'  : 768-dim [CLS] vector from `bert-base-uncased`. We freeze BERT and
              pool [CLS] only — fine-tuning is left for future work.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils import log, Timer


def _bert_encode(texts, batch_size=16, model_name="bert-base-uncased"):
    """Encode a list of strings into 768-dim [CLS] vectors."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        log("transformers/torch not installed — falling back to hashed-text vectors.")
        # Deterministic hash-based fallback so the pipeline still runs.
        rng = np.random.RandomState(0)
        proj = rng.randn(2**16, 768).astype(np.float32) * 0.05
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                out[i] += proj[hash(w) % 2**16]
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = np.zeros((len(texts), 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state[:, 0, :]   # [CLS]
            out[i:i + batch_size] = h.cpu().numpy()
    return out


def extract_text_features(data, mode="bert"):
    df = data["df"]
    texts = df["text"].astype(str).tolist()

    with Timer(f"text/{mode}"):
        if mode == "tfidf":
            vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)
            X = vec.fit_transform(texts).toarray().astype(np.float32)
            log(f"  TF-IDF shape: {X.shape}")
            return X
        elif mode == "bert":
            X = _bert_encode(texts)
            log(f"  BERT [CLS] shape: {X.shape}")
            return X
        else:
            raise ValueError(mode)
