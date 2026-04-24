"""
Emotion features.

We compute, per post:
  - 8 NRC emotion intensities (anger, fear, disgust, joy, sadness,
    surprise, trust, anticipation), normalized by token count
  - 2 NRC sentiment polarities (positive, negative)
  - VADER compound sentiment + (pos, neu, neg)
  - 2 derived intensity metrics:
      * total negative emotional load = anger + fear + disgust + sadness
      * affective polarization        = |pos - neg| / (pos + neg + eps)

A small inline NRC subset is bundled so the script runs without
downloading the full lexicon. For full reproducibility, drop the
official NRC Emotion Lexicon at data/nrc_lexicon.txt.
"""

import os
import re
from collections import defaultdict

import numpy as np

from ..utils import log, Timer


_NRC_PATH = os.path.join(os.environ.get("PR_DATA_DIR", "data"), "nrc_lexicon.txt")

# Tiny inline fallback (used if the official lexicon is absent).
_INLINE_NRC = {
    "anger":   {"hate", "rage", "furious", "outrage", "attack", "destroy"},
    "fear":    {"terror", "afraid", "panic", "danger", "threat", "warning"},
    "disgust": {"disgusting", "vile", "corrupt", "filthy", "scandal"},
    "joy":     {"happy", "joy", "love", "wonderful", "celebrate", "great"},
    "sadness": {"sad", "tragic", "loss", "mourn", "grief", "cry"},
    "surprise":{"shocking", "unbelievable", "stunning", "sudden", "breaking"},
    "trust":   {"confirm", "verified", "report", "according", "research", "official"},
    "anticipation": {"expect", "soon", "upcoming", "predict", "forecast"},
    "positive":{"good", "great", "wonderful", "success", "win"},
    "negative":{"bad", "wrong", "fail", "loss", "crisis", "problem"},
}

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _load_nrc():
    if os.path.exists(_NRC_PATH):
        d = defaultdict(set)
        with open(_NRC_PATH, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[2] == "1":
                    d[parts[1]].add(parts[0].lower())
        return d
    log("  NRC lexicon not found — using inline fallback (10 emotions, ~60 words).")
    return _INLINE_NRC


def _vader_scores(texts):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return np.array([[sia.polarity_scores(t)[k] for k in ("compound", "pos", "neu", "neg")]
                         for t in texts], dtype=np.float32)
    except ImportError:
        log("  VADER not installed — using zeros for sentiment scores.")
        return np.zeros((len(texts), 4), dtype=np.float32)


def extract_emotion_features(data):
    df = data["df"]
    texts = df["text"].astype(str).tolist()
    nrc = _load_nrc()
    cats = list(nrc.keys())

    with Timer("emotion"):
        n = len(texts); k = len(cats)
        E = np.zeros((n, k), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w.lower() for w in _TOKEN_RE.findall(t)]
            if not toks: continue
            for j, c in enumerate(cats):
                hits = sum(1 for w in toks if w in nrc[c])
                E[i, j] = hits / len(toks)

        V = _vader_scores(texts)

        # Derived intensity metrics
        try:
            ang = E[:, cats.index("anger")];  fea = E[:, cats.index("fear")]
            dis = E[:, cats.index("disgust")]; sad = E[:, cats.index("sadness")]
            pos = E[:, cats.index("positive")]; neg = E[:, cats.index("negative")]
        except ValueError:
            ang = fea = dis = sad = pos = neg = np.zeros(n, dtype=np.float32)

        neg_load   = (ang + fea + dis + sad).reshape(-1, 1)
        polariz    = (np.abs(pos - neg) / (pos + neg + 1e-6)).reshape(-1, 1)

        X = np.hstack([E, V, neg_load, polariz]).astype(np.float32)
        log(f"  Emotion feature shape: {X.shape}  ({k} NRC + 4 VADER + 2 derived)")
        return X
