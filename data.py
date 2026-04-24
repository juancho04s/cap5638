"""
Dataset loading.

We support FakeNewsNet (PolitiFact + GossipCop subsets) and LIAR. The
FakeNewsNet repository ships news articles + Twitter cascades; LIAR is
text-only (we synthesize a trivial 1-node 'cascade' for graph features).

If the raw data is not present under ./data/, we fall back to a small
bundled toy split so the pipeline still runs end-to-end. Replace this
with the real loader when you place the datasets on disk.
"""

import os
import json
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import log


DATA_DIR = os.environ.get("PR_DATA_DIR", "data")


def _load_fakenewsnet() -> pd.DataFrame:
    """Expected layout:
        data/fakenewsnet/politifact_real.csv   (id, title, text)
        data/fakenewsnet/politifact_fake.csv
        data/fakenewsnet/cascades/<id>.json    (retweet tree)
    """
    base = os.path.join(DATA_DIR, "fakenewsnet")
    real_p = os.path.join(base, "politifact_real.csv")
    fake_p = os.path.join(base, "politifact_fake.csv")
    if not (os.path.exists(real_p) and os.path.exists(fake_p)):
        log("FakeNewsNet not found on disk — using synthetic toy data.")
        return _toy_dataset(n=400)
    real = pd.read_csv(real_p); real["label"] = 0
    fake = pd.read_csv(fake_p); fake["label"] = 1
    df = pd.concat([real, fake], ignore_index=True)
    df["cascade_path"] = df["id"].apply(
        lambda i: os.path.join(base, "cascades", f"{i}.json"))
    return df.dropna(subset=["text"]).reset_index(drop=True)


def _load_liar() -> pd.DataFrame:
    base = os.path.join(DATA_DIR, "liar")
    train_p = os.path.join(base, "train.tsv")
    if not os.path.exists(train_p):
        log("LIAR not found on disk — using synthetic toy data.")
        return _toy_dataset(n=400)
    cols = ["id", "label", "statement", "subject", "speaker", "job",
            "state", "party", "barely_true_ct", "false_ct", "half_true_ct",
            "mostly_true_ct", "pants_fire_ct", "context"]
    train = pd.read_csv(train_p, sep="\t", names=cols)
    valid = pd.read_csv(os.path.join(base, "valid.tsv"), sep="\t", names=cols)
    test  = pd.read_csv(os.path.join(base,  "test.tsv"), sep="\t", names=cols)
    df = pd.concat([train, valid, test], ignore_index=True)
    fake_labels = {"false", "pants-fire", "barely-true"}
    df["label"] = df["label"].apply(lambda x: 1 if x in fake_labels else 0)
    df = df.rename(columns={"statement": "text"})
    df["cascade_path"] = None
    return df.dropna(subset=["text"]).reset_index(drop=True)


def _toy_dataset(n: int = 400) -> pd.DataFrame:
    """Tiny synthetic dataset so the pipeline runs without external data."""
    rng = random.Random(0)
    fake_phrases = ["BREAKING shocking truth", "you won't believe", "MUST READ now",
                    "they don't want you to know", "wake up sheeple"]
    real_phrases = ["according to a report", "researchers found", "in a statement today",
                    "data released by", "officials confirmed"]
    rows = []
    for i in range(n):
        label = rng.randint(0, 1)
        seed = fake_phrases if label == 1 else real_phrases
        text = " ".join(rng.choices(seed, k=3) +
                        rng.choices(["the policy", "the study", "the event",
                                     "the leader", "the city"], k=4))
        rows.append({"id": i, "text": text, "label": label,
                     "cascade_path": None})
    return pd.DataFrame(rows)


def load_dataset(name: str, quick: bool = False, seed: int = 42) -> Dict:
    if name == "fakenewsnet":
        df = _load_fakenewsnet()
    elif name == "liar":
        df = _load_liar()
    else:
        raise ValueError(name)

    if quick:
        df = df.sample(frac=0.10, random_state=seed).reset_index(drop=True)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20,
        stratify=df["label"], random_state=seed,
    )
    return {
        "df":      df,
        "split":   {"train": train_idx, "test": test_idx},
        "y_train": df["label"].values[train_idx],
        "y_test":  df["label"].values[test_idx],
    }
