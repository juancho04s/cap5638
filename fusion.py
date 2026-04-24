"""
Feature fusion.

Two strategies are supported:

  - Early fusion: concatenate per-modality vectors and feed one classifier.
    This is the default and is what we report in the main results table.

  - Late fusion (stacking): each modality trains its own classifier; the
    out-of-fold probabilities form a meta-feature vector that a second-stage
    LR combines. See `train.py::stacking_train` for the full implementation.

Each modality is L2-normalized independently before concatenation so that
high-dimensional BERT vectors do not overwhelm the 14-dim emotion vector.
"""

import numpy as np
from sklearn.preprocessing import normalize, StandardScaler


def fuse_features(feats: dict, split: dict, mode: str = "early"):
    train_idx, test_idx = split["train"], split["test"]

    blocks = []
    for name, X in feats.items():
        if X is None or X.size == 0:
            continue
        # L2-normalize per modality
        Xn = normalize(X, norm="l2", axis=1)
        # Standardize on the train split only
        scaler = StandardScaler(with_mean=False)
        scaler.fit(Xn[train_idx])
        blocks.append(scaler.transform(Xn))

    X_full = np.hstack(blocks).astype(np.float32)
    return X_full[train_idx], X_full[test_idx]
