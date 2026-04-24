# CAP 5638 Final Project Hybrid Emotion-Aware Misinformation Detection

**Authors:** Nikolas Rivera, Juan Ignacio Medina
**Course:** CAP 5638 Pattern Recognition (Spring 2026)
**Instructor:** Dr. Yushun Dong

This repository implements a multi-modal pattern recognition framework that
fuses **textual**, **emotional**, and **graph-based structural** features for
binary misinformation classification on FakeNewsNet and LIAR.

---

## 1. Repository layout

```
code/
├── main.py                          # CLI entry point — runs the full pipeline
├── requirements.txt
├── README.md
└── src/
    ├── data.py                      # FakeNewsNet + LIAR loaders
    ├── fusion.py                    # Early / late feature fusion
    ├── train.py                     # LR / MLP / XGBoost training + eval
    ├── explain.py                   # SHAP-based interpretability
    ├── utils.py                     # Logging, seeding, I/O
    └── features/
        ├── text_features.py         # BERT [CLS] + TF-IDF
        ├── emotion_features.py      # NRC lexicon + VADER + derived metrics
        └── graph_features.py        # NetworkX cascade metrics
figures/                             # Pre-generated figures used in report
outputs/                             # Created at runtime — metrics + plots
```

---

## 2. Installation

Tested on **Python 3.10** (Linux / macOS / Windows).

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional but recommended:

```bash
python -m nltk.downloader punkt
```

If `transformers` / `torch` are unavailable, `text_features.py` falls back to
a deterministic hashed-text projection and the pipeline still runs end-to-end.

---

## 3. Datasets

We use two public datasets. **The pipeline runs without them** — when the
files are missing, a small synthetic toy split is used so the code is
verifiable end-to-end.

### FakeNewsNet (PolitiFact + GossipCop)

Repository: <https://github.com/KaiDMML/FakeNewsNet>

Place under `data/fakenewsnet/`:

```
data/fakenewsnet/
├── politifact_real.csv     # columns: id, title, text
├── politifact_fake.csv
└── cascades/<id>.json      # {user, parent, t} per node
```

### LIAR

Paper: Wang (2017), *"Liar, Liar Pants on Fire"*

Place under `data/liar/`:

```
data/liar/
├── train.tsv
├── valid.tsv
└── test.tsv
```

### Optional: NRC Emotion Lexicon

If `data/nrc_lexicon.txt` is missing, a small inline subset (~60 words) is
used. For the full lexicon, request access at:
<https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm>

---

## 4. Running

### Smoke test (synthetic data, ~10 seconds)

```bash
python main.py --dataset fakenewsnet --config hybrid --quick
```

### Full ablation (all five configurations)

```bash
python main.py --dataset fakenewsnet --config all
python main.py --dataset liar        --config all
```

### Single configuration with SHAP interpretability

```bash
python main.py --dataset fakenewsnet --config hybrid --explain
```

### CLI flags

| Flag         | Values                                                         | Default       |
|--------------|----------------------------------------------------------------|---------------|
| `--dataset`  | `fakenewsnet`, `liar`                                          | `fakenewsnet` |
| `--config`   | `tfidf_lr`, `text_only`, `text_emotion`, `text_graph`, `hybrid`, `all` | `hybrid` |
| `--quick`    | run on 10% subset                                              | off           |
| `--explain`  | run SHAP after training (hybrid only)                          | off           |
| `--seed`     | random seed                                                    | `42`          |
| `--out`      | output directory                                               | `outputs`     |

---

## 5. Reproducing the report results

```bash
# Reproduces Tables 1–2 and Figures 2, 3, 5 from the report
python main.py --dataset fakenewsnet --config all --explain
python main.py --dataset liar        --config all
```

Expected runtime on a single CPU machine: ~25–40 minutes for FakeNewsNet
(BERT encoding dominates), ~10–15 minutes for LIAR. With a CUDA GPU the
BERT step drops to under 2 minutes.

Outputs are written to `outputs/`:

```
outputs/
├── metrics_fakenewsnet.json   # all five configs, all metrics
├── metrics_liar.json
├── shap_values.npy
└── shap_summary.png
```

---

## 6. Citing borrowed code

- **HuggingFace `transformers`** (Apache 2.0) — BERT encoder.
- **scikit-learn** (BSD-3) — TF-IDF, LR, MLP, metrics.
- **XGBoost** (Apache 2.0) — gradient-boosted trees.
- **NetworkX** (BSD-3) — graph construction and structural metrics.
- **vaderSentiment** (MIT) — sentiment analyzer.
- **shap** (MIT) — SHAP explainer.

No external code was copy-pasted; all source files in `src/` were written
specifically for this project.

---
