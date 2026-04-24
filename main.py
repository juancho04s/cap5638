"""
CAP 5638 Final Project Hybrid Emotion-Aware Misinformation Detection
======================================================================
Authors : Nikolas Rivera, Juan Ignacio Medina
Course  : CAP 5638 Pattern Recognition (Spring 2026)
Instr.  : Dr. Yushun Dong

This is the main entry point. It runs the entire pipeline end-to-end:

    1. Load (or download) FakeNewsNet / LIAR
    2. Extract textual features (BERT [CLS] + TF-IDF baseline)
    3. Extract emotional features (NRC lexicon + VADER sentiment)
    4. Extract graph features (NetworkX cascade metrics)
    5. Fuse modalities and train MLP / XGBoost classifiers
    6. Run ablation, evaluation, and SHAP interpretability
    7. Save metrics + figures to ./outputs

USAGE
-----
    python main.py --dataset fakenewsnet --config hybrid
    python main.py --dataset liar        --config text_only
    python main.py --dataset fakenewsnet --config all   # runs every ablation row

Use --quick to run on a 10% subset for smoke testing.
"""

import argparse
import os
import time
import json
import warnings

warnings.filterwarnings("ignore")

from src.data import load_dataset
from src.features.text_features import extract_text_features
from src.features.emotion_features import extract_emotion_features
from src.features.graph_features import extract_graph_features
from src.fusion import fuse_features
from src.train import train_and_evaluate
from src.explain import run_shap_analysis
from src.utils import set_seed, save_metrics, log


CONFIGS = ["tfidf_lr", "text_only", "text_emotion", "text_graph", "hybrid"]


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid misinformation detector")
    p.add_argument("--dataset", choices=["fakenewsnet", "liar"], default="fakenewsnet")
    p.add_argument("--config", choices=CONFIGS + ["all"], default="hybrid")
    p.add_argument("--quick", action="store_true", help="Use 10% subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    p.add_argument("--explain", action="store_true", help="Run SHAP after training")
    return p.parse_args()


def run_one(config_name, data, args):
    log(f"=== Running config: {config_name} ===")

    use_text    = config_name in ("text_only", "text_emotion", "text_graph", "hybrid")
    use_emotion = config_name in ("text_emotion", "hybrid")
    use_graph   = config_name in ("text_graph", "hybrid")
    use_tfidf   = config_name == "tfidf_lr"

    # ---- Feature extraction -------------------------------------------------
    feats = {}
    if use_tfidf:
        feats["text"] = extract_text_features(data, mode="tfidf")
    elif use_text:
        feats["text"] = extract_text_features(data, mode="bert")

    if use_emotion:
        feats["emotion"] = extract_emotion_features(data)

    if use_graph:
        feats["graph"] = extract_graph_features(data)

    # ---- Fusion -------------------------------------------------------------
    X_train, X_test = fuse_features(feats, data["split"])

    # ---- Train + evaluate ---------------------------------------------------
    classifier = "logreg" if use_tfidf else ("xgboost" if config_name == "hybrid" else "mlp")
    metrics, model = train_and_evaluate(
        X_train, X_test, data["y_train"], data["y_test"],
        classifier=classifier, seed=args.seed,
    )
    metrics["config"]    = config_name
    metrics["dataset"]   = args.dataset
    metrics["n_train"]   = len(data["y_train"])
    metrics["n_test"]    = len(data["y_test"])
    log(f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")

    # ---- Explainability -----------------------------------------------------
    if args.explain and config_name == "hybrid":
        run_shap_analysis(model, X_test, out_dir=args.out)

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    log(f"Dataset: {args.dataset}    Configs: {args.config}    Quick: {args.quick}")
    t0 = time.time()

    data = load_dataset(args.dataset, quick=args.quick, seed=args.seed)
    log(f"Loaded {len(data['y_train'])} train / {len(data['y_test'])} test samples")

    configs_to_run = CONFIGS if args.config == "all" else [args.config]
    all_metrics = []
    for cfg in configs_to_run:
        m = run_one(cfg, data, args)
        all_metrics.append(m)

    save_metrics(all_metrics, os.path.join(args.out, f"metrics_{args.dataset}.json"))
    log(f"Total runtime: {time.time() - t0:.1f}s")
    log(f"Saved metrics to {args.out}/metrics_{args.dataset}.json")


if __name__ == "__main__":
    main()
