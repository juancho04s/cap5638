"""
SHAP-based interpretability.

We compute mean(|SHAP|) over the test split for the hybrid model and save:
  - outputs/shap_summary.png    : top-20 feature importance bar chart
  - outputs/shap_values.npy     : raw SHAP matrix
"""

import os
import numpy as np

from .utils import log


def run_shap_analysis(model, X_test, out_dir="outputs", n_samples=200):
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        log("shap/matplotlib not installed — skipping SHAP analysis.")
        return

    os.makedirs(out_dir, exist_ok=True)
    Xs = X_test[:n_samples]

    # TreeExplainer is fastest for XGBoost; KernelExplainer otherwise.
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, Xs[:50])
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1]   # positive class

    np.save(os.path.join(out_dir, "shap_values.npy"), sv)
    mean_abs = np.abs(sv).mean(axis=0)
    log(f"  SHAP computed on {n_samples} samples; saved to {out_dir}/shap_values.npy")

    # Save a quick top-20 plot
    top = np.argsort(mean_abs)[::-1][:20]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(top))[::-1], mean_abs[top])
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels([f"feat_{i}" for i in top])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=150)
    plt.close()
