"""
Objective 4: test and validate the trained global model on the held-out
NIH ChestX-ray14 test split, and provide a helper to compare multiple
experiment variants (e.g. centralized vs FL vs BCFL+privacy) side by side.
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def evaluate_model(model, test_gen, unique_labels):
    """Run the model over the whole test generator and compute per-class
    AUC-ROC and F1 (multi-label). Returns a dict you can json.dump or plot."""
    y_true, y_pred = [], []
    for i in range(len(test_gen)):
        x_batch, y_batch = test_gen[i]
        if len(x_batch) == 0:
            continue
        preds = model.predict(x_batch, verbose=0)
        y_true.append(y_batch)
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    per_class = {}
    for idx, label in enumerate(unique_labels):
        entry = {}
        try:
            entry['auc'] = float(roc_auc_score(y_true[:, idx], y_pred[:, idx]))
        except ValueError:
            entry['auc'] = None  # only one class present in this split
        entry['f1'] = float(f1_score(y_true[:, idx], (y_pred[:, idx] > 0.5).astype(int),
                                      zero_division=0))
        per_class[label] = entry

    valid_aucs = [v['auc'] for v in per_class.values() if v['auc'] is not None]
    summary = {
        'mean_auc': float(np.mean(valid_aucs)) if valid_aucs else None,
        'mean_f1': float(np.mean([v['f1'] for v in per_class.values()])),
        'per_class': per_class,
    }
    return summary


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def compare_experiments(results_dict, save_path=None):
    """results_dict: {"centralized": {...}, "fl_no_privacy": {...}, "bcfl_hybrid": {...}}
    Each value is the dict returned by evaluate_model(). Prints a simple
    comparison table and optionally saves a bar chart."""
    print(f"{'Experiment':<25}{'Mean AUC':<12}{'Mean F1':<12}")
    for name, res in results_dict.items():
        auc = res.get('mean_auc')
        f1 = res.get('mean_f1')
        auc_s = f"{auc:.3f}" if auc is not None else "n/a"
        f1_s = f"{f1:.3f}" if f1 is not None else "n/a"
        print(f"{name:<25}{auc_s:<12}{f1_s:<12}")

    if save_path:
        import matplotlib.pyplot as plt
        names = list(results_dict.keys())
        aucs = [results_dict[n].get('mean_auc') or 0 for n in names]
        plt.figure(figsize=(6, 4))
        plt.bar(names, aucs)
        plt.ylabel('Mean AUC-ROC')
        plt.title('Privacy / FL trade-off comparison')
        plt.xticks(rotation=20)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved comparison plot to {save_path}")
