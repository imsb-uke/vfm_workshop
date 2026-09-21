"""Shared helpers for probing frozen VFM encodings saved by
notebooks/00_preparation/01_encodings: locating the parameterized
.npy/.npz/.csv output files, scoring predictions, and plotting confusion
matrices. Fitting the probes themselves (standardize + logistic regression /
kNN) is left to each analysis notebook rather than hidden here.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


def latest_matching(dir_: Path, pattern: str) -> Path:
    """Returns the most recently modified file in `dir_` matching glob `pattern`.

    01_encodings runs are parameterized (subset size, sample size, N_SLIDES...),
    so the exact output filename varies run to run -- glob for it rather than
    hardcoding a suffix.
    """
    matches = sorted(Path(dir_).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No file matching {pattern!r} in {dir_} -- run the matching "
            "01_encodings notebook first."
        )
    return matches[-1]


def probe_metrics(y_true, y_pred, labels=None) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """PANDA's own competition metric for ISUP grade (0-5, ordinal) -- penalizes
    predictions further from the true grade more than adjacent-grade misses."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def plot_confusion_matrix(cm: np.ndarray, class_names, ax, title: str = ""):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    return im
