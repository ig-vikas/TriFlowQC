"""
Visualization utilities for qutrit quantum computing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, TYPE_CHECKING


def plot_loss_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss/metrics over epochs.

    Parameters
    ----------
    history : dict
        Keys: 'loss', 'val_loss', and any metric names.
    title : str
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    if "loss" in history:
        ax.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Metrics
    ax = axes[1]
    metric_keys = [k for k in history if k not in ("loss", "val_loss")]
    for key in metric_keys:
        ax.plot(history[key], label=key)
    if metric_keys:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a confusion matrix for classification results.

    Parameters
    ----------
    y_true : ndarray
    y_pred : ndarray
    labels : list of str or None
    title : str
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    n_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.flat, y_pred.flat):
        cm[int(t), int(p)] += 1

    if labels is None:
        labels = [f"|{i}⟩" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_state_city(
    rho: NDArray[np.complexfloating],
    title: str = "Density Matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    3D bar plot of a density matrix (city plot).

    Parameters
    ----------
    rho : ndarray
    title : str
    save_path : str or None
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    d = rho.shape[0]
    xpos, ypos = np.meshgrid(range(d), range(d))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    fig = plt.figure(figsize=(10, 8))

    # Real part
    ax1 = fig.add_subplot(121, projection="3d")
    values_real = np.real(rho).flatten()
    colors_real = ["blue" if v >= 0 else "red" for v in values_real]
    ax1.bar3d(xpos, ypos, zpos, 0.8, 0.8, values_real, color=colors_real, alpha=0.7)
    ax1.set_title("Re(ρ)")
    ax1.set_xlabel("Row")
    ax1.set_ylabel("Col")

    # Imaginary part
    ax2 = fig.add_subplot(122, projection="3d")
    values_imag = np.imag(rho).flatten()
    colors_imag = ["green" if v >= 0 else "orange" for v in values_imag]
    ax2.bar3d(xpos, ypos, zpos, 0.8, 0.8, values_imag, color=colors_imag, alpha=0.7)
    ax2.set_title("Im(ρ)")
    ax2.set_xlabel("Row")
    ax2.set_ylabel("Col")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_bloch_sphere_qutrit(
    bloch_vector: NDArray[np.float64],
    title: str = "Qutrit Bloch Vector",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise the 8-dimensional qutrit Bloch vector.

    Since 8D cannot be directly visualised, we show:
    1. Bar chart of all 8 components.
    2. 3D projection onto first 3 components (SU(2) subgroup).

    Parameters
    ----------
    bloch_vector : ndarray (8,)
    title : str
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of all 8 components
    ax = axes[0]
    labels = [f"n{i+1}" for i in range(8)]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in bloch_vector]
    ax.bar(labels, bloch_vector, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Component")
    ax.set_ylabel("Value")
    ax.set_title("Bloch Vector Components")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # 3D projection onto first 3 components
    ax3d = fig.add_subplot(122, projection="3d")
    n1, n2, n3 = bloch_vector[0], bloch_vector[1], bloch_vector[2]

    # Draw unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(x, y, z, alpha=0.05, color="gray")

    # Plot state point
    ax3d.scatter([n1], [n2], [n3], color="red", s=100, zorder=5)
    ax3d.plot([0, n1], [0, n2], [0, n3], color="red", linewidth=2)

    ax3d.set_xlabel("n₁")
    ax3d.set_ylabel("n₂")
    ax3d.set_zlabel("n₃")
    ax3d.set_title("SU(2) Projection")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_probabilities(
    probs: NDArray[np.floating],
    n_qutrits: int,
    title: str = "Measurement Probabilities",
    top_k: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar plot of measurement probabilities.

    Parameters
    ----------
    probs : ndarray
    n_qutrits : int
    title : str
    top_k : int
    save_path : str or None
    """
    import matplotlib.pyplot as plt

    dim = len(probs)
    labels = []
    for i in range(dim):
        digits = []
        val = i
        for _ in range(n_qutrits):
            digits.append(str(val % 3))
            val //= 3
        labels.append("".join(reversed(digits)))

    # Sort by probability, take top_k
    indices = np.argsort(probs)[::-1][:top_k]
    top_labels = [labels[i] for i in indices]
    top_probs = probs[indices]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(top_probs)), top_probs, color="#2ca02c", alpha=0.8)
    ax.set_xticks(range(len(top_probs)))
    ax.set_xticklabels(top_labels, rotation=45)
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
