"""
Circuit visualisation utilities.

Provides ASCII and matplotlib circuit diagram rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


def draw_ascii(circuit: "QutritCircuit") -> str:
    """Return an ASCII string representation of the circuit."""
    return circuit._draw_ascii()


def draw_matplotlib(circuit: "QutritCircuit") -> None:
    """
    Draw the circuit using matplotlib.

    Opens a matplotlib figure window.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        raise ImportError("matplotlib is required for graphical circuit drawing.")

    n = circuit.num_qutrits
    instructions = circuit.instructions

    fig, ax = plt.subplots(figsize=(max(8, len(instructions) * 1.5), max(3, n * 0.8)))

    # Draw wires
    x_end = len(instructions) * 2 + 3
    for i in range(n):
        y = n - 1 - i
        ax.plot([0, x_end], [y, y], "k-", linewidth=0.8)
        ax.text(-0.5, y, f"q{i}", ha="right", va="center", fontsize=10)

    # Draw gates
    for step, (name, _, targets, param_indices) in enumerate(instructions):
        x = step * 2 + 2
        if len(targets) == 1:
            y = n - 1 - targets[0]
            rect = patches.FancyBboxPatch(
                (x - 0.4, y - 0.3), 0.8, 0.6,
                boxstyle="round,pad=0.05",
                facecolor="lightblue", edgecolor="black"
            )
            ax.add_patch(rect)
            ax.text(x, y, name, ha="center", va="center", fontsize=7)
        elif len(targets) == 2:
            y0 = n - 1 - targets[0]
            y1 = n - 1 - targets[1]
            ax.plot([x, x], [min(y0, y1), max(y0, y1)], "k-", linewidth=1.5)
            ax.plot(x, y0, "ko", markersize=8)
            rect = patches.FancyBboxPatch(
                (x - 0.4, y1 - 0.3), 0.8, 0.6,
                boxstyle="round,pad=0.05",
                facecolor="lightyellow", edgecolor="black"
            )
            ax.add_patch(rect)
            ax.text(x, y1, name, ha="center", va="center", fontsize=7)

    # Measurements
    for m in circuit.measurements:
        x = len(instructions) * 2 + 2
        y = n - 1 - m
        rect = patches.FancyBboxPatch(
            (x - 0.3, y - 0.3), 0.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor="lightyellow", edgecolor="black"
        )
        ax.add_patch(rect)
        ax.text(x, y, "M", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-1, x_end + 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("QutritCircuit", fontsize=12)
    plt.tight_layout()
    plt.show()
