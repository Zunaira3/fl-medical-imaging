"""Result plots: federated vs centralized (and local-only) convergence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402

_STYLES = {
    "Federated (FedAvg)": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    "Centralized": {"color": "tab:green", "marker": "s", "linestyle": "--"},
    "Local-only (client 0)": {"color": "tab:red", "marker": "^", "linestyle": ":"},
}


def plot_results(
    histories: dict[str, list[dict[str, float]]],
    path: str | Path,
    title: str,
) -> Path:
    """Save a 3-panel figure (test accuracy, test loss, macro-F1) against communication round."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("accuracy", "Test accuracy"), ("loss", "Test loss"), ("macro_f1", "Test macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (key, label) in zip(axes, panels):
        for name, hist in histories.items():
            rounds = [h["round"] for h in hist]
            values = [h[key] for h in hist]
            ax.plot(rounds, values, label=name, linewidth=1.8, markersize=4, **_STYLES.get(name, {}))
        ax.set_xlabel("Communication round (= local_epochs epochs over the pooled data)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
