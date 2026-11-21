import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_runs(log_dir="./scratch/logs", metrics=("train_loss", "val_loss", "grad_norm"), save_path="comparison.png"):
    files = sorted(Path(log_dir).glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {log_dir}")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for f in files:
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        for ax, metric in zip(axes, metrics):
            pts = [(r.get("step", i), r[metric]) for i, r in enumerate(rows) if metric in r]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, label=f.stem, linewidth=1.5)

    for ax, metric in zip(axes, metrics):
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    plot_runs()
