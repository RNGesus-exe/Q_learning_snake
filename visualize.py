"""
Load a saved Q-table and produce:
  - q_table.csv   : one row per state, columns for each action + max_q + best_action
  - q_heatmap.png : 2x2 heatmap grid (one subplot per action) + max-Q heatmap
"""

import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
GRID_HEIGHT = 10
GRID_WIDTH  = 10
Q_TABLE_PATH = "models/q_table.pkl"
OUT_DIR = "images"


def load_q_table(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def export_csv(q_table: dict, path: str = f"{OUT_DIR}/q_table.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snake_x", "snake_y", "Q_UP", "Q_DOWN", "Q_LEFT", "Q_RIGHT", "max_q", "best_action"])
        for state, q_vals in sorted(q_table.items()):
            x, y = state
            best = ACTIONS[int(np.argmax(q_vals))]
            writer.writerow([x, y, *q_vals.tolist(), float(np.max(q_vals)), best])
    print(f"CSV saved → {path}")


def build_grid(q_table: dict, action_idx: int) -> np.ndarray:
    """Build a GRID_HEIGHT × GRID_WIDTH matrix for one action (or max across actions)."""
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)
    for (x, y), q_vals in q_table.items():
        xi, yi = int(x), int(y)
        if 0 <= yi < GRID_HEIGHT and 0 <= xi < GRID_WIDTH:
            grid[yi, xi] = q_vals[action_idx] if action_idx >= 0 else float(np.max(q_vals))
    return grid


def plot_heatmaps(q_table: dict, path: str = f"{OUT_DIR}/q_heatmap.png"):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    panels = [(i, ACTIONS[i]) for i in range(4)] + [(-1, "MAX Q")]

    for ax, (idx, title) in zip(axes, panels):
        grid = build_grid(q_table, idx)
        im = ax.imshow(grid, cmap="RdYlGn", aspect="equal")

        # Annotate each cell
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                val = grid[r, c]
                text = f"{val:.2f}" if not np.isnan(val) else "—"
                ax.text(c, r, text, ha="center", va="center", fontsize=7)

        ax.set_title(title)
        ax.set_xlabel("snake_x")
        ax.set_ylabel("snake_y")
        ax.set_xticks(range(GRID_WIDTH))
        ax.set_yticks(range(GRID_HEIGHT))
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Q-Table Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Heatmap saved → {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    q_table = load_q_table(Q_TABLE_PATH)
    print(f"Loaded Q-table: {len(q_table)} states")
    export_csv(q_table)
    plot_heatmaps(q_table)
