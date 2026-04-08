import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Constants (must match model.py)
# -----------------------------
ACTIONS     = ["UP", "DOWN", "LEFT", "RIGHT"]
GRID_HEIGHT = 10
GRID_WIDTH  = 10
STATE_DIM   = 4
ACTION_DIM  = 4
HIDDEN_DIM  = 128
OUT_DIR     = "images"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Network (must match model.py)
# -----------------------------
def build_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, ACTION_DIM),
    ).to(DEVICE)

def load_checkpoint(path: str) -> tuple[nn.Sequential, int]:
    ckpt = torch.load(path, map_location=DEVICE)
    net  = build_network()
    net.load_state_dict(ckpt["online_state_dict"])
    net.eval()
    print(f"Loaded: {path}  (episode {ckpt['episode']}, ε={ckpt['epsilon']:.4f})")
    return net, ckpt["episode"]

# -----------------------------
# Core: query network over grid
# -----------------------------
def query_network(net: nn.Sequential, fruit_x: int, fruit_y: int) -> np.ndarray:
    """
    Feed every (snake_x, snake_y) position through the network with a fixed fruit.
    Returns shape (GRID_HEIGHT, GRID_WIDTH, 4) — Q-values for all actions.

    The whole grid is sent as one batch so there's only a single forward pass.
    """
    rows = []
    for sy in range(GRID_HEIGHT):
        for sx in range(GRID_WIDTH):
            rows.append([
                sx       / (GRID_WIDTH  - 1),   # snake_x normalised
                sy       / (GRID_HEIGHT - 1),   # snake_y normalised
                fruit_x  / (GRID_WIDTH  - 1),   # fruit_x normalised
                fruit_y  / (GRID_HEIGHT - 1),   # fruit_y normalised
            ])

    batch = torch.tensor(rows, dtype=torch.float32).to(DEVICE)  # (H*W, 4)

    with torch.no_grad():
        q_batch = net(batch).cpu().numpy()                       # (H*W, 4)

    return q_batch.reshape(GRID_HEIGHT, GRID_WIDTH, ACTION_DIM) # (H, W, 4)

# -----------------------------
# CSV export
# -----------------------------
def export_csv(net: nn.Sequential, fruit_x: int, fruit_y: int,
               path: str = f"{OUT_DIR}/q_table.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    q_grid = query_network(net, fruit_x, fruit_y)   # (H, W, 4)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snake_x", "snake_y",
                         "Q_UP", "Q_DOWN", "Q_LEFT", "Q_RIGHT",
                         "max_q", "best_action"])
        for sy in range(GRID_HEIGHT):
            for sx in range(GRID_WIDTH):
                if (sx, sy) == (fruit_x, fruit_y):
                    continue    # skip degenerate state (head == fruit)
                q_vals = q_grid[sy, sx]
                best   = ACTIONS[int(np.argmax(q_vals))]
                writer.writerow([sx, sy, *q_vals.tolist(),
                                 float(np.max(q_vals)), best])

    print(f"CSV saved → {path}")

# -----------------------------
# Heatmap
# -----------------------------
def plot_heatmaps(net: nn.Sequential, fruit_x: int, fruit_y: int,
                  episode: int, path: str = f"{OUT_DIR}/q_heatmap.png"):
    """
    5-panel layout: one subplot per action + MAX Q panel.
    Fruit cell is marked with a blue star.
    Cell annotations show the raw Q-value.
    Colormap is symmetric around 0 so negative = red, positive = green.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    q_grid = query_network(net, fruit_x, fruit_y)   # (H, W, 4)
    max_q  = q_grid.max(axis=2)                     # (H, W)

    panels = [(i, ACTIONS[i]) for i in range(4)] + [(-1, "MAX Q")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    for ax, (idx, title) in zip(axes, panels):
        grid = q_grid[:, :, idx] if idx >= 0 else max_q

        # Symmetric colormap so 0 = neutral yellow, + = green, - = red
        vmax = max(float(np.nanmax(np.abs(grid))), 1e-6)
        im   = ax.imshow(grid, cmap="RdYlGn", aspect="equal",
                         vmin=-vmax, vmax=vmax)

        # Per-cell value annotation
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                ax.text(c, r, f"{grid[r, c]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="black")

        # Mark fruit
        ax.plot(fruit_x, fruit_y, marker="*", color="blue",
                markersize=14, zorder=5, label="fruit")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("snake_x")
        ax.set_ylabel("snake_y")
        ax.set_xticks(range(GRID_WIDTH))
        ax.set_yticks(range(GRID_HEIGHT))
        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(
        f"DQN Q-value Heatmaps  |  fruit=({fruit_x},{fruit_y})  |  episode {episode}",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved → {path}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="models/dqn.pt", help="Checkpoint path")
    parser.add_argument("--fruit",   nargs=2, type=int, default=[4, 4],
                        metavar=("FX", "FY"), help="Fixed fruit position (default: 4 4)")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    args = parser.parse_args()

    OUT_DIR  = args.out_dir
    fx, fy   = args.fruit
    os.makedirs(OUT_DIR, exist_ok=True)

    net, episode = load_checkpoint(args.ckpt)

    plot_heatmaps(net, fx, fy, episode,
                  path=f"{OUT_DIR}/q_heatmap_{episode}.png")

    export_csv(net, fx, fy,
               path=f"{OUT_DIR}/q_table_{episode}.csv")