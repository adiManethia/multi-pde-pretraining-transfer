"""One example trajectory of each PDE family (initial field -> later fields).

    python plot_examples.py                 # writes results/pde_examples.png

Needs only NumPy and Matplotlib (no GPU, no PyTorch).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data

FAMILIES = [("heat", "Heat (pretraining)"),
            ("advection", "Advection (pretraining)"),
            ("reaction_diffusion", "Reaction-diffusion (pretraining)"),
            ("burgers", "Burgers (held out)")]
FRAMES = [0, 5, 10, 20]                      # t = 0, 0.25, 0.5, 1.0


def main(path="results/pde_examples.png", seed=3):
    x = np.arange(data.N_GRID) / data.N_GRID
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(FRAMES)))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), dpi=150, sharex=True)
    for ax, (pde, title) in zip(axes, FAMILIES):
        d = data.generate(pde, 1, seed)
        u = d["u"][0]
        for color, f in zip(colors, FRAMES):
            ax.plot(x, u[f], color=color, lw=1.8, label=f"t = {f * data.DT_OUT:.2f}")
        params = ", ".join(f"{k} = {v[0]:.3g}" for k, v in d["params"].items())
        ax.set_title(f"{title}\n{params}", fontsize=9)
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("u(x, t)")
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()