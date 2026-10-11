"""Regenerate the constructed teaching figures for background-beginner.md."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dora-background-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np


OUTPUT = Path(__file__).resolve().parent
BLUE = "#2864B5"
ORANGE = "#B95718"
GREEN = "#23734F"
INK = "#1F2933"
MUTED = "#596573"
GRID = "#DFE4EA"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(name)


def arrow(ax, start, end, color=INK, width=2.2):
    ax.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "color": color, "lw": width,
        "mutation_scale": 17, "shrinkA": 0, "shrinkB": 0,
    })


def box(ax, x, y, width, height, text, color):
    rect = FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        facecolor="white", edgecolor=color, linewidth=1.7)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=12, color=INK)


def geometry(ax, xlim, ylim, xlabel="Coordinate 1", ylabel="Coordinate 2"):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.grid(color=GRID, alpha=0.45, lw=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def lora_path():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 3.2))
    ax.axis("off")
    ax.set_title("LoRA keeps the pretrained path and adds a trainable path", pad=15)
    ax.text(0.3, 1.6, "$x$\ninput", ha="center", va="center")
    box(ax, 3.0, 2.15, 2.0, 0.7, "$W_0$\nfrozen", MUTED)
    box(ax, 1.9, 0.4, 1.65, 0.7, "$A$\ntrainable", BLUE)
    box(ax, 5.0, 0.4, 1.65, 0.7, "$B$\ntrainable", BLUE)
    ax.add_patch(Circle((8.0, 1.6), 0.27, fill=False, edgecolor=INK, lw=1.7))
    ax.text(8.0, 1.6, "+", ha="center", va="center", fontsize=19)
    ax.plot([0.7, 1.0, 1.0, 2.7], [1.6, 1.6, 2.5, 2.5], color=MUTED, lw=2)
    arrow(ax, (2.7, 2.5), (3.0, 2.5), MUTED)
    ax.plot([5.0, 8.0, 8.0], [2.5, 2.5, 1.9], color=MUTED, lw=2)
    arrow(ax, (8.0, 1.9), (8.0, 1.87), MUTED)
    ax.plot([1.0, 1.0, 1.65], [1.6, 0.75, 0.75], color=BLUE, lw=2)
    arrow(ax, (1.65, 0.75), (1.9, 0.75), BLUE)
    arrow(ax, (3.55, 0.75), (5.0, 0.75), BLUE)
    ax.text(4.27, 0.34, "$r$ coordinates", ha="center", va="center", fontsize=11)
    ax.plot([6.65, 8.0, 8.0], [0.75, 0.75, 1.25], color=BLUE, lw=2)
    arrow(ax, (8.0, 1.25), (8.0, 1.33), BLUE)
    arrow(ax, (8.27, 1.6), (9.0, 1.6))
    ax.text(9.45, 1.6, "$y$\noutput", ha="center", va="center")
    ax.text(6.4, 2.78, "$W_0x$", ha="center", color=MUTED)
    ax.text(7.13, 0.35, "$BAx$", ha="center", color=BLUE)
    fig.text(0.5, 0.02, "Shown with adapter multiplier s = 1 and bias omitted.",
        ha="center", color=MUTED, fontsize=11)
    fig.subplots_adjust(bottom=0.12, top=0.85)
    save(fig, "01-lora-path")


def svd_stretch():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7), layout="constrained")
    theta = np.linspace(0, 2 * np.pi, 500)
    x = np.vstack([np.cos(theta), np.sin(theta)])
    y = np.diag([4.0, 1.0]) @ x
    for ax in axes:
        geometry(ax, (-4.7, 4.7), (-2.3, 2.3))
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-2, -1, 0, 1, 2])
    axes[0].plot(*x, color=MUTED, lw=2)
    axes[1].plot(*y, color=MUTED, lw=2)
    arrow(axes[0], (0, 0), (1, 0), BLUE)
    arrow(axes[0], (0, 0), (0, 1), ORANGE)
    arrow(axes[1], (0, 0), (4, 0), BLUE)
    arrow(axes[1], (0, 0), (0, 1), ORANGE)
    axes[0].text(1.12, -0.45, "$v_1$: length 1", fontsize=11)
    axes[0].text(0.2, 1.25, "$v_2$: length 1", fontsize=11)
    axes[1].text(1.8, -0.45, "$4u_1$: length 4", fontsize=11)
    axes[1].text(0.2, 1.25, "$1u_2$: length 1", fontsize=11)
    axes[0].set_title("Inputs with length 1")
    axes[1].set_title("Outputs after $W_0=\mathrm{diag}(4,1)$")
    fig.suptitle("Singular values measure stretch along special directions", fontsize=14)
    save(fig, "02-svd-stretch")


def scaled_bases():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.7), layout="constrained")
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    factors = [np.eye(2), np.diag([2, 1]), rotation @ np.diag([2, 1])]
    titles = ["Unit directions", "Scaled spectral columns", "Scaled random columns"]
    for ax, matrix, title in zip(axes, factors, titles):
        geometry(ax, (-1.45, 2.6), (-0.55, 2.6))
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_yticks([0, 1, 2])
        for i, color in enumerate((BLUE, ORANGE)):
            vec = matrix[:, i]
            arrow(ax, (0, 0), vec, color)
            # Offset labels at endpoints, away from axes and the other arrow.
            shift = np.array([0.10, 0.13])
            if vec[0] < 0:
                shift = np.array([-0.14, 0.13])
            ax.text(*(vec + shift), f"length {np.linalg.norm(vec):.0f}",
                ha="right" if vec[0] < 0 else "left", fontsize=11)
        e1 = matrix[:, 0] / np.linalg.norm(matrix[:, 0])
        e2 = matrix[:, 1] / np.linalg.norm(matrix[:, 1])
        corner = np.vstack([0.15 * e1, 0.15 * (e1 + e2), 0.15 * e2])
        ax.plot(corner[:, 0], corner[:, 1], color=MUTED, lw=1)
        ax.set_title(title, fontsize=12)
    fig.suptitle("Scaling preserves perpendicularity; rotating changes orientation", fontsize=14)
    save(fig, "03-scaled-bases")


def dora_length():
    fig, ax = plt.subplots(figsize=(6.8, 5.2), layout="constrained")
    geometry(ax, (-0.5, 5.1), (-0.5, 6.1))
    u = np.array([0.6, 0.8])
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=GRID, lw=1.4, linestyle="--"))
    arrow(ax, (0, 0), 6 * u, ORANGE, 2.8)
    arrow(ax, (0, 0), 5 * u, BLUE, 2.8)
    arrow(ax, (0, 0), u, GREEN, 2.8)
    ax.scatter(*u, color=GREEN, s=35, zorder=5)
    ax.scatter(*(5 * u), color=BLUE, s=35, zorder=5)
    ax.scatter(*(6 * u), color=ORANGE, s=35, zorder=5)
    ax.annotate("$u=(0.6,0.8)$\nunit direction", xy=u, xytext=(1.7, 0.7),
        arrowprops={"arrowstyle": "-", "color": MUTED}, fontsize=11)
    ax.annotate("$z=(3,4)$\nlength 5", xy=5 * u, xytext=(0.1, 3.8),
        arrowprops={"arrowstyle": "-", "color": MUTED}, fontsize=11)
    ax.annotate("$w=6u=(3.6,4.8)$\nlength set by $m=6$", xy=6 * u,
        xytext=(2.0, 5.55), arrowprops={"arrowstyle": "-", "color": MUTED}, fontsize=11)
    ax.set_title("DoRA uses a unit direction and a separate length", pad=14)
    save(fig, "04-dora-length")


def gradient_geometry():
    fig, ax = plt.subplots(figsize=(7.1, 5.5), layout="constrained")
    geometry(ax, (-1.2, 2.0), (-1.2, 1.75))
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=GRID, lw=1.7))
    u = np.array([0.6, 0.8])
    radial = np.array([0.36, 0.48])
    tangent = np.array([0.64, -0.48])
    g = np.array([1.0, 0.0])
    ax.plot([0, u[0]], [0, u[1]], color=MUTED, lw=1.2, linestyle="--")
    ax.plot([*(u + radial)[0:1], (u + g)[0]],
            [*(u + radial)[1:2], (u + g)[1]], color=GRID, lw=1.4, linestyle="--")
    arrow(ax, u, u + g, BLUE)
    arrow(ax, u, u + radial, GREEN)
    arrow(ax, u, u + tangent, ORANGE)
    ax.scatter(*u, color=INK, s=32, zorder=5)
    ax.text(u[0] - 0.16, u[1] - 0.2, "$u$", ha="right")
    ax.annotate("Detached: $(1,0)$", xy=u + g, xytext=(1.15, 1.04),
        fontsize=11, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.annotate("Removed radial part\n$(0.36,0.48)$", xy=u + radial,
        xytext=(-0.1, 1.46), fontsize=11,
        arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.annotate("Exact: $(0.64,-0.48)$\nperpendicular to $u$", xy=u + tangent,
        xytext=(0.5, -0.35), fontsize=11,
        arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.text(-1.02, -0.82, "Circle = unit directions\nArrows = gradients, not optimizer steps", fontsize=11)
    ax.set_title("Same forward direction, different backward gradients", pad=14)
    save(fig, "05-gradient-geometry")


def interaction():
    fig, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
    ax.plot([0, 1], [59, 62], marker="o", lw=2.4, color=BLUE)
    ax.plot([0, 1], [59, 60], marker="s", lw=2.4, color=ORANGE)
    ax.set_xlim(-0.16, 1.85)
    ax.set_ylim(58.4, 62.7)
    ax.set_xticks([0, 1], ["Random", "Spectral"])
    ax.set_yticks([59, 60, 61, 62])
    ax.set_ylabel("Hypothetical accuracy (%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.text(1.05, 62.0, "Exact: +3 points", va="center")
    ax.text(1.05, 60.0, "Detached: +1 point", va="center")
    ax.text(0.18, 62.15, "Interaction = 3 − 1 = 2 points", fontsize=12)
    ax.set_title("Interaction compares the two spectral gains", pad=14)
    save(fig, "06-interaction")


if __name__ == "__main__":
    lora_path()
    svd_stretch()
    scaled_bases()
    dora_length()
    gradient_geometry()
    interaction()
