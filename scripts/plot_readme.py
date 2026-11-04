# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Render README figures from assets/performance.json: uv run scripts/plot_readme.py."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
GREEN = "#1E7A68"
GRAY = "#71818A"
INK = "#193B35"
MUTED = "#526660"
GRID = "#E2EAE7"


def canvas(title, subtitle, height):
    fig, ax = plt.subplots(figsize=(10.8, height), facecolor="white")
    fig.subplots_adjust(left=0.235, right=0.86, top=0.78, bottom=0.23)
    fig.text(0.05, 0.92, title, fontsize=23, weight="bold", color=INK)
    fig.text(0.05, 0.855, subtitle, fontsize=12, color=MUTED)
    ax.set_facecolor("white")
    for edge in ("top", "right", "left"):
        ax.spines[edge].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, name, title, description):
    fig.savefig(
        ASSETS / name,
        format="svg",
        metadata={"Date": None, "Title": title, "Description": description},
    )
    plt.close(fig)


def compression(data):
    rows = sorted(
        data["compression"]["rows"], key=lambda row: row["ratio"], reverse=True
    )
    fig, ax = canvas(
        "Smaller matrices, independent cells",
        "Compression ratio · higher is better",
        5.7,
    )
    for i, row in enumerate(rows):
        focus = row["method"] == "CellPack"
        ax.barh(i, row["ratio"], height=0.48, color=GREEN if focus else GRAY)
        ax.text(
            row["ratio"] + 0.1,
            i,
            f"{row['ratio']:.2f}×",
            va="center",
            color=GREEN if focus else INK,
            fontsize=12,
            weight="bold" if focus else "normal",
        )
    ax.set_yticks(range(len(rows)), [row["method"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 6.5)
    ax.set_xticks(range(7))
    ax.set_xlabel("Uncompressed bytes / compressed bytes", color=MUTED, labelpad=12)
    fig.text(
        0.05,
        0.075,
        "734 matrix blocks for CellPack; 724 for general codecs. Mean ratios, selected codecs.",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.035,
        "Frozen experimental results · not a current-version benchmark",
        fontsize=10.5,
        color=MUTED,
    )
    finish(
        fig,
        "performance-compression.svg",
        "CellPack compression ratios",
        data["compression"]["protocol"],
    )


def loading(data):
    rows = sorted(
        data["loading"]["rows"], key=lambda row: row["cells_per_s"], reverse=True
    )
    fig, ax = canvas(
        "Random access, ready for training",
        "Global random order · 1 NVMe SSD · 16 workers · higher is better",
        6.8,
    )
    for i, row in enumerate(rows):
        focus = row["method"] == "CellPack"
        ax.scatter(
            row["cells_per_s"],
            i,
            s=90 if focus else 55,
            color=GREEN if focus else GRAY,
            zorder=3,
            marker="D" if focus else "o",
        )
        ax.text(
            1.04,
            i,
            f"{row['cells_per_s']:,}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=11.5,
            color=GREEN if focus else INK,
            weight="bold" if focus else "normal",
        )
    ax.set_yticks(range(len(rows)), [row["method"] for row in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xscale("log")
    ax.set_xlim(800, 1_700_000)
    ax.xaxis.set_major_locator(FixedLocator([1000, 10_000, 100_000, 1_000_000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1K", "10K", "100K", "1M"])
    ax.set_xlabel("Cells per second (log scale)", color=MUTED, labelpad=12)
    fig.text(
        0.05,
        0.105,
        "CellPack: no cache, report-selected 5 s point, fixed 2,048-gene output.",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.07,
        "Baselines: 20 GB cache, three-seed means, 60 s windows, sample output.",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.05,
        0.035,
        "Frozen experimental results · not a current-version benchmark",
        fontsize=10.5,
        color=MUTED,
    )
    finish(
        fig,
        "performance-loading.svg",
        "CellPack random-order loading throughput",
        data["loading"]["protocol"],
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "svg.fonttype": "none",
            "svg.hashsalt": "cellpack-readme",
        }
    )
    data = json.loads((ASSETS / "performance.json").read_text())
    compression(data)
    loading(data)


if __name__ == "__main__":
    main()
