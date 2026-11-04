# /// script
# requires-python = ">=3.12"
# dependencies = ["fonttools[woff]==4.55.3"]
# ///
"""Redraw README assets from frozen inputs: uv run scripts/plot_readme.py.

The figures share the website's visual language. Fonts come from the docs
site's npm dependencies (run `npm ci` in docs/cellpack first) and are subset
and embedded, so GitHub renders the same typefaces as the website.
"""

import base64
import csv
import hashlib
import io
import json
import statistics
from functools import cache
from html import escape
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MODULES = ROOT / "docs/cellpack/node_modules"
WIDTH = 1120
LEFT, RIGHT = 48, 1072
FONTS = {
    "head": ("CP Head", "'Space Grotesk', sans-serif", {None: "@fontsource-variable/space-grotesk/files/space-grotesk-latin-wght-normal.woff2"}),
    "body": ("CP Body", "'DM Sans', -apple-system, 'Segoe UI', sans-serif", {None: "@fontsource-variable/dm-sans/files/dm-sans-latin-wght-normal.woff2"}),
    "mono": (
        "CP Mono",
        "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
        {w: f"@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-{w}-normal.woff2" for w in (400, 500, 600, 700)},
    ),
}
# Website tokens (docs/cellpack/.vitepress/theme/style.css).
THEMES = {
    "light": {
        "bg": "#faf9f6",
        "surface": "#f0f0e9",
        "text": "#252923",
        "text2": "#60645d",
        "text3": "#72776e",
        "divider": "#dedfd7",
        "border": "#c9cdc1",
        "teal": "#25745f",
        "tealSoft": "#dfede4",
        "accent": "#bc512e",
        "accentSoft": "#f1e0d7",
    },
    "dark": {
        "bg": "#171a18",
        "surface": "#1e251f",
        "text": "#edf0e7",
        "text2": "#b3bcae",
        "text3": "#a0ab9b",
        "divider": "#343c34",
        "border": "#4f5b4e",
        "teal": "#80cbb2",
        "tealSoft": "#203a2e",
        "accent": "#f19a76",
        "accentSoft": "#3b2b24",
    },
}
# AnnData's own colours, as on the website's format scene.
HUE = {
    "obsp": "#d9695f",
    "obsm": "#e3913f",
    "X": "#5fae74",
    "obs": "#d9b44a",
    "var": "#4f97c2",
    "varm": "#3f6f86",
    "varp": "#8d6ab3",
}


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def mix(a, b, t):
    """Solid colour t of the way from a to b."""
    pa = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


@cache
def font_file(family, weight):
    files = FONTS[family][2]
    path = MODULES / files.get(weight, files.get(None) or files[400])
    if not path.exists():
        raise SystemExit(f"Missing {path}; run `npm ci` in docs/cellpack first.")
    return path


@cache
def metrics(family, weight):
    font = TTFont(font_file(family, weight))
    return font.getBestCmap(), font["hmtx"].metrics, font["head"].unitsPerEm


def text_width(value, size, family="body", weight=400):
    static = None if None in FONTS[family][2] else min(FONTS[family][2], key=lambda w: abs(w - weight))
    cmap, hmtx, upm = metrics(family, static)
    advance = sum(hmtx[cmap.get(ord(ch), cmap[ord("x")])][0] for ch in value)
    # Variable fonts are measured at their default instance; bolder text runs wider.
    return advance * size / upm * (1 + max(0, weight - 400) / 2500 if static is None else 1)


def wrap(value, size, width, family="mono"):
    lines, current = [], ""
    for word in value.split(" "):
        trial = f"{current} {word}".strip()
        if current and text_width(trial, size, family) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    return [*lines, current]


class Figure:
    def __init__(self, c, height, kicker, title, subtitle, desc, tone="teal"):
        self.c, self.height, self.title, self.desc = c, height, title, desc
        self.used = {}
        self.body = [
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="16" fill="{c["bg"]}" stroke="{c["divider"]}"/>'
        ]
        self.text(LEFT, 52, kicker, 12, c[tone], 500, family="mono", spacing=1.8)
        self.text(LEFT, 90, title, 30, c["text"], 500, family="head")
        self.text(LEFT, 119, subtitle, 15.5, c["text2"])

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, width=1, opacity=None, dash=None):
        extra = f' rx="{num(rx)}"' if rx else ""
        if stroke:
            extra += f' stroke="{stroke}" stroke-width="{num(width)}"'
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            extra += f' fill-opacity="{num(opacity)}"'
        self.body.append(
            f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" fill="{fill}"{extra}/>'
        )

    def text(self, x, y, value, size=14.0, color=None, weight=400, anchor="start", family="body", spacing=None, opacity=None, halo=False):
        self.used.setdefault((family, weight), set()).update(value)
        extra = f' font-family="{self.family(family)}"' if family != "body" else ""
        if weight != 400:
            extra += f' font-weight="{weight}"'
        if anchor != "start":
            extra += f' text-anchor="{anchor}"'
        if spacing:
            extra += f' letter-spacing="{num(spacing)}"'
        if opacity is not None:
            extra += f' fill-opacity="{num(opacity)}"'
        if halo:
            extra += f' stroke="{self.c["bg"]}" stroke-width="5" stroke-linejoin="round" paint-order="stroke"'
        self.body.append(
            f'<text x="{num(x)}" y="{num(y)}" font-size="{num(size)}" fill="{color or self.c["text"]}"{extra}>{escape(value)}</text>'
        )

    def line(self, x1, y1, x2, y2, color, width=1, dash=None, opacity=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        if opacity is not None:
            extra += f' stroke-opacity="{num(opacity)}"'
        self.body.append(
            f'<line x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}" stroke="{color}" stroke-width="{num(width)}"{extra}/>'
        )

    def path(self, d, stroke="none", fill="none", width=1, opacity=None, dash=None):
        extra = f' fill-opacity="{num(opacity)}"' if opacity is not None else ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.body.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{num(width)}" stroke-linejoin="round" stroke-linecap="round"{extra}/>'
        )

    @staticmethod
    def family(key):
        name, fallback, _ = FONTS[key]
        return f"'{name}', {fallback}"

    def notes(self, lines):
        wrapped = [w for value in lines for w in wrap(value, 11.5, RIGHT - LEFT)]
        for i, value in enumerate(wrapped):
            y = self.height - 30 - 18 * (len(wrapped) - 1 - i)
            self.text(LEFT, y, value, 11.5, self.c["text3"], family="mono")

    def legend(self, x, y, items):
        for color, label in items:
            self.rect(x, y - 10, 11, 11, color, rx=2.5)
            self.text(x + 18, y, label, 12, self.c["text2"], family="mono")
            x += 18 + text_width(label, 12, "mono") + 26

    def xgrid(self, scale, ticks, top, bottom, label=None):
        for value, tick in ticks:
            x = scale(value)
            self.line(x, top, x, bottom, self.c["divider"])
            self.text(x, bottom + 22, tick, 11.5, self.c["text3"], anchor="middle", family="mono")
        self.line(scale(0), top, scale(0), bottom, self.c["border"])
        if label:
            self.text(scale(ticks[-1][0]), bottom + 42, label, 11, self.c["text3"], anchor="end", family="mono")

    def save(self, name, mode):
        merged = {}
        for (family, weight), chars in self.used.items():
            key = (family, None if None in FONTS[family][2] else weight)
            merged.setdefault(key, set()).update(chars)
        self.used = {(f, w if w is not None else 400): chars for (f, w), chars in merged.items()}
        faces = []
        for (family, weight), chars in sorted(self.used.items()):
            static = None if None in FONTS[family][2] else weight
            options = subset.Options()
            options.flavor = "woff2"
            options.layout_features = ["kern", "tnum"]
            options.name_IDs = []
            font = TTFont(font_file(family, static))
            sub = subset.Subsetter(options)
            sub.populate(text="".join(sorted(chars | {" "})))
            sub.subset(font)
            out = io.BytesIO()
            font.flavor = "woff2"
            font.save(out)
            data = base64.b64encode(out.getvalue()).decode()
            span = f"{weight}" if static else "100 900"
            faces.append(
                f"@font-face{{font-family:'{FONTS[family][0]}';font-weight:{span};src:url(data:font/woff2;base64,{data}) format('woff2')}}"
            )
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{self.height}" viewBox="0 0 {WIDTH} {self.height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(self.title)}</title>',
            f'<desc id="desc">{escape(self.desc)}</desc>',
            f"<style>{''.join(faces)}text{{font-variant-numeric:tabular-nums}}</style>",
            f'<g font-family="{self.family("body")}">',
            *self.body,
            "</g>",
            "</svg>",
        ]
        suffix = "" if mode == "light" else "-dark"
        (ASSETS / f"{name}{suffix}.svg").write_text("\n".join(svg) + "\n")


def verify(data):
    """Refuse to redraw labels from a changed or mismatched source snapshot."""
    for entry in [*data["sources"].values(), data["report"]]:
        path = ROOT / entry["repository_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    sizes = json.loads((ASSETS / "data/corpus-storage-bytes.json").read_text())
    matrices = [m for dataset in sizes for m in dataset["matrices"]]
    nonmatrix = sum(d["h5cp"] - d["streams"] for d in sizes)
    scale = data["storage_scale"]
    assert scale["cells"] == sum(d["cells"] for d in sizes)
    assert scale["nonmatrix_bytes"] == nonmatrix
    assert scale["h5cp_bytes"] == sum(d["h5cp"] for d in sizes)
    assert scale["h5ad_scipy_bytes"] == nonmatrix + sum(
        m["data"] + m["indices"] + m["indptr"] for m in matrices
    )
    assert scale["h5ad_compact_indices_bytes"] == nonmatrix + sum(
        m["data"] + m["indices_min"] + m["indptr"] for m in matrices
    )
    assert scale["counts_stream_bytes"] == sum(
        m["cpk"] for m in matrices if m["path"] == "X"
    )
    slots = {"X": "X", "layers/normed": "layers/normed"}
    for row in data["expression_storage"]["rows"]:
        group = [m for m in matrices if m["path"] == slots[row["slot"]]]
        assert len(group) == row["datasets"]
        assert row["h5ad_scipy_bytes"] == sum(
            m["data"] + m["indices"] + m["indptr"] for m in group
        )
        assert row["h5ad_compact_indices_bytes"] == sum(
            m["data"] + m["indices_min"] + m["indptr"] for m in group
        )
        assert row["cellpack_matrix_bytes"] == sum(m["cpk"] for m in group)
        cpk = row["cellpack_matrix_bytes"]
        assert row["h5ad_scipy_to_cellpack_ratio"] == row["h5ad_scipy_bytes"] / cpk
        assert (
            row["h5ad_compact_indices_to_cellpack_ratio"]
            == row["h5ad_compact_indices_bytes"] / cpk
        )
    granularity = [
        r
        for r in csv.DictReader(
            line
            for line in (
                ROOT / data["sources"]["compression_granularity.csv"]["repository_path"]
            )
            .read_text()
            .splitlines()
            if not line.startswith("#")
        )
        if r["matrix"] == "pooled"
    ]
    for row in data["codecs"]["rows"]:
        for key, cells in (("per_cell", 1), ("block_2048", 2048)):
            if key in row:
                assert row[key] == float(
                    next(
                        r["mean_ratio"]
                        for r in granularity
                        if r["codec"] == row["codec"]
                        and int(r["cells_per_unit"]) == cells
                    )
                )
    methods = list(
        csv.DictReader(
            line
            for line in (
                ROOT / data["sources"]["loader_methods.csv"]["repository_path"]
            )
            .read_text()
            .splitlines()
            if not line.startswith("#")
        )
    )
    assert len(data["loading"]["rows"]) == len(methods) + 1
    for row in data["loading"]["rows"]:
        if row["method"] != "CellPack":
            assert row["cells_per_s"] == float(
                next(m for m in methods if m["method"].rstrip("*") == row["method"])["nvme"]
            )
    paper = json.loads(
        (
            ROOT / data["sources"]["cellpack_paper_values.json"]["repository_path"]
        ).read_text()
    )
    assert (
        next(r for r in data["loading"]["rows"] if r["method"] == "CellPack")[
            "cells_per_s"
        ]
        == paper["fig5"]["nvme_16"]
    )





def loading(data, mode, c):
    rows = sorted(data["loading"]["rows"], key=lambda r: r["cells_per_s"], reverse=True)
    top, pitch, bar = 172, 32, 16
    bottom = top + pitch * len(rows)
    fig = Figure(
        c,
        bottom + 66,
        "LOADING · RANDOM ORDER",
        "1.165M cells/s from one SSD",
        "Shuffled cells per second, read from one NVMe SSD",
        data["loading"]["protocol"],
        "accent",
    )
    x0, x1, vmax = 200, 960, 1_250_000

    def scale(v):
        return x0 + v / vmax * (x1 - x0)

    fig.text(x0, top - 18, "cells / second", 11, c["text3"], family="mono")
    fig.xgrid(
        scale,
        [(0, "0"), (250_000, "250K"), (500_000, "500K"), (750_000, "750K"), (1e6, "1M")],
        top - 6,
        bottom,
    )
    for i, row in enumerate(rows):
        cy = top + pitch * i + pitch / 2
        focus = row["method"] == "CellPack"
        w = max(scale(row["cells_per_s"]) - x0, 2)
        fig.text(x0 - 14, cy + 5, row["method"], 14, c["accent"] if focus else c["text2"], 700 if focus else 500, "end")
        fig.rect(x0, cy - bar / 2, w, bar, c["accent"] if focus else mix(c["bg"], c["text3"], 0.5), rx=min(3, w / 2))
        fig.text(x0 + w + 10, cy + 4.5, f"{row['cells_per_s']:,}", 13, c["accent"] if focus else c["text2"], 600 if focus else 400, family="mono", halo=True)
    fig.save("performance-loading", mode)


def storage(data, mode, c):
    s = data["storage_scale"]
    x = next(r for r in data["expression_storage"]["rows"] if r["slot"] == "X")
    groups = [
        ("Whole files", "all matrices and annotations", s["h5ad_scipy_bytes"], s["h5cp_bytes"]),
        ("Expression matrix X", "", x["h5ad_scipy_bytes"], x["cellpack_matrix_bytes"]),
    ]
    top, bar, step, gap = 184, 26, 34, 44
    span = 2 * step
    bottom = top + span * len(groups) + gap * (len(groups) - 1)
    fig = Figure(
        c,
        bottom + 66,
        "STORAGE",
        f"124M cells in {s['h5cp_bytes'] / 1e9:.0f} GB",
        "The same 124M-cell corpus as AnnData H5AD and as CellPack H5CP",
        s["protocol"],
    )
    h5ad, h5cp = mix(c["bg"], c["text3"], 0.5), c["teal"]
    fig.legend(LEFT, 152, [(h5ad, "AnnData H5AD"), (h5cp, "CellPack H5CP")])
    x0, x1, vmax = 240, 1000, 5.2e12

    def scale(v):
        return x0 + v / vmax * (x1 - x0)

    fig.xgrid(scale, [(t * 1e12, f"{t} TB" if t else "0") for t in range(6)], top - 8, bottom)
    for g, (name, detail, before, after) in enumerate(groups):
        gy = top + g * (span + gap)
        fig.text(x0 - 16, gy + span / 2 + (-3 if detail else 5), name, 15, c["text"], 600, "end")
        if detail:
            fig.text(x0 - 16, gy + span / 2 + 16, detail, 12.5, c["text3"], anchor="end")
        for i, (size, color) in enumerate([(before, h5ad), (after, h5cp)]):
            y = gy + i * step + (step - bar) / 2
            fig.rect(x0, y, scale(size) - x0, bar, color, rx=3)
            label = f"{size / 1e12:.2f} TB" if size > 1e12 else f"{size / 1e9:.1f} GB" if size < 3e11 else f"{size / 1e9:.0f} GB"
            end = scale(size) + 10
            fig.text(end, y + bar - 8, label, 13, c["teal"] if i else c["text2"], 600 if i else 400, family="mono", halo=True)
            if i:
                end += text_width(label, 13, "mono", 600) + 16
                fig.text(end, y + bar - 7, f"{before / after:.2f}× smaller", 17, c["teal"], 600, family="head", halo=True)
    fig.save("performance-storage", mode)


def compression(data, mode, c):
    rows = data["codecs"]["rows"]
    top, pitch, bar = 186, 40, 13
    bottom = top + pitch * len(rows)
    fig = Figure(
        c,
        bottom + 66,
        "COMPRESSION",
        "Best compression, one cell at a time",
        "Compression ratio on single-cell expression matrices · higher is better",
        data["codecs"]["protocol"],
    )
    block, cell = mix(c["bg"], c["text3"], 0.32), mix(c["bg"], c["text3"], 0.75)
    fig.legend(
        LEFT,
        152,
        [(c["teal"], "CellPack · one cell at a time"), (cell, "codec · one cell at a time"), (block, "codec · 2,048 cells per block")],
    )
    x0, x1, vmax = 200, 1000, 6.5

    def scale(v):
        return x0 + v / vmax * (x1 - x0)

    fig.xgrid(scale, [(t, f"{t}×" if t else "0") for t in range(7)], top - 8, bottom)
    for i, row in enumerate(rows):
        cy = top + pitch * i + pitch / 2
        focus = row["codec"] == "cellpack"
        fig.text(x0 - 14, cy + 5, row["method"], 14, c["teal"] if focus else c["text2"], 700 if focus else 500, "end")
        if focus:
            fig.rect(x0, cy - bar, scale(row["per_cell"]) - x0, 2 * bar, c["teal"], rx=3)
            fig.text(scale(row["per_cell"]) + 10, cy + 6, f"{row['per_cell']:.2f}×", 16, c["teal"], 600, family="head", halo=True)
            continue
        for k, (key, color, y) in enumerate([("per_cell", cell, cy - bar), ("block_2048", block, cy)]):
            fig.rect(x0, y + 1, scale(row[key]) - x0, bar - 2, color, rx=2)
            fig.text(scale(row[key]) + 8, y + bar - 2.5, f"{row[key]:.2f}×", 11.5, c["text2"] if k == 0 else c["text3"], family="mono", halo=True)
    fig.save("performance-compression", mode)


def billion(data, mode, c):
    source = (
        ROOT / data["sources"]["billion_cell_online.sources.json"]["repository_path"]
    )
    receipt = json.loads(source.read_text())
    epochs = receipt["epochs"]
    end = epochs[-1]["end_seconds"]
    mean = epochs[-1]["cells_total"] / end
    steady = sum(e["cells"] for e in epochs[1:]) / sum(e["seconds"] for e in epochs[1:])
    total = end + receipt["metadata_open_seconds"]
    rss = receipt["process_peak_rss_bytes"]
    assert f"{mean / 1e6:.2f}" == "3.24"
    assert f"{steady / 1e6:.2f}" == "3.67"
    assert f"{total / 60:.1f}" == "29.8"
    assert f"{rss / 1e12:.2f}" == "1.29"
    fig = Figure(
        c,
        600,
        "SCALE",
        f"5B cells in {end / 60:.1f} minutes",
        "Five epochs over a 1B-cell corpus, freshly shuffled every epoch",
        data["billion_cell"]["protocol"],
        "accent",
    )
    gap = 14
    card = (RIGHT - LEFT - 2 * gap) / 3
    for k, (label, value, tone) in enumerate(
        [
            ("AVERAGE", f"{mean / 1e6:.2f}M cells/s", c["accent"]),
            ("STEADY STATE", f"{steady / 1e6:.2f}M cells/s", c["teal"]),
            ("TOTAL TIME · 5B CELLS", f"{end / 60:.1f} min", c["text2"]),
        ]
    ):
        x = LEFT + k * (card + gap)
        fig.rect(x, 144, card, 82, c["surface"], rx=10)
        fig.text(x + 18, 170, label, 11, c["text3"], family="mono", spacing=0.6)
        fig.text(x + 18, 208, value, 27, tone, 600, family="head")

    px0, px1, top, bottom, ymax = 92, RIGHT, 270, 520, 4.4

    def sx(seconds):
        return px0 + seconds / end * (px1 - px0)

    def sy(rate):
        return bottom - rate / ymax * (bottom - top)

    for tick in range(1, 5):
        fig.line(px0, sy(tick), px1, sy(tick), c["divider"])
    for tick in range(5):
        fig.text(px0 - 12, sy(tick) + 4, f"{tick}M" if tick else "0", 11.5, c["text3"], anchor="end", family="mono")
    fig.text(px0, top - 30, "cells / second", 11, c["text3"], family="mono")
    for minute in range(0, int(end // 60) + 1, 5):
        fig.text(sx(minute * 60), bottom + 20, str(minute), 11.5, c["text3"], anchor="middle", family="mono")
    fig.text(px1, bottom + 40, "minutes", 11, c["text3"], anchor="end", family="mono")

    series = list(
        csv.DictReader(
            (
                ROOT / data["sources"]["billion_cell_online.csv"]["repository_path"]
            ).open()
        )
    )
    # Weight uneven samples by their actual duration, without crossing epoch boundaries.
    points = [(0.0, 0.0)]
    for epoch in epochs:
        start = epoch["end_seconds"] - epoch["seconds"]
        previous_time, previous_cells = start, epoch["cells_total"] - epoch["cells"]
        for row in series:
            t, cells = float(row["seconds"]), int(row["cells"])
            if start < t <= epoch["end_seconds"] and t - previous_time >= 5:
                rate = (cells - previous_cells) / (t - previous_time) / 1e6
                points.append(((t + previous_time) / 2, rate))
                previous_time, previous_cells = t, cells
    line = " ".join(f"{num(sx(t))},{num(sy(r))}" for t, r in points)
    fig.path(f"M {line} L {num(sx(points[-1][0]))},{bottom} L {px0},{bottom} Z", fill=c["accent"], opacity=0.1)
    fig.path(f"M {line}", stroke=c["accent"], width=1.7)
    fig.line(px0, bottom, px1, bottom, c["border"])

    for epoch in epochs:
        start = epoch["end_seconds"] - epoch["seconds"]
        rate = epoch["cells_per_second"] / 1e6
        mid = (sx(start) + sx(epoch["end_seconds"])) / 2
        if epoch["epoch"] < len(epochs):
            fig.line(sx(epoch["end_seconds"]), top - 22, sx(epoch["end_seconds"]), bottom, c["text3"], 1, "2 3", 0.6)
        fig.line(sx(start), sy(rate), sx(epoch["end_seconds"]), sy(rate), c["text"], 1.1, "4 3", 0.5)
        fig.text(mid, top - 12, f"E{epoch['epoch']}", 11.5, c["text3"], 600, "middle", family="mono")
        fig.text(mid, top + 4, f"{rate:.2f}M/s", 11, c["text2"], anchor="middle", family="mono")
    fig.text(sx(receipt["first_batch_seconds"] / 2), bottom - 10, "startup", 11, c["text3"], anchor="middle", family="mono")
    fig.save("performance-billion", mode)


def anndata(data, mode, c):
    """A static companion to the website's format scene: swap only X."""
    ratio = next(r for r in data["expression_storage"]["rows"] if r["slot"] == "X")["h5ad_scipy_to_cellpack_ratio"]
    fig = Figure(
        c,
        540,
        "FORMAT · H5CP",
        "Swap only X.",
        "An H5CP file is an H5AD file whose X is a CellPack matrix. Everything else keeps its AnnData encoding.",
        "Two AnnData layouts. In H5AD, X is stored as CSR arrays (indptr, indices, data). In H5CP, the same X is a CellPack matrix that takes about a tenth of the space; obs, var, obsm, obsp, varm, varp, layers and uns are unchanged.",
    )
    U, g = 92, 6
    width = 3.6 * U + 3 * g

    def block(x, y, w, h, key):
        fig.rect(x, y, w, h, HUE[key], rx=4, opacity=0.85)
        fig.text(x + w / 2, y + h / 2 + 5, key, 13.5, "#ffffff", 700, "middle", family="mono")

    for ax, packed in [(LEFT + 8, False), (RIGHT - 8 - width, True)]:
        ay = 186
        fig.text(ax, ay - 22, "H5CP" if packed else "H5AD", 12, c["teal"] if packed else c["text2"], 700, family="mono", spacing=1.2)
        X = {"x": ax + 1.0 * U + 0.55 * U + 2 * g, "y": ay + 0.4 * U + g, "w": 1.6 * U, "h": 1.3 * U}
        block(X["x"], ay, X["w"], 0.4 * U, "var")
        block(ax, X["y"], 1.0 * U, X["h"], "obsp")
        block(ax + 1.0 * U + g, X["y"], 0.55 * U, X["h"], "obsm")
        block(X["x"] + X["w"] + g, X["y"], 0.45 * U, X["h"], "obs")
        below = X["y"] + X["h"] + g + 26
        block(X["x"], below, X["w"], 0.4 * U, "varm")
        block(X["x"], below + 0.4 * U + g, X["w"], 0.8 * U, "varp")
        fig.text(X["x"] + 4, X["y"] + X["h"] + 19, "layers", 11.5, c["text2"], 500, family="mono")
        fig.text(ax + 0.2 * U, X["y"] + X["h"] + 0.95 * U, "{ uns … }", 17, c["text2"], 700, family="mono")
        for k in (2, 1):
            fig.rect(X["x"] - 3 * k, X["y"] + 3 * k, X["w"], X["h"], HUE["X"], rx=4, opacity=0.35)
        if packed:
            fig.rect(X["x"], X["y"], X["w"], X["h"], c["bg"], rx=4, stroke=c["text3"], dash="3 3")
            side = ratio**-0.5
            w, h = X["w"] * side * 1.55, X["h"] * side / 1.55
            fig.rect(X["x"] + 6, X["y"] + 6, w, h, c["teal"], rx=4, opacity=0.95)
            fig.text(X["x"] + 6 + w / 2, X["y"] + 6 + h / 2 + 4, "CellPack", 11, c["bg"], 700, "middle", family="mono")
            fig.text(X["x"] + X["w"] / 2, X["y"] + X["h"] - 34, f"{ratio:.2f}×", 22, c["teal"], 600, "middle", family="head")
            fig.text(X["x"] + X["w"] / 2, X["y"] + X["h"] - 16, "smaller", 11, c["text2"], anchor="middle", family="mono")
        else:
            fig.rect(X["x"], X["y"], X["w"], X["h"], HUE["X"], rx=4, opacity=0.92)
            fig.text(X["x"] + 10, X["y"] + 20, "X · CSR", 13, "#ffffff", 700, family="mono")
            bh = (X["h"] - 40) / 3 - 5
            for i, label in enumerate(["indptr", "indices", "data"]):
                yy = X["y"] + 32 + i * (bh + 5)
                fig.rect(X["x"] + 8, yy, X["w"] - 16, bh, "#ffffff", rx=3, opacity=0.3)
                fig.text(X["x"] + X["w"] / 2, yy + bh / 2 + 4, label, 11, "#ffffff", 600, "middle", family="mono")

    ay = 186 + 0.4 * U + g + 0.65 * U
    a0, a1 = LEFT + 8 + width + 28, RIGHT - 8 - width - 28
    fig.path(f"M {num(a0)} {num(ay)} H {num(a1)}", stroke=c["teal"], width=1.5, dash="6 5")
    fig.path(f"M {num(a1 - 8)} {num(ay - 7)} L {num(a1)} {num(ay)} L {num(a1 - 8)} {num(ay + 7)}", stroke=c["teal"], width=1.5)
    mid = (a0 + a1) / 2
    fig.text(mid, ay - 14, "replace X", 15, c["teal"], 600, "middle", family="head")
    fig.text(mid, ay + 24, "same values", 11, c["text3"], anchor="middle", family="mono")
    fig.save("anndata-integration", mode)


def main():
    data = json.loads((ASSETS / "performance.json").read_text())
    verify(data)
    for mode, theme in THEMES.items():
        loading(data, mode, theme)
        anndata(data, mode, theme)
        storage(data, mode, theme)
        compression(data, mode, theme)
        billion(data, mode, theme)
    print(
        "Verified source hashes and source-derived values; redrew five figures in light and dark themes."
    )


if __name__ == "__main__":
    main()
