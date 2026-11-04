"""Freeze technical-report results into the data that drives the homepage scenes.

Usage: python home_data.py <CellPack-lab/technical-report/results> [--format-steps <results.json>]

`--format-steps` takes the per-step byte measurements of the CellPack format
(native CSR to packed pages) for the format chapter.

Writes `.vitepress/theme/home/scenes.json` and copies each source file into
`public/research/home/` with its checksum in `sources.json`, so the website
never depends on a neighbouring checkout.
"""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
OUT = DOCS / ".vitepress/theme/home/scenes.json"
PUBLIC = DOCS / "public/research/home"

# Panel (a) of the report's Figure 2, in its order and with its labels.
READ_METHODS = [
    ("parquet (tuned rg128)", "Parquet"),
    ("hfdatasets", "HF Datasets"),
    ("annbatch chunk1", "annbatch"),
    ("memmap", "memmap"),
    ("scdl", "BioNeMo"),
    ("zarr", "Zarr"),
    ("mappedcollection", "H5AD"),
    ("tiledbsoma (sparse X)", "TileDB"),
    ("lance", "Lance"),
    ("cellpack", "CellPack"),
]
PAGE_KB = 4.096
UNIT_CODECS = [("lz4", "LZ4"), ("zstd3", "Zstd-3"), ("zstd22", "Zstd-22"), ("xz9", "XZ-9"), ("rans1-byte", "rANS-1")]
SMOOTH = 5  # seconds, as in the report's billion-cell figure
STEP = 5  # seconds between retained samples


def rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def reads(results: Path):
    table = {r["method"]: r for r in rows(results / "read_amplification.csv")}
    out = []
    for key, label in READ_METHODS:
        r = table[key]
        read, stored = float(r["read_kb"]), float(r["format_kb"])
        out.append({"name": label, "read_kb": read, "stored_kb": stored, "amp": read / max(stored, PAGE_KB)})
    return out


def units(results: Path):
    pooled = [r for r in rows(results / "compression_granularity.csv") if r["matrix"] == "pooled"]
    curves = []
    for key, label in UNIT_CODECS:
        points = sorted((int(r["cells_per_unit"]), round(float(r["mean_ratio"]), 4)) for r in pooled if r["codec"] == key)
        curves.append({"name": label, "points": points})
    cellpack = next(float(r["mean_ratio"]) for r in pooled if r["codec"] == "cellpack")
    return {"codecs": curves, "cellpack": round(cellpack, 4)}


def record(results: Path):
    d = json.loads((results / "format_example.json").read_text())
    return {
        "cell": d["cell"],
        "chunk_cells": d["n_cells"],
        "genes": d["n_genes"],
        "elements": [
            {k: e[k] for k in ("source", "coded", "gap", "rank")} | {"value": round(e["value"], 4), "mode": round(e["mode"], 4)}
            for e in d["elements"]
        ],
        "sizes": d["sizes"],
        "equal_mode": round(d["record_equal_mode"], 4),
        "zero_gaps": round(d["record_zero_gaps"], 4),
    }


def loader(results: Path):
    d = json.loads((results / "loader_example.json").read_text())
    first = d["window"][0]["position"]
    events = [
        {
            "i": w["position"] - first,
            "kind": w["kind"],
            "worker": w["worker"],
            "claim": w["claim"],
            "read": w["read"],
            "cpu": w["cpu"],
            "done": w["done"],
        }
        for w in d["window"]
    ]
    return {"workers": d["workers"], "slot_cells": d["slot_cells"], "slots": d["slots"], "events": events}


def billion(results: Path):
    data = rows(results / "billion_cell_online.csv")
    sources = json.loads((results / "billion_cell_online.sources.json").read_text())
    series = []
    for i in range(0, len(data), STEP):
        window = data[max(0, i - SMOOTH + 1) : i + 1]
        mean = lambda key: sum(float(r[key] or 0) for r in window) / len(window)
        r = data[i]
        series.append([
            round(float(r["seconds"]), 1),
            int(r["cells"]),
            round(mean("cells_per_s")),
            round(mean("read_GBps"), 3),
            round(mean("page_hit_rate"), 4),
            round(int(r["rss_bytes"]) / 1e12, 4),
        ])
    epochs = [{"epoch": e["epoch"], "end": round(e["end_seconds"], 1), "rate": round(e["cells_per_second"])} for e in sources["epochs"]]
    stats = sources["native_stats"]
    last = [r for r in data if float(r["seconds"]) > sources["epochs"][-2]["end_seconds"]]
    last_mean = lambda key: sum(float(r[key] or 0) for r in last) / len(last)
    return {
        "last_epoch": {
            "cells_per_s": round(last_mean("cells_per_s")),
            "page_hit_rate": round(last_mean("page_hit_rate"), 4),
            "read_GBps": round(last_mean("read_GBps"), 3),
        },
        "columns": ["seconds", "cells", "cells_per_s", "read_GBps", "page_hit_rate", "rss_TB"],
        "series": series,
        "epochs": epochs,
        "page_hit_rate": round(stats["page_hits"] / (stats["page_hits"] + stats["page_reads"]), 4),
        "peak_rss_TB": round(sources["process_peak_rss_bytes"] / 1e12, 3),
        "cache_TB": 1.0,
        "devices": "two NVMe SSDs, RAID-0",
    }


# The report's cache-policy figure (Fig. 6): measured three-run means for every
# policy, and Belady's simulated optimum from the same-protocol sweep, exactly as
# the figure draws them.
CACHE_METHODS = [
    ("lru", "LRU"),
    ("clock", "CLOCK"),
    ("arc", "ARC"),
    ("s3fifo", "S3-FIFO"),
    ("tinylfu", "W-TinyLFU"),
    ("topk", "Static top-k"),
    ("cellpack-page-online", "CellPack*"),
    ("cellpack-page", "CellPack"),
]
CACHE_BUDGETS = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"]


def cache(results: Path):
    runs = rows(results / "cache_policy_panel2048_runs.csv")
    sim = rows(results / "cache_policy_sim.csv")
    meta = json.loads((results / "cache_policy_panel2048.json").read_text())
    at = lambda table, key, budget: next(r for r in table if r["method"] == key and float(r["budget"]) == float(budget))
    methods = []
    for key, label in CACHE_METHODS:
        points = [at(runs, key, b) for b in CACHE_BUDGETS]
        methods.append({
            "key": key,
            "name": label,
            "miss": [round(float(r["page_miss_reduction"]), 4) for r in points],
            "cells_per_s": [round(float(r["cells_per_s"])) for r in points],
            "read_GBps": [round(float(r["read_gb_per_s"]), 2) for r in points],
        })
    off = at(runs, "off", 0)
    return {
        "budgets": [float(b) for b in CACHE_BUDGETS],
        "methods": methods,
        "belady": [round(float(at(sim, "belady", b)["page_miss_reduction"]), 4) for b in CACHE_BUDGETS],
        "no_cache": round(float(off["cells_per_s"])),
        "no_cache_read_GBps": round(float(off["read_gb_per_s"]), 2),
        "cells": meta["cells"],
        "repeats": meta["repeats"],
    }


def steps(path: Path):
    """Pooled per-cell bytes for each format step, plus the section details."""
    d = json.loads(path.read_text())
    pooled, sec = d["pooled"], d["pooled"]["sections"]
    by = {s["step"]: s["bytes_per_cell"] for s in pooled["steps"]}
    con = pooled["contrast"]
    return {
        "cells": pooled["cells"],
        "chunks": pooled["chunks"],
        "datasets": len({s["dataset"] for s in d["sample"]}),
        "nnz_per_cell": pooled["nnz_per_cell"],
        "bytes": by,
        "index_bits": {"raw": 16.0, "source": sec["index_gap_source_order"]["bits_per_nnz"], "renumbered": sec["index_gap_renumbered"]["bits_per_nnz"]},
        "zero_gaps": sec["zero_gap_fraction"],
        "value_bits": {
            "raw_u16": d["raw"]["sections"]["value_dictionary"]["bits_per_nnz"],
            "normed_f32": d["normed"]["sections"]["value_dictionary"]["bits_per_nnz"],
        },
        "metadata_per_cell": sec["metadata"]["per_cell"],
        "metadata_parts": sec["metadata"]["parts_per_cell"],
        "padding_per_cell": sec["page_padding"]["per_cell"],
        "contrast": {
            "shared": con["cellpack_records_shared_tables"]["per_cell"],
            "per_cell_rans": con["per_cell_rans_records_only"]["per_cell"],
            "per_cell_rans_with_tables": con["per_cell_rans_with_own_metadata"]["per_cell"],
            "zstd3_per_cell": con["zstd3_per_cell"]["per_cell"],
            "zstd19_per_cell": con["zstd19_per_cell"]["per_cell"],
        },
    }


def main() -> None:
    results = Path(sys.argv[1]).resolve()
    format_steps = Path(sys.argv[sys.argv.index("--format-steps") + 1]).resolve() if "--format-steps" in sys.argv else None
    paper = json.loads((results / "cellpack_paper_values.json").read_text())
    scenes = {
        "reads": reads(results),
        "units": units(results),
        "record": record(results),
        "loader": loader(results),
        "billion": billion(results),
        "cellpack": paper["fig5"],
        "cache": cache(results),
    }
    if format_steps:
        scenes["steps"] = steps(format_steps)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(scenes, separators=(",", ":")) + "\n")
    files = [
        "read_amplification.csv",
        "compression_granularity.csv",
        "format_example.json",
        "loader_example.json",
        "billion_cell_online.csv",
        "billion_cell_online.sources.json",
        "cellpack_paper_values.json",
        "cache_policy_panel2048_runs.csv",
        "cache_policy_panel2048.json",
        "cache_policy_sim.csv",
    ]
    manifest_path = PUBLIC / "sources.json"
    manifest = json.loads(manifest_path.read_text())
    for name in files:
        shutil.copyfile(results / name, PUBLIC / name)
        manifest["sources"][name] = {
            "source": f"CellPack-lab/technical-report/results/{name}",
            "sha256": hashlib.sha256((results / name).read_bytes()).hexdigest(),
        }
    if format_steps:
        shutil.copyfile(format_steps, PUBLIC / "format_steps.json")
        manifest["sources"]["format_steps.json"] = {
            "source": "CellPack/tmp/format-steps/results.json (measured on CellPack-lab/data/cellxgene-10m)",
            "sha256": hashlib.sha256(format_steps.read_bytes()).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(DOCS)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
