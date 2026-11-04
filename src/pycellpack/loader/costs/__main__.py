"""`python -m pycellpack.loader.costs`: the cost records of this machine.

    list                              records under $CELLPACK_HOME
    show FILE...                      the cost model of a loader over FILEs and where each part comes from
    model FILE... -o OUT              save that model as JSON (for the Rust crate's `serde` feature)
    calibrate FILE... [--steady S]    measure it
    export KIND KEY OUT | import IN   move one record between machines
    rm KIND KEY                       remove one record

FILEs are `.cpk` files loaded with the built-in dense output, each file's
column c given standard id c (a cost measurement, not a gene panel).
"""

from __future__ import annotations

import argparse
import json
import sys

from ._store import KINDS, CostStore


def _loader(files, workers):
    """A loader over `FILE`s with the built-in dense output; column c of each
    file takes standard id c, which is a cost measurement rather than a gene
    panel."""
    import numpy as np

    from ... import cellpack, loader

    widths = [cellpack.File(f).shape[1] for f in files]
    matrices = [(f, np.arange(n, dtype=np.uint32)) for f, n in zip(files, widths)]
    return loader.Loader(loader.DataSource(matrices, max(widths)), workers=workers)


def _summary(record):
    """One line summarizing a record: the identity it applies to and its key
    rates or curve."""
    v, m = record["values"], record.get("measured", {})
    if record["kind"] == "storage":
        i = record["identity"]
        return f"{i.get('fs')}:{i.get('source')} {i.get('backend')} direct_io={i.get('direct_io')} iops={v['iops']:.3g} bw={v['bandwidth']:.3g}"
    if record["kind"] == "host":
        i = record["identity"]
        return f"{i.get('cpu')} {i.get('kernel_set')} v{i.get('version')} access={v['access']:.3g}"
    return f"process={v.get('process', 0):.3g} bands={len(v.get('curve', ()))} runs={m.get('runs', 0)}"


def main(argv=None):
    """Runs one subcommand; `argv` defaults to the process's arguments."""
    p = argparse.ArgumentParser(prog="python -m pycellpack.loader.costs", description=(__doc__ or "CellPack cost records").splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("show", "model", "calibrate"):
        q = sub.add_parser(name)
        q.add_argument("files", nargs="+")
        q.add_argument("--workers", type=int, default=None)
        if name == "model":
            q.add_argument("-o", "--out", required=True)
        if name == "calibrate":
            q.add_argument("--steady", type=float, default=30.0, help="steady seconds per run")
            q.add_argument("--budget", type=int, default=None, help="cache bytes of the plan runs")
            q.add_argument("--force", action="store_true", help="save even unsteady or misfit calibrations")
            q.add_argument("--no-save", action="store_true")
    q = sub.add_parser("export")
    q.add_argument("kind", choices=KINDS)
    q.add_argument("key")
    q.add_argument("out")
    q = sub.add_parser("import")
    q.add_argument("file")
    q = sub.add_parser("rm")
    q.add_argument("kind", choices=KINDS)
    q.add_argument("key")
    a = p.parse_args(argv)
    store = CostStore()

    if a.command == "list":
        for r in store.records():
            print(f"{r['kind']:8s} {r['key']}  {r['updated']}  {_summary(r)}")
    elif a.command in ("show", "model"):
        from ._resolve import resolve

        model = resolve(_loader(a.files, a.workers), store=store)
        if a.command == "model":
            model.save(a.out)
        else:
            print(json.dumps({"provenance": dict(model.provenance), "model": model.to_dict()}, indent=1, default=dict))
    elif a.command == "calibrate":
        from ._calibrate import calibrate

        model = calibrate(_loader(a.files, a.workers), steady_seconds=a.steady, budget=a.budget,
                          save=not a.no_save, force=a.force, store=store)
        print(json.dumps({"provenance": dict(model.provenance), "model": model.to_dict()}, indent=1, default=dict))
    elif a.command == "export":
        record = store.get(a.kind, a.key)
        if record is None:
            sys.exit(f"no {a.kind} record {a.key}")
        with open(a.out, "w") as f:
            json.dump(record, f, indent=1)
    elif a.command == "import":
        with open(a.file) as f:
            r = json.load(f)
        store.put(r["kind"], r["key"], r["identity"], r["values"], r.get("measured"))
        print(f"imported {r['kind']} {r['key']}")
    elif a.command == "rm" and not store.remove(a.kind, a.key):
        sys.exit(f"no {a.kind} record {a.key}")


if __name__ == "__main__":
    main()
