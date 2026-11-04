"""Real two-rank Gloo runs with a separate rank-zero data process."""

from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import signal
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pycellpack import cellpack, loader


def _rank(rank, rendezvous, path, queue, mode):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=25),
    )
    try:
        source = path
        if mode == "embedded" and rank == 0:
            from pycellpack.io import open_matrix

            source = open_matrix(path)
        def build():
            return loader.Loader(loader.DataSource([(source, np.arange(4))], 4),
                                 workers=2, inflight=2, backend="pread", slot_rows=2)

        loading = loader.DistributedLoader(build if rank == 0 else None)
        accesses = (
            np.array([8, 3, 2, 3, 1, 0, 9, 7, 4, 6, 2, 5, 0, 1], np.uint32)
            if rank == 0
            else None
        )
        if mode == "empty" and rank == 0:
            accesses = np.empty(0, np.uint32)
        batches = []
        if mode == "missing":
            try:
                loading.start(accesses, slots_per_rank=1, drop_last=True)
            except RuntimeError as error:
                queue.put({"rank": rank, "error": str(error), "pid": None})
            else:
                raise AssertionError("a missing matrix started")
            return
        with loading.start(accesses, slots_per_rank=1, drop_last=True) as run:
            producer_pid = run.producer_pid
            assert producer_pid != os.getpid()
            names = [spec["name"] for spec in run._shared.descriptor["regions"].values()]
            if mode in ("memfd", "file"):
                assert run._shared.descriptor["transport"] == mode
            if mode == "kill" and rank == 0:
                os.kill(producer_pid, signal.SIGKILL)
            if mode == "cancel" and rank == 1:
                run.cancel()
            dist.barrier()
            if mode in ("kill", "cancel"):
                try:
                    for batch in run:
                        batch.release()
                except RuntimeError as error:
                    queue.put({"rank": rank, "error": str(error), "pid": producer_pid})
                else:
                    raise AssertionError("failed producer yielded a batch")
            else:
                count = len(run)
                last_view = None
                for batch in run:
                    with batch:
                        x = batch.data
                        batches.append((batch.index, batch.first_position, x.copy()))
                        last_view = x
                        del x
                        # Retaining the previous slot prevents its next turn.
                        # Only keep a view once this rank reaches its final batch.
                        if len(batches) != count:
                            last_view = None
                stats = run.join()
                queue.put(
                    {
                        "rank": rank,
                        "own_pid": os.getpid(),
                        "pid": producer_pid,
                        "batches": batches,
                        "stats": stats.to_dict(),
                        "count": count,
                    }
                )
        if mode not in ("missing", "kill", "cancel") and last_view is not None:
            np.testing.assert_array_equal(last_view, batches[-1][2])
        dist.barrier()
        if rank == 0:
            assert not any((Path("/dev/shm") / name).exists() for name in names)
            if mode == "file":
                assert not list(Path(os.environ["PYCELLPACK_SHM_DIR"]).iterdir())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "mode", ["success", "embedded", "empty", "missing", "kill", "cancel", "memfd", "file"]
)
def test_rank_zero_process_and_shared_consumers(tmp_path, monkeypatch, mode):
    if mode in ("memfd", "file"):
        # Ranks inherit the transport choice; regions keep their sizes.
        (tmp_path / "shm").mkdir()
        monkeypatch.setenv("PYCELLPACK_SHM", mode)
        monkeypatch.setenv("PYCELLPACK_SHM_DIR", str(tmp_path / "shm"))
    path = tmp_path / "matrix.cpk"
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4)
    with cellpack.Writer(path, 4, np.float32, format="dense", chunk_rows=3) as writer:
        writer.append(matrix)
    if mode == "missing":
        path = tmp_path / "missing.cpk"
    if mode == "embedded":
        ad = pytest.importorskip("anndata")
        from pycellpack.io import write_h5cp

        path = tmp_path / "cells.h5cp"
        write_h5cp(path, ad.AnnData(matrix), encode=True)
    context = mp.get_context("spawn")
    queue = context.Queue()
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    processes = [
        context.Process(target=_rank, args=(rank, rendezvous, str(path), queue, mode))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=35) for _ in range(2)]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
        queue.close()
    assert results[0]["pid"] == results[1]["pid"]
    if mode in ("missing", "kill", "cancel"):
        assert all("error" in r for r in results)
        messages = [r["error"] for r in results]
        if mode == "missing":
            assert all("No such file" in m or "not found" in m for m in messages)
        elif mode == "kill":
            assert all("-9" in m for m in messages)
        else:
            assert all("cancelled" in m for m in messages)
    else:
        assert results[0]["pid"] not in {p.pid for p in processes}
        assert all(
            r["stats"]["accesses"] == (0 if mode == "empty" else 12) for r in results
        )
        batches = sorted(
            [batch for r in results for batch in r["batches"]], key=lambda b: b[0]
        )
        if mode != "empty":
            assert all(r["count"] == 3 for r in results)
            accesses = [8, 3, 2, 3, 1, 0, 9, 7, 4, 6, 2, 5]
            np.testing.assert_array_equal(
                np.concatenate([b[2] for b in batches]), matrix[accesses]
            )
        else:
            assert batches == []
