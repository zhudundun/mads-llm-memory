"""Threading-stress regression tests for the SQLite concurrency bug.

Background
==========

The EDA A/B eval surfaced ``sqlite3.OperationalError: not an error`` and
``cannot start a transaction within a transaction`` on every session A run
of every scenario. langgraph's ``ToolNode`` dispatches tool calls through a
``ThreadPoolExecutor``, and the shared store ``self._conn`` (constructed
with ``check_same_thread=False``) is racy on explicit ``BEGIN
IMMEDIATE`` blocks: when two threads concurrently start ``BEGIN
IMMEDIATE`` on the same connection, SQLite raises because you can't nest
transactions on a single connection.

The fix is a ``threading.RLock`` on each Store (and on MemoryManager) wrapping
every multi-statement transaction. These tests reproduce the original
failure mode and lock down the post-fix invariant: under high thread
contention, every write succeeds and the final row count is correct.

The tests deliberately use file-path DBs (not ``:memory:``) because the bug
is more likely to manifest on a real file due to disk I/O timing — matching
the runtime conditions of the eval harness.
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pytest

from mads.memory.analysis_episode import AnalysisEpisodeStore
from mads.memory.dataset_schema import DatasetSchemaStore
from mads.memory.embedding import HashEmbedder
from mads.memory.metric_definition import MetricDefinitionStore
from mads.memory.user_preference import PreferenceStore


# Threadpool size mirroring the langgraph ToolNode pattern. 8 threads is
# enough to reliably elicit the bug WITHOUT the lock fix on a modern dev
# machine; 16 ups the probability further if a test machine is faster.
NUM_THREADS = 8

# Small sleep inside each task to encourage scheduler interleaving so two
# threads sit at ``BEGIN IMMEDIATE`` at the same time.
INNER_SLEEP_S = 0.001


# ----------------------------------------------------------------------- helpers


def _spawn(fn, args_list, max_workers=NUM_THREADS):
    """Run ``fn(*args)`` for every args tuple in ``args_list`` on a pool.

    Returns ``(results, exceptions)``. Both lists are aligned with
    ``args_list`` order so per-task failures are debuggable.
    """
    results: list = [None] * len(args_list)
    exceptions: list = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, *args): i for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001
                exceptions[i] = e
    return results, exceptions


# ============================================================ MetricDefinitionStore


def test_metric_upsert_under_threads(tmp_path):
    """8 threads upserting distinct metric names share one Store; no errors."""
    db_path = str(tmp_path / "metric_concurrency.sqlite")
    store = MetricDefinitionStore(db_path=db_path)

    def _do_upsert(i: int):
        time.sleep(INNER_SLEEP_S)
        return store.upsert(
            user_id="alice",
            project_id="proj",
            metric_name=f"metric_{i}",
            business_definition=f"the {i}th metric",
            confidence=1.0,
        )

    args = [(i,) for i in range(NUM_THREADS)]
    results, exceptions = _spawn(_do_upsert, args)

    # ANY exception is the bug. Surface the first one so the failure
    # message is informative.
    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"thread {i} raised: {exc!r}")

    # Every upsert should have inserted a new row (all distinct names).
    actions = [r[1] for r in results]
    assert all(a == "inserted" for a in actions), actions

    active = store.list_active(project_id="proj")
    assert len(active) == NUM_THREADS
    names = sorted(m.metric_name for m in active)
    assert names == [f"metric_{i}" for i in range(NUM_THREADS)]
    store.close()


def test_metric_upsert_with_embedder_under_threads(tmp_path):
    """B3 regression: 8 threads upserting distinct metrics with an embedder wired.

    The vec INSERT lives inside the existing ``BEGIN IMMEDIATE``/``COMMIT``
    block; this test verifies the dual-table write stays atomic under
    contention and the lock keeps the shared connection safe across
    threads (sqlite-vec is loaded on the conn — same race surface).
    """
    db_path = str(tmp_path / "metric_b3_concurrency.sqlite")
    store = MetricDefinitionStore(db_path=db_path, embedder=HashEmbedder())

    def _do_upsert(i: int):
        time.sleep(INNER_SLEEP_S)
        return store.upsert(
            user_id="alice",
            project_id="proj",
            metric_name=f"metric_{i}",
            business_definition=f"the {i}th metric with semantic body",
        )

    args = [(i,) for i in range(NUM_THREADS)]
    results, exceptions = _spawn(_do_upsert, args)

    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"thread {i} raised: {exc!r}")

    actions = [r[1] for r in results]
    assert all(a == "inserted" for a in actions), actions

    # Both tables agree: NUM_THREADS active rows AND NUM_THREADS vec rows.
    active = store.list_active(project_id="proj")
    assert len(active) == NUM_THREADS
    vec_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM metric_vec"
    ).fetchone()["c"]
    assert vec_count == NUM_THREADS, (
        f"vec table out of sync: {vec_count} vec rows vs {NUM_THREADS} metric rows"
    )
    store.close()


# ============================================================ DatasetSchemaStore


def test_dataset_register_column_under_threads(tmp_path):
    """8 threads concurrently register distinct columns on the same dataset."""
    db_path = str(tmp_path / "dataset_concurrency.sqlite")
    store = DatasetSchemaStore(db_path=db_path)

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    _, fingerprint = store.register_dataset(
        user_id="alice",
        df=df,
        dataset_name="d",
        table_name="t",
    )

    def _do_register(i: int):
        time.sleep(INNER_SLEEP_S)
        return store.register_column(
            fingerprint=fingerprint,
            table_name="t",
            column_name=f"col_{i}",
            semantic_type="string",
            business_meaning=f"the {i}th col",
        )

    args = [(i,) for i in range(NUM_THREADS)]
    _, exceptions = _spawn(_do_register, args)

    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"thread {i} raised: {exc!r}")

    cols = store.list_columns(fingerprint=fingerprint, table_name="t")
    names = sorted(c["column_name"] for c in cols)
    assert names == [f"col_{i}" for i in range(NUM_THREADS)], names
    store.close()


# ============================================================ AnalysisEpisodeStore


def test_episode_record_under_threads(tmp_path):
    """8 threads concurrently record analysis episodes; all 8 persist."""
    db_path = str(tmp_path / "episode_concurrency.sqlite")
    store = AnalysisEpisodeStore(db_path=db_path, embedder=HashEmbedder())

    def _do_record(i: int):
        time.sleep(INNER_SLEEP_S)
        return store.record(
            user_id="alice",
            project_id="proj",
            question=f"what is question {i}?",
            conclusion=f"the answer is {i}",
        )

    args = [(i,) for i in range(NUM_THREADS)]
    results, exceptions = _spawn(_do_record, args)

    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"thread {i} raised: {exc!r}")

    persisted = store.list_recent(project_id="proj", limit=NUM_THREADS + 5)
    assert len(persisted) == NUM_THREADS
    # Verify all episode ids returned by record() come back from list_recent.
    recorded_ids = {r.id for r in results}
    persisted_ids = {e.id for e in persisted}
    assert recorded_ids == persisted_ids
    store.close()


# ============================================================ Mixed read+write


def test_mixed_read_write_under_threads(tmp_path):
    """4 writer threads + 4 reader threads on one Store; no errors,
    final state consistent."""
    db_path = str(tmp_path / "mixed_concurrency.sqlite")
    store = MetricDefinitionStore(db_path=db_path)

    # Seed one metric so readers always have something to find.
    store.upsert(
        user_id="alice",
        project_id="proj",
        metric_name="seed",
        business_definition="seed metric",
    )

    NUM_WRITERS = 4
    NUM_READERS = 4

    def _write(i: int):
        time.sleep(INNER_SLEEP_S)
        return store.upsert(
            user_id="alice",
            project_id="proj",
            metric_name=f"writer_{i}",
            business_definition=f"writer {i}'s metric",
        )

    def _read(i: int):
        time.sleep(INNER_SLEEP_S)
        active = store.list_active(project_id="proj")
        seed = store.get_active(project_id="proj", metric_name="seed")
        return (len(active), seed.metric_name if seed else None)

    with ThreadPoolExecutor(max_workers=NUM_WRITERS + NUM_READERS) as ex:
        futures = []
        for i in range(NUM_WRITERS):
            futures.append(("write", i, ex.submit(_write, i)))
        for i in range(NUM_READERS):
            futures.append(("read", i, ex.submit(_read, i)))

        for kind, i, fut in futures:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"{kind} thread {i} raised: {e!r}")

    # Final state: seed + 4 writer rows.
    active = store.list_active(project_id="proj")
    names = sorted(m.metric_name for m in active)
    assert names == ["seed", "writer_0", "writer_1", "writer_2", "writer_3"], names
    store.close()


# ============================================================ PreferenceStore (v1.3 dataset axis)


def test_preference_set_mixed_scopes_under_threads(tmp_path):
    """8 threads hammering ``set()`` with a mix of global / project / dataset /
    project+dataset scopes — verifying the four-axis active-row index
    + lock hold under contention with the new column."""
    db_path = str(tmp_path / "pref_concurrency.sqlite")
    store = PreferenceStore(db_path=db_path)

    # Each thread sets a unique (key, scope-tuple) so we expect 8 distinct
    # active rows. The scope assignment cycles deterministically to mix
    # all four scope variants.
    def _do_set(i: int):
        time.sleep(INNER_SLEEP_S)
        scope_mode = i % 4
        if scope_mode == 0:
            pid, fp = None, None
            tag = "g"
        elif scope_mode == 1:
            pid, fp = f"proj_{i}", None
            tag = "p"
        elif scope_mode == 2:
            pid, fp = None, f"fp_{i}"
            tag = "d"
        else:
            pid, fp = f"proj_{i}", f"fp_{i}"
            tag = "pd"
        return store.set(
            "alice",
            f"k_{i}",
            f"{tag}_v_{i}",
            project_id=pid,
            dataset_fingerprint=fp,
        )

    args = [(i,) for i in range(NUM_THREADS)]
    results, exceptions = _spawn(_do_set, args)

    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"thread {i} raised: {exc!r}")

    # Every set should have inserted a new row id.
    assert all(isinstance(r, str) and len(r) == 32 for r in results)

    # Verify each row landed at the right scope.
    for i in range(NUM_THREADS):
        scope_mode = i % 4
        if scope_mode == 0:
            row = store.get("alice", f"k_{i}")
            assert row["project_id"] is None
            assert row["dataset_fingerprint"] is None
        elif scope_mode == 1:
            row = store.get("alice", f"k_{i}", project_id=f"proj_{i}")
            assert row["project_id"] == f"proj_{i}"
            assert row["dataset_fingerprint"] is None
        elif scope_mode == 2:
            row = store.get("alice", f"k_{i}", dataset_fingerprint=f"fp_{i}")
            assert row["project_id"] is None
            assert row["dataset_fingerprint"] == f"fp_{i}"
        else:
            row = store.get(
                "alice", f"k_{i}",
                project_id=f"proj_{i}",
                dataset_fingerprint=f"fp_{i}",
            )
            assert row["project_id"] == f"proj_{i}"
            assert row["dataset_fingerprint"] == f"fp_{i}"

    store.close()


def test_preference_same_key_four_scopes_under_threads(tmp_path):
    """4 threads each writing the SAME key at a different scope — verifies
    the four scope variants coexist without index collisions under
    contention. Run several times to elicit any latent race."""
    db_path = str(tmp_path / "pref_four_scope.sqlite")
    store = PreferenceStore(db_path=db_path)

    def _do_set(scope_mode: int):
        time.sleep(INNER_SLEEP_S)
        if scope_mode == 0:
            return store.set("alice", "k", "g_v")
        if scope_mode == 1:
            return store.set("alice", "k", "p_v", project_id="proj1")
        if scope_mode == 2:
            return store.set("alice", "k", "d_v", dataset_fingerprint="fp1")
        return store.set(
            "alice", "k", "pd_v",
            project_id="proj1", dataset_fingerprint="fp1",
        )

    args = [(i,) for i in range(4)]
    _, exceptions = _spawn(_do_set, args, max_workers=4)
    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"scope {i} raised: {exc!r}")

    # All four scope variants are active simultaneously.
    assert store.get("alice", "k")["preference_value"] == "g_v"
    assert store.get("alice", "k", project_id="proj1")["preference_value"] == "p_v"
    assert (
        store.get("alice", "k", dataset_fingerprint="fp1")["preference_value"]
        == "d_v"
    )
    assert (
        store.get(
            "alice", "k", project_id="proj1", dataset_fingerprint="fp1"
        )["preference_value"]
        == "pd_v"
    )
    store.close()


# ============================================================ WAL pragma


def test_wal_mode_enabled_on_file_path_db(tmp_path):
    """File-path stores should open in WAL mode for concurrent reads."""
    db_path = str(tmp_path / "wal_check.sqlite")
    store = MetricDefinitionStore(db_path=db_path)

    mode = store._conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal", f"expected wal, got {mode!r}"
    store.close()


def test_memory_db_skips_wal(tmp_path):
    """``:memory:`` DBs cannot use WAL — make sure the constructor doesn't try."""
    store = MetricDefinitionStore(db_path=":memory:")
    mode = store._conn.execute("PRAGMA journal_mode;").fetchone()[0]
    # In-memory DBs report ``memory`` for journal_mode (or sometimes the
    # default ``delete`` if the PRAGMA is a no-op). Either way, NOT wal.
    assert mode.lower() != "wal", f"expected non-wal, got {mode!r}"
    store.close()


# ============================================================ Shared MemoryManager


def test_memory_manager_mixed_writes_under_threads(tmp_path):
    """One MemoryManager (shared conn) — multiple stores hit concurrently.

    This is the closest reproduction of the exact eval failure: ``MemoryManager``
    owns ``self._conn``, and three stores (metric / dataset / episode)
    share it; langgraph dispatches their tools across threads.
    """
    from mads.manager import MemoryManager

    db_path = str(tmp_path / "manager_concurrency.sqlite")
    mgr = MemoryManager(
        user_id="alice",
        project_id="proj",
        db_path=db_path,
        embedder=HashEmbedder(),
    )

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mgr.bind_dataframe(df, dataset_name="d", table_name="t")
    # Eagerly initialise all three lazy stores so the first concurrent
    # access in the worker threads doesn't race on the property getter.
    # In real usage (wrapper / eval harness) this priming happens during
    # setup before tool dispatch begins.
    _ = mgr._dataset_store_
    _ = mgr._metric_store_
    _ = mgr._episode_store_
    # Bootstrap a dataset row so we have a fingerprint for column writes.
    _, fingerprint = mgr._dataset_store_.register_dataset(
        user_id="alice", df=df, dataset_name="d", table_name="t"
    )

    def _metric_write(i: int):
        time.sleep(INNER_SLEEP_S)
        return mgr._metric_store_.upsert(
            user_id="alice",
            project_id="proj",
            metric_name=f"m_{i}",
            business_definition=f"def {i}",
        )

    def _column_write(i: int):
        time.sleep(INNER_SLEEP_S)
        return mgr._dataset_store_.register_column(
            fingerprint=fingerprint,
            table_name="t",
            column_name=f"col_{i}",
        )

    def _episode_write(i: int):
        time.sleep(INNER_SLEEP_S)
        return mgr._episode_store_.record(
            user_id="alice",
            project_id="proj",
            question=f"q_{i}",
            conclusion=f"c_{i}",
        )

    tasks: list = []
    for i in range(3):
        tasks.append(("metric", i, _metric_write))
        tasks.append(("column", i, _column_write))
        tasks.append(("episode", i, _episode_write))

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        futures = [(kind, i, ex.submit(fn, i)) for kind, i, fn in tasks]
        for kind, i, fut in futures:
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"{kind} {i} raised: {e!r}")

    metrics = mgr._metric_store_.list_active(project_id="proj")
    assert len(metrics) == 3, [m.metric_name for m in metrics]
    cols = mgr._dataset_store_.list_columns(
        fingerprint=fingerprint, table_name="t"
    )
    assert len(cols) == 3, [c["column_name"] for c in cols]
    eps = mgr._episode_store_.list_recent(project_id="proj", limit=10)
    assert len(eps) == 3, [e.question for e in eps]
    mgr.close()


# ================================================ v1.4: required_columns under threads


def test_metric_upsert_with_required_columns_under_threads(tmp_path):
    """v1.4 regression: 8 threads upserting distinct metrics each carrying
    a required_columns payload share one Store; no errors and all rows
    persist with the right required_columns_json.

    The new column write lives inside the existing BEGIN IMMEDIATE/COMMIT
    block; this test verifies the additional payload doesn't perturb the
    serialiser under contention.
    """
    db_path = str(tmp_path / "metric_required_concurrency.sqlite")
    store = MetricDefinitionStore(db_path=db_path)

    def _do_upsert(i: int):
        time.sleep(INNER_SLEEP_S)
        return store.upsert(
            user_id="alice",
            project_id="proj",
            metric_name=f"metric_{i}",
            business_definition=f"the {i}th metric with required cols",
            required_columns=[f"col_{i}_a", f"col_{i}_b"],
        )

    args = [(i,) for i in range(NUM_THREADS)]
    results, exceptions = _spawn(_do_upsert, args)

    for i, exc in enumerate(exceptions):
        if exc is not None:
            pytest.fail(f"thread {i} raised: {exc!r}")

    # Every row landed with its required_columns intact.
    actives = store.list_active(project_id="proj")
    assert len(actives) == NUM_THREADS
    for m in actives:
        # Each metric_N has required_columns == ['col_N_a', 'col_N_b'].
        idx = m.metric_name.split("_")[1]
        assert m.required_columns == [f"col_{idx}_a", f"col_{idx}_b"]
    store.close()
