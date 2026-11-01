# Implementation notes: `analysis_episode` (memory category 5)

These notes capture design decisions and one deviation from the implementation brief. The user-facing README is owned by the Reporter agent.

## Shape

`analysis_episode` is **append-only**. There is no identity tuple, no `valid_from`/`valid_to`, no merge classifier. Each `record(...)` mints a fresh UUID and writes a brand-new row. The only mutable field is `user_feedback` (and the implicit `updated_at`). Compared to cats 3 and 4 the surface is much smaller: one writer (`record`), one mutator (`update_feedback`), three readers (`get`, `list_recent`, `search`), and one semantic reader (`find_similar`).

## Storage shape — two tables

1. `analysis_episode_memory` — normal sqlite table holding all metadata.
2. `analysis_episode_vec` — sqlite-vec `vec0` virtual table holding the 384-dim embeddings, keyed on `episode_id`, with `project_id` as a (filterable) metadata column.

Both tables live in the same connection. The store enables loadable extensions and calls `sqlite_vec.load(conn)` in its constructor — on macOS, Python's bundled sqlite blocks extensions unless `enable_load_extension(True)` is set first.

## Sync invariant

For every `id` in `analysis_episode_memory` there is exactly one matching `episode_id` in `analysis_episode_vec`, and the vec row's embedding is the embedding of the current `question + conclusion + assumptions + user_feedback` text.

Every write that touches either table is wrapped in `BEGIN IMMEDIATE` so the two writes commit together or not at all:

- `record(...)`: embed (outside the tx so a failure leaves the DB untouched) → `BEGIN IMMEDIATE` → INSERT main → INSERT vec → `COMMIT`.
- `update_feedback(...)`: lookup (outside tx, for early-return) → embed → `BEGIN IMMEDIATE` → UPDATE main → DELETE vec row → INSERT vec row → `COMMIT`.

`DELETE`-then-`INSERT` is the supported update path for `vec0` PRIMARY KEY rows; in-place UPDATE of a vec0 row is not exposed.

## Embedded text

Only the four narrative fields are embedded, concatenated with `"\n\n"`:

```
question + "\n\n" + conclusion + ["\n\n" + assumptions] + ["\n\n" + user_feedback]
```

Structured fields (`analysis_type`, `datasets_used`, `metrics_used`, `methods_used`, `code_refs`, `chart_refs`) are NOT embedded. They are display/filter metadata, not narrative. This also keeps the embed text tight — a row with a long `code_refs` blob doesn't dilute the semantic signal of the question/conclusion.

`search` (substring) covers the same four narrative fields; tests pin this so adding the structured fields to the substring search later would be a deliberate API change, not an accident.

## Embedder

`mads/memory/embedding.py` defines a `Protocol` plus two implementations:

- `SentenceTransformerEmbedder` — production. Lazy-loads `all-MiniLM-L6-v2` (384-dim, CPU). First instantiation in a fresh env triggers an ~80 MB HuggingFace download; subsequent runs reuse the cache. The embedder is created cheaply (no model load until first `.embed()` call), so the store constructor can default to it without forcing a download on import.
- `HashEmbedder` — deterministic test embedder. Lower-cases, tokenizes on word characters, hashes each token via `blake2b(8)`, mods to a 384-bucket histogram, L2-normalizes. Same input → identical vector; shared tokens land in shared buckets, so cosine distance approximately tracks token overlap. Used by every test in `tests/test_analysis_episode.py` so the suite never touches the real model.

Both embedders satisfy the `Embedder` protocol (verified by `isinstance(... , Embedder)`).

## Deviation from the brief: vec0 column kind

The brief asked for `+project_id TEXT` (auxiliary column) so KNN queries could filter on `project_id`. sqlite-vec **rejects WHERE constraints on auxiliary columns inside KNN queries** — auxiliary columns are returnable-only. The error message is literal: "An illegal WHERE constraint was provided on a vec0 auxiliary column in a KNN query."

The fix is to declare `project_id` as a regular (metadata) column with no prefix:

```sql
CREATE VIRTUAL TABLE analysis_episode_vec USING vec0(
    episode_id TEXT PRIMARY KEY,
    project_id TEXT,
    embedding float[384]
);
```

Metadata columns are filterable inside MATCH queries, so:

```sql
SELECT m.*, v.distance
  FROM analysis_episode_vec v
  JOIN analysis_episode_memory m ON m.id = v.episode_id
 WHERE v.embedding MATCH ? AND v.project_id = ? AND v.k = ?
 ORDER BY v.distance ASC
```

pushes the tenancy filter into the KNN scan. No Python-side post-filtering, no over-fetching `k*N`. The result is what the brief wanted (a tenant-safe KNN); just spelled differently.

## Tenancy

`AnalysisEpisodeStore` accepts `user_id` and `project_id` as keyword arguments on every write; the read side accepts `project_id` only. The factory closes over both:

```python
factory = make_analysis_episode_factory(store, user_id="alice", project_id="growth")
```

The 6 `@tool` callables returned by `factory.tools()` have neither `user_id` nor `project_id` in their argument schemas (verified in `test_tool_signatures_hide_tenancy`). Two users in the same project see one shared library; switching `project_id` produces a fresh, empty view of the same store.

## Immutability

The store deliberately exposes only one mutator (`update_feedback`). `test_other_fields_are_immutable_by_design` asserts via `dir()` that no `update_question`, `update_conclusion`, etc. method exists — adding one would be a structural break, not a runtime regression. If a recorder regrets a `question` or `conclusion`, the policy is "record a new episode and let `find_similar` surface both" — not "edit history."

## Cross-cutting

- No imports of cat 1–4 source modules (verified by inspection of `analysis_episode*.py`). The shared `embedding.py` is new infrastructure that future categories may reuse.
- No imports of `ai_data_science_team` anywhere in this category.
- All 52 tests in `tests/test_analysis_episode.py` use `HashEmbedder`, so the suite runs in ~2.3s with no model load. The demo (`eval/demo_analysis_episode_memory.py`) is the only file that exercises `SentenceTransformerEmbedder`, and it prints `loading sentence-transformers model (first run may take 5-15s)...` before instantiation so the operator knows what's happening.
