# Analysis Episode Memory

Fifth of the 8 memory categories defined for `mads` (see `CLAUDE.md`).
Stores immutable, project-scoped snapshots of analytical investigations
— question, conclusion, datasets/metrics/methods/code/charts,
assumptions, and any user feedback collected later. Backed by a normal
SQLite metadata table plus a sibling `sqlite-vec` virtual table that
holds 384-dim embeddings for semantic recall.

## When to use this

Analyst pain points: rediscovering past investigations from scratch,
losing context across sessions, and answering "did we already analyze
this?" without grepping notebooks. Three motivating examples: the
revenue drop on 2026-05-15 (root-cause, paid-search mix shift), the
churn model improvement on 2026-05-22 (adding tenure and
last-7d-sessions features), and the prior retention analysis
(signup-week cohorts, `is_test_user=false` exclusion).

## Architectural place

Category 5 in `CLAUDE.md`'s 8-category table maps to the
`analysis_conclusion` enum slot; cat 5 ships in its own dedicated table
(same pattern as cats 1-4). Cat 5 is structurally different in two
ways: it is the **first append-only category** (episodes are events,
not definitions — no identity tuple, no `valid_from`/`valid_to`, no
merge classifier), and the **first with semantic retrieval** (an
embedder populates a `sqlite-vec` virtual table joined by `episode_id`).

The implementer also flagged that `CLAUDE.md`'s "Don't add a second
vector store. We're on pgvector." line is stale: the project is on
SQLite + `sqlite-vec`; pgvector is not wired in.

## Schema

From `infra/migrations/0005_analysis_episode_memory.sql`:

```sql
CREATE TABLE IF NOT EXISTS analysis_episode_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    question TEXT NOT NULL,
    analysis_type TEXT,
    datasets_used TEXT,
    metrics_used TEXT,
    methods_used TEXT,
    code_refs TEXT,
    chart_refs TEXT,
    conclusion TEXT NOT NULL,
    assumptions TEXT,
    user_feedback TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE INDEX IF NOT EXISTS idx_episode_project_created
    ON analysis_episode_memory(project_id, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS analysis_episode_vec USING vec0(
    episode_id TEXT PRIMARY KEY,
    project_id TEXT,
    embedding float[384]
);
```

| Column | Meaning |
|---|---|
| `id` | 32-char `uuid4().hex`; primary key, immutable. |
| `user_id` | Recorder. Audit-only; does NOT gate read visibility. |
| `project_id` | Tenancy boundary. Two users in one project share one library. |
| `question`, `conclusion` | Required, non-empty. Embedded. |
| `assumptions`, `user_feedback` | Optional. Embedded if present. `user_feedback` is the one mutable field. |
| `analysis_type`, `datasets_used`, `metrics_used`, `methods_used`, `code_refs`, `chart_refs` | Optional free-text metadata. NOT embedded. |
| `created_at`, `updated_at` | UTC ISO-8601. `created_at` is never mutated; `updated_at` is bumped only by `update_feedback`. |

`analysis_episode_vec` is a `vec0` virtual table sharing the SQLite
connection, joined on `episode_id = id`. Beyond the join key it holds
the 384-dim `embedding` plus `project_id` as a filterable metadata
column. It is invisible to reads other than `find_similar`.

## Append-only model & mutability

Cat 5 is structurally different from cats 1-4: episodes are events, not
definitions. Consequences:

- **No versioning.** Each `record(...)` mints a fresh UUID. No identity
  tuple, no `valid_from`/`valid_to`, no upsert classifier returning
  `inserted` / `noop` / `updated_in_place` / `versioned`.
- **The ONLY mutable field is `user_feedback`.** The store exposes one
  mutator (`update_feedback`); no `update_question`, `update_conclusion`,
  `update_analysis_type`, `update_assumptions`, etc. Pinned via
  `dir(AnalysisEpisodeStore)` in the test suite.
- **No `get_history`.** The timeline is `created_at` plus the single
  optional `updated_at`. `update_feedback` overwrites prior feedback —
  no audit chain.
- **Policy for regret.** If the recorder regrets a `question` or
  `conclusion`, record a new episode and let `find_similar` surface
  both — don't edit history.

## Embedding pipeline

`mads/memory/embedding.py` defines a `runtime_checkable` `Embedder`
protocol (`dim: int` + `embed(text: str) -> list[float]`) with two
implementations:

- **`SentenceTransformerEmbedder`** — production. Wraps
  `sentence-transformers`' `all-MiniLM-L6-v2` (384-dim, CPU),
  lazy-loaded on first `embed()`. First run triggers an ~80 MB
  HuggingFace download; subsequent runs reuse the cache.
  `normalize_embeddings=True`.
- **`HashEmbedder`** — deterministic test embedder. Lowercases,
  tokenizes on `[A-Za-z0-9]+`, hashes each token via `blake2b(8)`, mods
  to a 384-bucket histogram, L2-normalizes. Shared tokens land in
  shared buckets, so cosine distance tracks token overlap.

**Embedded text.** Only the four free-text fields are concatenated with
`"\n\n"`, skipping any None:

```
question + "\n\n" + conclusion [+ "\n\n" + assumptions] [+ "\n\n" + user_feedback]
```

Structured fields (`analysis_type`, `datasets_used`, `metrics_used`,
`methods_used`, `code_refs`, `chart_refs`) are intentionally NOT
embedded — they are filter/display metadata.

**Sync invariant.** Every row in `analysis_episode_memory` has exactly
one matching row in `analysis_episode_vec` (by `episode_id`), and the
vec row's embedding reflects the *current* embedded text. Every write
to either table is wrapped in `BEGIN IMMEDIATE`. On `update_feedback`
the vec row is replaced via DELETE-then-INSERT (in-place UPDATE of a
`vec0` PRIMARY KEY row is not exposed).

## Vector search and tenancy

`find_similar(project_id, query, k)` embeds the query with the store's
own `Embedder` (so embed-time and search-time embedders cannot diverge),
then runs one KNN scan with tenancy pushed into the MATCH clause:

```sql
SELECT m.*, v.distance AS distance
  FROM analysis_episode_vec AS v
  JOIN analysis_episode_memory AS m ON m.id = v.episode_id
 WHERE v.embedding MATCH ? AND v.project_id = ? AND v.k = ?
 ORDER BY v.distance ASC
```

Returns `[(AnalysisEpisode, distance)]` sorted by ascending distance.

The original brief specified `+project_id` (auxiliary column);
`sqlite-vec` rejects WHERE constraints on auxiliary columns inside KNN
— literal error: *"An illegal WHERE constraint was provided on a vec0
auxiliary column in a KNN query."* The fix was to declare `project_id`
as an unprefixed metadata column, which `sqlite-vec` accepts inside
MATCH. End result is what the brief wanted — a tenant-safe KNN with no
Python-side post-filtering — spelled differently.

## Python API

`AnalysisEpisode` is a pydantic v2 model mirroring the 15 table
columns.

```python
class AnalysisEpisodeStore:
    def __init__(self, db_path=None, *, conn=None, embedder=None) -> None: ...
    def init_schema(self) -> None: ...
    def record(self, *, user_id, project_id, question, conclusion,
               analysis_type=None, datasets_used=None, metrics_used=None,
               methods_used=None, code_refs=None, chart_refs=None,
               assumptions=None, user_feedback=None) -> AnalysisEpisode: ...
    def update_feedback(self, *, user_id, project_id, episode_id,
                        feedback) -> Optional[AnalysisEpisode]: ...
    def get(self, *, project_id, episode_id) -> Optional[AnalysisEpisode]: ...
    def list_recent(self, *, project_id, limit=10) -> list[AnalysisEpisode]: ...
    def search(self, *, project_id, query) -> list[AnalysisEpisode]: ...
    def find_similar(self, *, project_id, query, k=5
                     ) -> list[tuple[AnalysisEpisode, float]]: ...
    def close(self) -> None: ...
```

`__init__` opens/reuses the connection, enables the loadable-extension
toggle (`sqlite-vec` requires it on macOS), loads `sqlite_vec`, defaults
the embedder to `SentenceTransformerEmbedder()`, applies the migration.
`record` validates, embeds outside the transaction, then INSERTs both
tables in one `BEGIN IMMEDIATE`. `update_feedback` re-embeds and
DELETE+INSERTs the vec row in one transaction; returns `None` if no
matching episode. `get` / `list_recent` / `search` are project-scoped
reads against the main table; `find_similar` runs the KNN scan above.

## LangChain tool surface

`make_analysis_episode_factory(store, user_id, project_id).tools()`
returns 6 `@tool`-decorated callables. None of the LLM-visible
signatures contain `user_id` or `project_id`.

| Tool | LLM-visible signature | Purpose |
|---|---|---|
| `record_analysis_episode` | `(question, conclusion, analysis_type=None, datasets_used=None, metrics_used=None, methods_used=None, code_refs=None, chart_refs=None, assumptions=None, user_feedback=None) -> str` | Record a new immutable episode. Free-text fields are concatenated and embedded. Returns the new id. |
| `get_analysis_episode` | `(episode_id: str) -> str` | Fetch one episode by id, pretty-printed. Empty-state string if absent. |
| `list_recent_analyses` | `(limit: int = 10) -> str` | Newest-first list, capped at `limit`. Compact table: id (short), created_at, question, conclusion. |
| `find_similar_analyses` | `(query: str, k: int = 5) -> str` | Semantic search: embed `query` and return the `k` most similar episodes by cosine distance. Use when you want analyses similar in MEANING. |
| `search_analyses` | `(query: str) -> str` | Substring search (case-insensitive) over question, conclusion, assumptions, user_feedback. Use for exact-term lookup. |
| `update_user_feedback` | `(episode_id: str, feedback: str) -> str` | Update the ONE mutable field. Bumps `updated_at` and re-embeds the row. |

All optional `record_analysis_episode` parameters are `Optional[str] = None`. Substring for known tokens (`is_test_user`, `root_cause`); semantic for intent-only queries.

## Tenancy contract

`CLAUDE.md`: *"All memory queries are user-scoped at the store layer.
`WHERE user_id=…` lives in `memory/store.py`, not in prompts."* For cat
5 the scoping column is `project_id` — episodes are a property of the
project, not the recorder. `user_id` is audit-only. The factory closes
over both:

```python
from mads.memory import AnalysisEpisodeStore, make_analysis_episode_factory
store = AnalysisEpisodeStore(":memory:")
factory = make_analysis_episode_factory(store, user_id="wang", project_id="growth")
tools = factory.tools()   # 6 BaseTool instances; closures hold the triple
```

`AnalysisEpisodeToolFactory.__init__` rejects empty `user_id` or
`project_id`.

## Usage example

Transcript-style; derived from `eval/demo_analysis_episode_memory.py`.

```text
>>> factory = make_analysis_episode_factory(store, "alice", "growth")
>>> by_name = {t.name: t for t in factory.tools()}

# record the revenue-drop investigation
>>> by_name["record_analysis_episode"].invoke({
...     "question": "Why did revenue drop on 2026-05-15?",
...     "conclusion": "Revenue drop was mainly caused by paid-search traffic mix shift.",
...     "analysis_type": "root_cause", "datasets_used": "payments, traffic_attribution"})
"Recorded analysis episode <uuid> (project=growth, created_at=...)."

# list recent (newest first)
>>> by_name["list_recent_analyses"].invoke({"limit": 5})
"Most recent analysis episodes in project 'growth' (3): ..."

# semantic recall against a paraphrased question
>>> by_name["find_similar_analyses"].invoke({
...     "query": "Did the revenue drop come from a specific traffic source?", "k": 2})
"Semantic matches for '...' in project 'growth' (2): ..."

# attach feedback — re-embeds the row
>>> by_name["update_user_feedback"].invoke({"episode_id": revenue_id, "feedback": "Follow-up: paid search recovered the next week; one-off mix-shift."})
"Updated user_feedback on episode <uuid> (updated_at=...)."

# re-embedding now ranks episode 1 at top for the new query text
>>> by_name["find_similar_analyses"].invoke({"query": "paid search recovery", "k": 2})
"Semantic matches for 'paid search recovery' in project 'growth' (2): ..."
```

## Testing

```bash
pytest tests/test_analysis_episode.py -q   # 52 passed
```

Coverage: `HashEmbedder` properties (7), record + get (8), list +
substring search (11), semantic `find_similar` (5), mutability incl.
assertion that no other mutators exist (6), tenancy isolation (4), tool
layer incl. hidden-tenancy schemas (11). All tests use `HashEmbedder`;
suite runs in ~2.3s with no model load.

## Demo

```bash
python eval/demo_analysis_episode_memory.py
```

12 labeled steps against an in-memory store using the real
`SentenceTransformerEmbedder`: build factory, record the three
motivating episodes, list, fetch one, substring + semantic search,
attach feedback (re-embed), re-fetch, fresh semantic query the
re-embedding ranks at top, switch `project_id` for isolation. First
invocation downloads the ~80 MB model; subsequent runs reuse the cache.

## Dependencies

New pip installs (pinned in `.venv`, not in any project manifest):

- `sqlite-vec==0.1.9` — the `vec0` virtual-table extension.
- `sentence-transformers==5.5.1` — embedder backbone; transitively pulls
  `torch`, `transformers`, `huggingface-hub`, `tokenizers`,
  `scikit-learn`, `scipy`, `numpy`.

There is no `pyproject.toml` / `setup.py` / `requirements.txt`, so deps
live only inside `.venv/`.

## Limitations / deferred work

- **No `pyproject.toml`.** Dependency tracking is informal.
- **No real migration runner.** `init_schema()` is the only path; no
  tracker, no down-migrations, no `make migrate` integration.
- **No PII scrubbing.** `CLAUDE.md`'s `memory/scrub.py` contract is not
  wired in for any category yet. Cat 5 is the FIRST category to persist
  embeddings, so the gap is higher-risk here.
- **No history of feedback edits.** `update_user_feedback` overwrites
  the prior `user_feedback`; only the latest survives.
- **Structured fields are not embedded.** Semantic search cannot find
  episodes by `datasets_used` / `metrics_used` content alone; use
  `search_analyses` for substring.
- **No structured filter API.** `find_similar` accepts only
  `project_id`, `query`, `k`. No `analysis_type` filter, no time-range
  filter.
- **`user_feedback` author not retained.** The `user_id` passed to
  `update_feedback` is validated and discarded.

## Related

- Cat 1: `mads/memory/README.md` (`user_preference`).
- Cat 2: `mads/memory/DATASET_SCHEMA_README.md` (`schema_fact`).
- Cat 3: `mads/memory/METRIC_DEFINITION_README.md` (`metric_definition`).
- Cat 4: `mads/memory/IMPLEMENTATION_NOTES_DATA_QUALITY.md` (`data_quality_issue`).
