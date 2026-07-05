# Metrics integration plan

Follow-up work on top of the interpreto 0.4.15 → 0.5.0 migration. This
plan is the single source of truth for wiring per-(model, method)
metric scores into the gallery, sorting the method chooser by them, and
displaying a small badge next to each explanation card.

**Status:** the 10 numbered steps below were all completed. See the
"Execution notes" section at the bottom for what was actually run,
what deviated from the plan, and follow-up work.

The other, out-of-scope follow-ups (probes, inputs-to-concepts, sample
bumps, upstream dtype bug report) live in the "Later" section at the
bottom.

## Goal

For every `(model, method, type, scope)` displayed in the gallery,
compute one or more metric scalars, embed them in `manifest.json`, and
let the user pick a metric from a new dropdown to sort the visible
methods and display each score as a small pill next to the card title.

## Design decisions (locked with the user)

- **Metrics in scope**: Insertion, Deletion (attribution); MSE, FID,
  Sparsity, SparsityRatio (concept). Stability and ConSim are deferred.
- **Attribution granularity**: **one aggregate scalar per (model,
  method, scope) across 100 fresh samples**. Same score is shared by
  every sample entry for that (model, method, scope) in the UI.
- **Attribution targets**: metric-time attributions are always computed
  with ``targets=None``. For classification, this means the metric
  scores the explanation of the **predicted class only** (both
  ``all-classes`` and ``single-class`` scopes therefore share the same
  score — the metric is target-agnostic). For generation, the
  interpreto API rejects ``targets=None``; the natural analog is to
  derive the target from the source corpus (100 Wikipedia articles
  split into 32 prompt tokens + 16 continuation tokens).
- **Concept granularity**: one scalar per (model, method); score
  repeated across every entry that references that concept method.
- **Held-out set for concept metrics**: 90/10 split of the cached
  `data/<model>/activations.pt` — no refit, no fresh dataset load.
  Accepted even for `gen:llama3.1-8b` (500 → 50 eval samples).
- **Storage**: embedded in `manifest.json`. Every entry gets a new
  optional `metric_scores` field; a top-level `metrics_meta` describes
  each metric (label, direction, unit).
- **Scripts**: 4 new dedicated files, one per (task, type):
  `classification_attribution_metrics.py`,
  `generation_attribution_metrics.py`,
  `classification_concept_metrics.py`,
  `generation_concept_metrics.py`.
  Each dumps a JSON under `data/<model>/metrics/…` which
  `build_manifest.py` merges at manifest-build time.
- **UI**: new `Metric` dropdown + `Sort ↑/↓` toggle in the filter row;
  score pill next to each method checkbox and card title.

## Manifest schema after Step 3

Every entry keeps its existing 6 fields and gains:

```jsonc
{
  "model": "clf:emotion:bert",
  "task":  "classification",
  "type":  "attribution",
  "scope": "single-class",
  "sample": "sample-003",
  "methods": ["lime.html", "kernel_shap.html", ...],
  "metric_scores": {
    "lime.html":        {"insertion": 0.42, "deletion": 0.11},
    "kernel_shap.html": {"insertion": 0.38, "deletion": 0.13}
  }
}
```

For attribution, the same ``metric_scores`` block is copied into every
sample entry of the same (model, method, scope). The ``all-classes``
and ``single-class`` scopes share the same scores because the metric
runs with ``targets=None`` (predicted class only).

Top-level `metrics_meta` (written once by `build_manifest.py`):

```jsonc
"metrics_meta": {
  "insertion":      {"label": "Insertion",      "direction": "higher_better", "applies_to": "attribution"},
  "deletion":       {"label": "Deletion",       "direction": "lower_better",  "applies_to": "attribution"},
  "mse":            {"label": "MSE",            "direction": "lower_better",  "applies_to": "concept"},
  "fid":            {"label": "FID",            "direction": "lower_better",  "applies_to": "concept"},
  "sparsity":       {"label": "Sparsity",       "direction": "lower_better",  "applies_to": "concept"},
  "sparsity_ratio": {"label": "Sparsity ratio", "direction": "lower_better",  "applies_to": "concept"}
}
```

## Steps

### Step 1 – Shared metric utilities in `scripts/_common.py`

- Add `metrics_cache_path(model_id, type_name, scope=None, sample=None)`
  returning `data/<model_id>/metrics/<type>/<scope>/<sample>.json`
  (creating intermediate dirs on demand). For concept clf:
  `data/<model_id>/metrics/concept/general.json`.
- Add `save_metric_scores(cache_path, method → dict)` /
  `load_metric_scores(cache_path)` — simple `json.dump`/`json.load`
  wrappers.
- Add `split_activations(activations, ratio=0.1, seed=0)` returning
  `(train, eval)` deterministic split. Used by concept metric scripts
  even though we do not refit — the eval half is still what we score.
- Add a `METRIC_DIRECTIONS` constant mirroring the `metrics_meta`
  block above; imported by `build_manifest.py` and by each metric
  script.

Deliverable: `_common.py` diff. No functional change to existing
scripts.

**Stop for user review.**

### Step 2 – `scripts/classification_attribution_metrics.py`

- Copy control flow from `classification_attributions.py`:
  - Load model + tokenizer once.
  - Load **100** samples (parameter `NUM_EVAL_SAMPLES=100`, override
    via `DEBUG_METRIC_SAMPLES` env). Draw from the same dataset with a
    deterministic seed independent of the 10 shown samples.
- For each method in `METHODS`:
  - Build the explainer once (same kwargs as
    `classification_attributions.py`).
  - Compute `attributions = explainer(model_inputs=inputs)` (no
    ``targets`` argument — the metric always scores the explanation of
    the predicted class).
  - Instantiate `Insertion(model, tokenizer, n_perturbations=50,
    batch_size=4)` and `Deletion(model, tokenizer, n_perturbations=50,
    batch_size=4)`.
  - `ins_auc, _ = insertion.evaluate(attributions)` etc.
  - Store two scalars per method: insertion + deletion.
- Dump the same JSON body to both scope files via
  `save_metric_scores(...)`:
  - `data/<model>/metrics/attribution/all-classes.json`
  - `data/<model>/metrics/attribution/single-class.json`

  Shape: `{"<method>.html": {"insertion": ..., "deletion": ...}}`.
- No HTML/snippet regeneration — this script is metric-only.
- Print a small summary table at the end.
- Notes:
  - Model + explainer stay in memory across methods; only re-run the
    perturbation loop per metric.
  - Reuse the exact `granularity=` and `batch_size=` from
    `classification_attributions.py` so scores describe the same
    explanations.

**Stop for user review.**

### Step 3 – `scripts/generation_attribution_metrics.py`

- Same shape as Step 2 but for `AutoModelForCausalLM`.
- **Sample source**: for classification we can freely subsample; for
  generation the current 3 shown samples are fixed prose strings in
  `SAMPLES = [...]` in `generation_attributions.py`. To keep the "100
  samples" bar we need a per-model sample set:
  - `gen:gpt2`, `gen:qwen3-0.6b`: 100 short prompts drawn from
    `wikimedia/wikipedia` (same as `generation_concepts.py`), truncated
    to the model context.
  - `gen:llama3.1-8b`: 100 prompts, but only run the 4 perturbation
    methods (mirrors `MODEL_CONFIGS[...]["skip_methods"]`). Gradient
    methods are still skipped.
- Reuse `MODEL_CONFIGS` shape from `generation_attributions.py`
  (dtype, skip_methods). Add an `n_perturbations` override per model:
  llama uses 20 to keep runtime manageable, others 50.
- Metric compute cost on gen is substantially higher than on clf. Log
  wall-clock per method so we know if we need to bump `batch_size`.
- Dump to `data/<model>/metrics/attribution/general.json`.

**Stop for user review.**

### Step 4 – `scripts/classification_concept_metrics.py`

- Load `data/<model>/activations.pt` via `cache_activations(...,
  force=False)` (already cached from Step 6 of migration).
- Deterministic 90/10 split (Step 1 helper). Eval slice is what we
  score.
- Build `SplitterForClassification(...)` (needed by
  `load_concept_model`) but do not touch the underlying model — no
  forward pass required.
- For each method in the classification concept `METHODS`:
  - Instantiate the concept explainer with the same init kwargs as
    `classification_concepts.py`, call `load_concept_model(...)`.
  - Compute:
    - `mse = MSE(concept_explainer).compute(eval_activations)`
    - `fid = FID(concept_explainer).compute(eval_activations)`
    - `sparsity = Sparsity(concept_explainer).compute(eval_activations)`
    - `sparsity_ratio = SparsityRatio(concept_explainer).compute(eval_activations)`
  - Special-case `NeuronsAsConcepts` (no `state_dict` on disk) — its
    metrics are still computable because encode+decode are identity.
    Just instantiate directly, skip `load_concept_model`.
- Dump to `data/<model>/metrics/concept/general.json`.

**Stop for user review.**

### Step 5 – `scripts/generation_concept_metrics.py`

- Same as Step 4 but with `SplitterForGeneration` and the 4 gen
  concept `METHODS`.
- Activations are float16 on disk; loader upcasts to float32 (already
  handled by `cache_activations`).
- Concept metrics are all `activation → concept → activation` so no
  base LLM inference — fast on all 3 gen models.
- `BatchTopKSAEConcepts` is refit-only in the current pipeline (no
  disk state). Fitting it here would be wasteful; instead **run this
  metrics script inside a fresh session that also refits** OR store
  the fitted explainer in memory from `generation_concepts.py` (not
  yet done). Cheapest fix: refit on the 90% train slice inside this
  script (SAE fitting on 90% of ~10k activations is 2-3 min on GPU).
- The gen concept manifest scope is `local` (per-sample HTMLs), but
  the metric is per-method, not per-sample. Store the same 4 scalars
  under **every** `sample-XXX.json` file for that (model, method),
  OR store once in `general.json` and have `build_manifest.py`
  broadcast the same score to every sample entry. **Choose the
  broadcast approach** (single file, simpler diff).
- Dump to `data/<model>/metrics/concept/general.json` (yes, gen too —
  the file name reflects that the score is not sample-specific).

**Stop for user review.**

### Step 6 – Extend `scripts/build_manifest.py`

- New helper `load_metrics_for_entry(model_id, type_name, scope,
  sample) -> dict | None`:
  - Attribution: look at
    `data/<model>/metrics/<type>/<scope>.json`. Every sample under
    that scope shares the same file.
  - Concept: look at `data/<model>/metrics/concept/general.json`.
    Every sample (gen) or the single sample-less entry (clf) shares
    the same file.
  - Return `dict[method_name, dict[metric_name, float]]` or `None`.
- Extend `add_entry(...)` with a new kwarg `metrics=None`; when
  non-None write `entry["metric_scores"] = metrics`.
- In `scan_scope_dir()`, right after `methods = collect_methods(...)`,
  call the loader and pass the resulting dict.
- At the top of `build_manifest()`, populate `manifest["metrics_meta"]`
  from `_common.METRIC_DIRECTIONS`.
- Add a `manifest["metrics_summary"]` that lists which metrics are
  actually present in the manifest (used by the UI to populate the
  dropdown).

Sanity check: re-run `python scripts/build_manifest.py` — expect the
new field on every entry where a JSON is present, absent otherwise.

**Stop for user review.**

### Step 7 – Frontend: normalize + state

`app.js`:
- Extend `normalizeManifest()` to preserve
  `entry.metric_scores` (currently the `.map()` drops unknown fields)
  and `manifest.metrics_meta` / `manifest.metrics_summary` on
  `state.manifest`.
- Add `state.metric = null` and `state.sortDirection = null` (auto-set
  from `metrics_meta[metric].direction`).
- Extend `updateUrl()`, `saveStoredState()`, `getUrlState()`,
  `loadStoredState()`, `hydrateState()` to sync `metric` and
  `sortDirection`. Sticky across reload and shareable via URL.
- New helper `listMetricsForCurrentEntry(entry)` returning the
  intersection of `entry.metric_scores` keys across all methods and
  `manifest.metrics_meta` — populates the dropdown, disabled if empty.

Deliverable: JS diff. No visual change yet — dropdown still hidden.

**Stop for user review.**

### Step 8 – Frontend: dropdown + sort

`index.html`:
- Add two `.control` blocks right after `#sample-control` and before
  `.control--methods`:
  - `<select id="metric-select">` for metric name.
  - `<button id="metric-sort-toggle" class="button button--ghost">`
    displaying `↑` or `↓` (starts hidden; only shown when a metric is
    picked). Clicking flips `state.sortDirection`.

`app.js`:
- In `updateControls()`, call `populateSelect(metricSelect,
  listMetricsForCurrentEntry(entry), state.metric, "Sort by…")`;
  disabled when no metrics exist.
- Wrap the return of `listMethodsForSelection()` with
  `sortMethodsByMetric(methods, entry, state.metric, state.sortDirection)`:
  - If `state.metric` is null → return alphabetical (current
    behaviour).
  - Otherwise sort by
    `entry.metric_scores[method]?.[state.metric]` respecting
    direction; push unscored methods to the bottom.
- Reflect the sort in both `state.availableMethods` (checkboxes) and
  `state.methods` (visible cards).
- Wire the toggle: click → `state.sortDirection =
  state.sortDirection === "desc" ? "asc" : "desc"` → refresh UI.
- Default `state.sortDirection` when picking a new metric =
  metric's declared direction (`higher_better` → `desc`).

**Stop for user review.**

### Step 9 – Frontend: score pill on each card and checkbox

`app.js`:
- In `renderMethodsList()`, append a `<span class="method-metric">`
  next to each checkbox label showing the current metric's score for
  that method (formatted to 3 significant digits), or `—` if missing.
- In `buildCard()`, append a `<span class="metric-badge">` next to
  `.card-title` showing `<metric_label>: <value>`.
- Both spans update whenever `state.metric` changes without a full
  re-render of the cards.

`styles.css`:
- Clone `.version-badge` (line 121) as `.metric-badge`. Use
  `accent-2` (`--color-accent-2`) so it visually differs from the
  interpreto version pill.
- Add `.method-metric` — smaller, monospace, right-aligned inside
  `.method-item`.
- Ensure the new `.control` cells inherit from the existing
  `.control` grid — no `.controls` layout change needed
  (`auto-fit`).

**Stop for user review.**

### Step 10 – Smoke test + docs

- Run in order:
  ```bash
  python scripts/classification_attribution_metrics.py    # per clf model
  python scripts/generation_attribution_metrics.py        # per gen model
  python scripts/classification_concept_metrics.py        # per clf model
  python scripts/generation_concept_metrics.py            # per gen model
  python scripts/build_manifest.py
  python -m http.server 8000 --bind 127.0.0.1
  ```
- Verify manually:
  - `manifest.json` contains `metrics_meta` at top level and
    `metric_scores` on entries that got a JSON.
  - UI: Metric dropdown appears, populates with 2 metrics for
    attribution scopes and 4 for concept scopes; picking one reorders
    methods and shows pills.
  - Toggle flips direction.
  - URL/localStorage restore reproduces the picked metric.
- Update `README.md`:
  - Add a "Compute metric scores" section listing the 4 scripts.
  - Extend the file hierarchy section under `data/` with
    `data/<model>/metrics/…`.
  - Extend the "Regenerate explanations" section noting that metric
    scripts are separate and only need to be re-run when the
    explanations or methods change.

**Migration to metrics complete.**

## Execution notes

What actually ran when Step 10 executed the plan end to end.

### Scores delivered

Manifest coverage after `build_manifest.py`:

- `clf:ag-news:roberta`: 21/21 entries scored (attribution × 2 scopes, concept).
- `clf:emotion:bert`: 21/21 entries scored.
- `clf:imdb:distilbert`: 21/21 entries scored.
- `gen:gpt2`: 6/6 entries scored (attribution + concept).
- `gen:qwen3-0.6b`: 6/6 entries scored.
- `gen:llama3.1-8b`: 3/6 entries scored (concept only; see limitation below).

**Total: 78/81 entries have `metric_scores`.**

### Deviations from the plan

1. **Classification attribution sample filter.** IMDB's SENTENCE
   granularity fails on single-sentence reviews (interpreto raises
   `ValueError: The mask dimension ... must be greater than 1`).
   `filter_by_granularity()` in
   `scripts/classification_attribution_metrics.py` drops such inputs
   pre-run; 3 of the first 100 IMDB test samples were single-sentence.

2. **Generation attribution scaled down.** The plan targeted 100
   samples × 50 perturbations × (32 prompt + 16 target). Reality: gpt2
   alone took 91 minutes at those settings. Defaults are now 50 samples
   × 16 prompt + 8 target tokens; per-model perturbation counts are
   `{gpt2: 50, qwen: 20, llama: 10}`. Total gen attribution time
   dropped to ~15 min (gpt2) + ~68 min (qwen).

3. **Generation concept eval slice capped.** FID's Wasserstein-1D
   sort OOMed on the raw 10% eval slice for gpt2 (688k tokens ×
   768 dim). `EVAL_MAX_ROWS = 50_000` in
   `scripts/generation_concept_metrics.py` sub-samples the eval slice
   deterministically before scoring. All 3 gen concept runs then fit
   easily in <5 min each.

4. **UX: hide the metric dropdown when no metrics exist.**
   `updateMetricControl` in `app.js` toggles `#metric-control.hidden`
   based on `metricsSummary.length`, so repositories without metric
   JSONs get the pre-Step-8 look with no visual clutter.

### Known limitations

**L1. Llama attribution metrics broken upstream.**
`gen:llama3.1-8b` is listed in `SKIP_MODELS` inside
`scripts/generation_attribution_metrics.py`. On the first sample,
`interpreto.commons.granularity.get_association_matrix` raises
`IndexError: index 16 is out of bounds for dimension 0 with size 16`
— the BOS token that Llama's tokenizer prepends shifts granularity
indices off by one. Concept metrics work fine on Llama; only the
attribution scores are missing. Fix belongs upstream (`interpreto`
package). Filed as an addendum to L6 below.

**L2. Metric magnitudes vary by model, not by method.**
MSE and FID scales depend on the raw activation magnitudes, which
differ between classification (CLS activations, small) and generation
(token activations, much larger). Cross-model comparison is not
meaningful; **within-model** comparison (the actual UI sort target)
works as designed.

**L3. Attribution scores on the same activations as fit.**
Concept metrics use a 90/10 split of the cached activations. Because
the fitted explainers (except `BatchTopKSAEConcepts` which we refit
on the train slice) were originally trained on the full tensor, the
eval slice was seen during fit — scores are mildly optimistic. This
is the accepted trade-off from the plan.

## Later (not blocked by this plan)

The four remaining items from the original TODO — kept here for
reference so this file remains the single follow-up doc.

### L1. Stability (deferred)

Requires retraining every concept explainer ≥ 2× with different seeds.
Prohibitive for SAEs on Llama; cheap for PCA/ICA/NMF variants. If we
want it later, add a `stability_seeds` config per method in the metric
scripts and a `stability` entry in `metrics_meta`.

### L2. ConSim (deferred)

Requires OpenAI API key + budget + 5–10 seeds for meaningful scores.
Output is LLM-brittle. Do not wire until the demo has a real use case
for it.

### L3. Inputs-to-concepts classification attributions

Release note [#150](https://github.com/FOR-sight-ai/interpreto/pull/150)
introduces `concept_explainer.get_inputs_to_concepts_model()`, letting
Lime attribute *concept* scores to *input* tokens. New script
`scripts/classification_inputs_to_concepts.py` writing to
`explanations/<clf-model>/concept/local/sample-XXX/lime.html`. The
build_manifest is generic and will pick up the new `local` scope
automatically.

### L4. Probes (post-hoc supervised concept explanations)

New script `scripts/classification_probes.py`. Probe types:
`LinearRegressionProbe`, `LogisticRegressionProbe`,
`CosineCentroidProbe`, `LinearSVMProbe`. Requires labeled concepts.

### L5. Bump generation-concept sample counts

Once disk + time budgets allow: `gen:qwen3-0.6b` 5k–10k articles,
`gen:llama3.1-8b` 2k–5k articles. Edit
`MODEL_CONFIGS[...][num_samples]` in `scripts/generation_concepts.py`
and re-run.

### L6. Report interpreto 0.5.0 bugs discovered during metric integration

Two upstream bugs blocked full coverage; both should be filed on
`FOR-sight-ai/interpreto`.

**Bug A. Attribution dtype cast missing.**
`inference_wrapper.py:397` moves `inputs_embeds` to device but not to
model dtype → gradient methods fail on non-float32 models. Impact:
`gen:llama3.1-8b` only ships perturbation attribution HTMLs
(see `MODEL_CONFIGS["gen:llama3.1-8b"]["skip_methods"]`). Proposed fix:
`padded_inputs[key] = padded_inputs[key].to(self.model.dtype)`.

**Bug B. `get_association_matrix` off-by-one on BOS token.**
`commons/granularity.py:621` writes into `assoc_matrix[j, gran_indices]`
where `gran_indices` can equal `assoc_matrix.shape[0]` when the
tokenizer prepends a BOS (Llama family). Reproduces on every call to
`Insertion(model, tokenizer).evaluate(...)` for `gen:llama3.1-8b`.
Impact: `SKIP_MODELS = {"gen:llama3.1-8b"}` in
`scripts/generation_attribution_metrics.py`.
