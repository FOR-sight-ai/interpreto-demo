# Migration plan: interpreto-demo from 0.4.15 to 0.5.0

Living document tracking the migration. Each step is a checkpoint that
must be reviewed by the user before moving on.

## Status legend

- `[ ]` not started
- `[~]` in progress / partial
- `[x]` done and confirmed by user
- `[!]` blocked or needs discussion

---

## Context recap

### API changes (from `references/release-note.md`)

- `from interpreto import ModelWithSplitPoints` (not `interpreto.model_wrapping`).
- `split_point=` (singular). `ModelWithSplitPoints` remains available for exotic setups.
- New splitters: `SplitterForClassification` (auto-detects head, forces CLS granularity) and `SplitterForGeneration` (`split_point` required, forces TOKEN granularity).
- `splitter.get_activations(inputs)` now returns a **tuple** `(activations, predictions)` – no more dict + `get_split_activations`.
- `encode_activations`/`decode_concepts` renamed to `activations_to_concepts`/`concepts_to_activations`.
- Interpretation methods no longer take `concept_model_device`; use `concept_explainer.to("cuda")`.
- Attributions do **not** accept `BatchEncodings` of multiple inputs anymore.
- New: `concept_explainer.get_inputs_to_concepts_model()` for inputs-to-concepts attributions (skipped for now, queued alongside metrics).
- New: probes (skipped for now, queued alongside metrics).

### User decisions

1. Remove all `class-wise` references from UI/build_manifest/README.
2. Cap datasets at **200k samples**; use whole train split otherwise. Debug on 1k first.
3. Classification-attribution: keep both `single-class` + `all-classes`, 10 samples (unchanged).
4. Generation concepts: only SAE variants (`vanilla_sae`, `mp_sae`, `batch_top_k_sae`) + `neurons_as_concepts`. No PCA/SVD/ICA/SemiNMF for generation.
5. Attempt Llama-3.1-8B concept extraction – tune batch size.
6. Cache activations **and** fitted concept explainer using `references/save_load_concepts.py` (BatchTopKSAE debug can be attempted opportunistically).
7. Inputs-to-concepts attributions and probes: **not** in this plan (queued alongside metrics).

---

## Ground rules for the implementing LLM

- Work in **small, verified steps.** After every step: stop, `git status`/`git diff`, and hand back to the user for review.
- **Never** touch multiple scripts in the same commit unless the user asks.
- **Every script must be end-to-end runnable on 1k samples first (`DEBUG_SAMPLES=1000` env var) before scaling up.** Do not proceed to the full dataset until the user confirms.
- The rendered `.py` snippet must be **minimal and self-contained**: no `data/` cache, no `save_concept_model`, no error handling. It matches the notebook style. The actual script may cache activations, deduplicate work between methods, but the snippet strips all of that.
- **Verify snippet reproduces the HTML.** After generating a snippet, spot-check it by running the snippet standalone (or diff against the demo notebook) on at least one sample. Do not skip this check – the user was burned by mismatches before.
- Snippet imports must match exactly what is used inside the snippet – no dead imports.
- No `class-wise` code path anywhere: delete it, do not comment it out.
- Prefer `SplitterForClassification` and `SplitterForGeneration` over `ModelWithSplitPoints`.
- Preserve deterministic seeding (`torch.manual_seed(SEED)`).

---

## Steps

### Step 0 – Save this plan
- [x] Written to `migration_plan.md`.

### Step 1 – Environment sanity
- [x] Upgraded `interpreto` from 0.4.15 to 0.5.0 in `.venv` (`uv pip install --upgrade --no-deps "interpreto>=0.5,<0.6"`).
- [x] Upgraded `nnsight` from 0.5.15 to 0.7.0 (required by interpreto 0.5.0).
- [x] Pinned `interpreto>=0.5,<0.6` and `nnsight>=0.7,<0.8` in `requirements.txt`.
- [x] Verified `SplitterForClassification` and `SplitterForGeneration` importable.
- [x] `interpreto.__version__` **not defined** — resolved via `importlib.metadata.version("interpreto")` and exposed in `manifest.json["interpreto_version"]`.
- [x] Added `data/` to `.gitignore`.
- [x] Added version badge (`#interpreto-version`) rendered by `app.js` from `manifest.interpreto_version`.
- [x] Rebuilt manifest — confirmed `"interpreto_version": "0.5.0"`.
- [ ] **Awaiting user review.**

### Step 2 – Introduce `scripts/_common.py`
- [x] Shared helpers: `save_concept_model`, `load_concept_model`, `cache_activations`, `format_value`, `format_kwargs_lines`, `dedupe`.
- [x] Copied and hardened `save_concept_model` / `load_concept_model` from `references/save_load_concepts.py`.
- [x] Covers `SemiNMFConcepts` (D tensor) and any concept model exposing `state_dict` (SAEs, PCA/SVD/ICA). `BatchTopKSAE` will hit the `state_dict` branch — may still be flaky (documented TODO in ground rules).
- [x] Import + smoke test passed (see Step 2.3).
- [ ] **Awaiting user review.**

### Step 3 – Rewrite `scripts/build_manifest.py` and UI to drop `class-wise`
- [x] `build_manifest.py` was already generic (scope name inferred from directory) — no code change needed once the folders are gone.
- [x] `app.js`, `index.html`, `styles.css` were already scope-agnostic — no code change needed.
- [x] Deleted `explanations/{clf:ag-news:roberta, clf:emotion:bert, clf:imdb:distilbert}/concept/class-wise/`.
- [x] Updated `README.md` hierarchy diagram: removed class-wise, fixed the stale generation-concept path (`concept/local/` not `concept/general/`), added `data/` under Project layout, added `scripts/_common.py` mention.
- [x] Regenerated `manifest.json`: 78 → 75 entries; no `class-wise` scope; interpreto_version still present.
- [x] Confirmed the only remaining `class-wise` string outside `migration_plan.md` is in `scripts/classification_concepts_classwise.py`, which is deleted in Step 7.
- [ ] **Awaiting user review.**

### Step 4 – Rewrite `scripts/classification_attributions.py`
- [x] Rewritten as a clean 0.5.0 script. `Granularity` is now an enum value in the config (not a string).
- [x] Fixed snippet `targets` argument: was `torch.arange(len(classes_names))` (1-D), now `torch.tensor([[0, 1, ...]])` matching the demo notebook.
- [x] Snippet omits `granularity=` when it equals the 0.5.0 default (`Granularity.WORD`); only imports `Granularity` when needed.
- [x] Single-class snippet has no `targets=` line (default is predicted class).
- [x] Debug run (emotion, 2 samples × 2 methods) succeeded; standalone snippet run confirmed HTML reproduction.
- [x] SENTENCE-granularity debug run (imdb occlusion) succeeded and produced the correct snippet.
- [x] Full runs completed for all three classification models: emotion (10×10×2), ag-news (10×10×2), imdb (10×10×2). Sobol worked on all three.
- [x] Manifest regenerated: 75 entries, all 10 attribution methods per classification sample.
- [x] Snippet diff against `classification_demonstration.ipynb` matches.
- [ ] **Awaiting user review.**

### Step 5 – Rewrite `scripts/generation_attributions.py`
- [ ] Remove `KernelShap` and `Sobol` (already commented out).
- [ ] Confirm `n_perturbations` / `n_token_perturbations` extras still apply.
- [ ] Verify snippet reproduces HTML on `gpt2` sample 0.
- [ ] Run on GPT-2, Qwen, Llama.
- [ ] **Stop for user review.**

### Step 6 – Rewrite `scripts/classification_concepts.py`
- [ ] Use `SplitterForClassification`, unpack `(activations, predictions)`.
- [ ] Cache activations to `data/<model>/activations.pt`; cache explainers to `data/<model>/explainers/<method>.pt`.
- [ ] `TopKInputs` without `activation_granularity=` (CLS forced by splitter).
- [ ] Snippet template matches `classification_demonstration.ipynb`.
- [ ] Debug flow: 1k samples, `ica` only, emotion → open HTML → run snippet.
- [ ] Then all methods on emotion at 1k → then scale to 200k → repeat for imdb, ag-news.
- [ ] **Stop for user review at every substep above.**

### Step 7 – Delete `scripts/classification_concepts_classwise.py`
- [ ] Delete script.
- [ ] Delete leftover `explanations/*/concept/class-wise/` directories.
- [ ] **Stop for user review.**

### Step 8 – Rewrite `scripts/generation_concepts.py`
- [ ] Method set: `NeuronsAsConcepts`, `VanillaSAEConcepts`, `MpSAEConcepts`, `BatchTopKSAEConcepts`.
- [ ] Use `SplitterForGeneration`, unpack `(activations, _)`.
- [ ] Cache activations + explainer.
- [ ] Snippet template matches `generation_concept_tutorial.ipynb`.
- [ ] Debug flow: GPT-2, 1k, `vanilla_sae`, 3 epochs. Then 4 methods on GPT-2 at 1k → 10k → Qwen 10k → Llama (tuned batch size).
- [ ] **Stop for user review at every substep above.**

### Step 9 – Activation storage sizes (float16)
- [ ] Convert cached activations to `float16` before `torch.save`; upcast on load if fit demands.
- [ ] Verify PCA/ICA still fit correctly.
- [ ] **Stop for user review.**

### Step 10 – Regenerate manifest, externalize, browser check
- [ ] `python scripts/build_manifest.py`.
- [ ] `python scripts/externalize_explanations.py`.
- [ ] Manual browser click-through of all combinations.
- [ ] **Stop for user review.**

### Step 11 – Diff snippets against notebooks
- [ ] Pick three snippets (one per family), diff line-by-line with the matching notebook cell.
- [ ] Fix any argument name / import path mismatch, regenerate.
- [ ] **Stop for user review.**

### Step 12 – Delete dead code, finalize
- [ ] Confirm no `from interpreto.model_wrapping` imports remain.
- [ ] Confirm no `model_with_split_points` / `get_split_activations` references remain.
- [ ] Confirm no `class-wise` string remains outside git history.
- [ ] Regenerate `manifest.json` once more.
- [ ] **Stop for user review.**

### Step 13 – Wrap up
- [ ] Update `README.md` hierarchy diagram (no `class-wise`).
- [ ] Document new `data/` cache directory.
- [ ] `python scripts/build_manifest.py; python -m http.server 8000` smoke test.
- [ ] Leave a `TODO_METRICS.md` with follow-up plan (metrics + inputs-to-concepts + probes).

---

## Risks / open questions to flag during implementation

1. **BatchTopKSAE serialization** – `references/save_load_concepts.py` explicitly does not cover it. If loading is broken, metrics scripts will always have to retrain. Ask before spending time debugging.
2. **Llama-3.1-8B concepts** – not previously runnable. If OOM at bs=1, propose dropping / quantization / `device_map="auto"` with offloading.
3. **`activation_granularity` on `TopKInputs` for classification** – notebook does not pass it; verify API in 0.5.0 does not require it.
4. **`forward_kwargs={"truncation": True}`** – some notebooks pass this for long IMDB reviews; add if OOM.
5. **Concept count for generation SAEs** – tune per model in `MODEL_CONFIGS`.
6. **`concept_output_gradient` batch size** – tune per model to fit VRAM.
