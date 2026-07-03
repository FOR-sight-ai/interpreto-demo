# Follow-up work

Everything below is out of scope for the interpreto 0.4.15 → 0.5.0
migration. It is queued for a later pass and does **not** block deploying
the gallery.

## 1. Metrics

Interpreto 0.5.0 ships several metric modules that the gallery does not
yet expose:

- `interpreto.attributions.metrics.{Insertion, Deletion}` — evaluate
  attribution HTML files.
- `interpreto.concepts.metrics.{MSE, FID, Sparsity, SparsityRatio,
  Stability}` — evaluate concept explainers.
- `interpreto.concepts.metrics.consim.ConSim` — end-to-end
  concept-based explanation quality.

Suggested wiring:

1. Add a `scripts/*_metrics.py` companion per explanation family. Each
   loads the same activations + fitted concept explainer from `data/`
   (see `scripts/_common.load_concept_model`) and computes metrics.
2. Extend `manifest.json` with a `metrics` field per entry (or a
   sibling `metrics.json`) so the UI can render metric values.
3. Add a small metrics panel in `app.js` (right next to the method
   dropdown) to show the score of the currently displayed method.

## 2. Inputs-to-concepts attributions (classification only)

Release note ([#150](https://github.com/FOR-sight-ai/interpreto/pull/150))
introduces `concept_explainer.get_inputs_to_concepts_model()`, letting
an attribution method attribute *concept* scores to *input* tokens.

Suggested scope:

- New script `scripts/classification_inputs_to_concepts.py`.
- Method: `Lime` (as answered in the plan Q&A).
- Output path: `explanations/<clf-model>/concept/local/sample-XXX/lime.html`
  (introduces a new `local` concept scope for classification — the UI
  build_manifest is generic and will pick it up automatically).
- Snippet: use `Lime(concept_explainer.get_inputs_to_concepts_model(),
  splitter.tokenizer)` — see `references/classification_demonstration.ipynb`.

## 3. Probes (post-hoc supervised concept explanations)

Release note also introduces probes (see
`references/{classification,generation}_probe_tutorial.ipynb`).

Suggested scope:

- New script `scripts/classification_probes.py` for the fairness /
  linguistic-probe use case.
- Probe types to explore: `LinearRegressionProbe`,
  `LogisticRegressionProbe`, `CosineCentroidProbe`, `LinearSVMProbe`.
- Requires *labeled* concepts; design a small taxonomy of labels
  (e.g. text length, negation presence, tense) per the notebook.
- Rendering: probe explainers implement the same
  `activations_to_concepts` interface as unsupervised concept
  explainers, so the existing HTML flow works.

## 4. Bump generation-concept sample counts

The current per-model sample counts are conservative to keep training
time and disk usage manageable:

- `gen:gpt2` — 10 000 Wikipedia articles.
- `gen:qwen3-0.6b` — 2 000 articles.
- `gen:llama3.1-8b` — 500 articles.

If disk & time budgets allow later, edit
`MODEL_CONFIGS[...][num_samples]` in `scripts/generation_concepts.py`
and re-run. Recommended targets:

- `gen:qwen3-0.6b` — 5 000 to 10 000 articles (~25–50 GB float16).
- `gen:llama3.1-8b` — 2 000 to 5 000 articles (~50–120 GB float16).

The snippets automatically reflect the updated dataset slice via the
`num_samples` render parameter.

## 5. Report the interpreto 0.5.0 dtype bug

Attribution's `InferenceWrapper` moves `inputs_embeds` to the device
(`inference_wrapper.py:397`) but never casts them to the model dtype.
As a result, gradient-based attribution methods (`GradientShap`,
`IntegratedGradients`, `Saliency`, `SmoothGrad`, `SquareGrad`,
`VarGrad`) fail on non-float32 models.

Impact for this repo: `gen:llama3.1-8b` only ships perturbation
methods (`KernelShap`, `Lime`, `Occlusion`, `Sobol`). See
`generation_attributions.py::MODEL_CONFIGS["gen:llama3.1-8b"]["skip_methods"]`.

Suggested action: open an issue on
[FOR-sight-ai/interpreto](https://github.com/FOR-sight-ai/interpreto)
proposing to add `padded_inputs[key] = padded_inputs[key].to(self.model.dtype)`
where relevant (or gate on the wrapper's model dtype).
