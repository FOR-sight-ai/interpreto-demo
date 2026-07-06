#!/usr/bin/env python3
"""Generate classification concept HTML files and minimal .py snippets.

Interpreto 0.5.0 API. For each classification model, this script:

1. Loads a ``SplitterForClassification`` (auto-detects the head, forces
   the [CLS] granularity).
2. Loads the dataset (capped at ``NUM_SAMPLES``) and computes the [CLS]
   activations once. The tuple ``(activations, predictions)`` is cached
   under ``data/<model_id>/activations.pt``.
3. For every concept method, trains (or loads) the concept explainer,
   interprets it with ``TopKInputs``, ranks concepts with
   ``concept_output_gradient`` and writes ``<method>.html`` +
   a matching minimal ``.py`` snippet.

Set ``DEBUG_SAMPLES`` in the environment (``DEBUG_SAMPLES=1000``) to
override ``NUM_SAMPLES`` for quick iteration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpreto import SplitterForClassification, plot_concepts
from interpreto.concepts import (
    ICAConcepts,
    LLMLabels,
    MpSAEConcepts,
    NeuronsAsConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SVDConcepts,
    VanillaSAEConcepts,
)
from interpreto.concepts.interpretations import TopKInputs
from interpreto.concepts.methods.overcomplete import (
    DeadNeuronsReanimationLoss,
    MSELoss,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    HuggingFaceLLM,
    activations_cache_path,
    cache_activations,
    explainers_cache_dir,
    format_kwargs_lines,
    load_concept_model,
    save_concept_model,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Configuration (edit these)
# ----------------------------
model_id = "clf:emotion:bert"

MODEL_CONFIGS = {
    "clf:emotion:bert": {
        "hf_model_id": "nateraw/bert-base-uncased-emotion",
        "hf_dataset_id": "dair-ai/emotion",
        "hf_dataset_config": "split",
        "classes_names": [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise",
        ],
        "forward_kwargs": None,
    },
    "clf:imdb:distilbert": {
        "hf_model_id": "lvwerra/distilbert-imdb",
        "hf_dataset_id": "stanfordnlp/imdb",
        "hf_dataset_config": None,
        "classes_names": [
            "negative",
            "positive",
        ],
        # IMDB reviews are long; DistilBERT accepts up to 512 tokens.
        "forward_kwargs": {"truncation": True},
    },
    "clf:ag-news:roberta": {
        "hf_model_id": "arman1o1/roberta_ag_news_model",
        "hf_dataset_id": "fancyzhx/ag_news",
        "hf_dataset_config": None,
        "classes_names": [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ],
        "forward_kwargs": None,
    },
}

DATASET_SPLIT = "train"
NUM_SAMPLES = 200_000  # cap; will be truncated if the dataset is smaller
SEED = 0

NB_CONCEPTS = 30
TOP_K = 10
TOPK_WORDS = 5

# ----------------------------
# LLM Labels configuration
# ----------------------------
# The labeler is a small local causal LM used by ``LLMLabels`` to name
# concepts. It is loaded once per script run and shared across all
# methods. Set ``SKIP_LLM_LABELS=1`` to bypass the labeler entirely and
# only emit the TopK HTMLs (matches the pre-LLM-Labels behavior). Set
# ``ONLY_LLM_LABELS=1`` to skip the TopK HTMLs and only (re)write the
# ``_llm_labels`` variants; useful when the TopK HTMLs are already on
# disk and you only want to iterate on the labeler.
#
# The primary labeler is ``Qwen/Qwen3.5-2B`` (per the requested demo
# spec). Its architecture (``qwen3_5``) is only supported by very
# recent releases of ``transformers``; on older installs the load
# raises a ``ValueError`` about an unknown ``model_type``. When that
# happens we fall back to ``LABELER_FALLBACK_HF_ID`` (a Qwen3-family
# model that ships in mainline ``transformers``).
#
# ``LLM_LABELS_K_CONTEXT`` is intentionally omitted: for classification
# the concept explainer sits on top of the [CLS] token, so there is no
# surrounding-token context to feed the LLM (``LLMLabels`` silently
# clamps ``k_context`` to 0 with a warning in that case).
LABELER_HF_ID = "Qwen/Qwen3.5-2B"
LABELER_FALLBACK_HF_ID = "Qwen/Qwen3-1.7B"
LABELER_TORCH_DTYPE = torch.bfloat16
LLM_LABELS_K_EXAMPLES = 20

BATCH_SIZE = 64
GRADIENT_BATCH_SIZE = 64

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "explanations"

METHODS = {
    "ica": ICAConcepts,
    "mp_sae": MpSAEConcepts,
    "neurons_as_concepts": NeuronsAsConcepts,
    "pca": PCAConcepts,
    "semi_nmf": SemiNMFConcepts,
    "svd": SVDConcepts,
    "vanilla_sae": VanillaSAEConcepts,
}

DEFAULT_INIT_PARAMETERS = {"nb_concepts": NB_CONCEPTS, "device": device}
INIT_PARAMETERS: dict[str, dict] = {k: DEFAULT_INIT_PARAMETERS.copy() for k in METHODS}
INIT_PARAMETERS["neurons_as_concepts"] = {}

SAES_FIT_PARAMETERS = {
    "criterion": DeadNeuronsReanimationLoss,
    "optimizer_class": torch.optim.Adam,
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {"T_max": 20, "eta_min": 1e-6},
    "lr": 1e-3,
    "nb_epochs": 30,
    "batch_size": 32 * BATCH_SIZE,
    "monitoring": 0,
}
FIT_PARAMETERS: dict[str, dict] = {
    k: SAES_FIT_PARAMETERS.copy() for k in METHODS if "sae" in k
}
FIT_PARAMETERS["ica"] = {"max_iter": 5000}
FIT_PARAMETERS["mp_sae"]["criterion"] = MSELoss


# ----------------------------------------------------------------------
# Snippet rendering
# ----------------------------------------------------------------------


def render_code_snippet(
    method_name: str,
    explainer_cls: type,
    hf_model_id: str,
    hf_dataset_id: str,
    hf_dataset_config: str | None,
    classes_names: list[str],
    init_params: dict,
    fit_params: dict,
    count_min_threshold: int,
    forward_kwargs: dict | None,
) -> str:
    """Return a self-contained snippet reproducing one HTML file."""
    init_lines, init_imports = format_kwargs_lines(init_params, indent="    ")
    fit_lines, fit_imports = format_kwargs_lines(fit_params, indent="    ")
    extra_imports = sorted(init_imports | fit_imports)

    lines: list[str] = [
        "import torch",
        "from datasets import load_dataset",
        "from interpreto import SplitterForClassification, plot_concepts",
        f"from interpreto.concepts import {explainer_cls.__name__}",
        "from interpreto.concepts.interpretations import TopKInputs",
    ]
    if extra_imports:
        lines.append(
            "from interpreto.concepts.methods.overcomplete import "
            + ", ".join(extra_imports)
        )
    lines.append("")
    lines.append('device = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("")
    lines.append(
        f"splitter = SplitterForClassification({hf_model_id!r}, "
        f"batch_size={BATCH_SIZE}, device_map=device)"
    )

    if hf_dataset_config is None:
        load_call = f"load_dataset({hf_dataset_id!r})"
    else:
        load_call = f"load_dataset({hf_dataset_id!r}, {hf_dataset_config!r})"
    lines.append(f'inputs = {load_call}[{DATASET_SPLIT!r}]["text"]')
    lines.append(f"classes_names = {classes_names!r}")
    lines.append("")

    if forward_kwargs:
        lines.append(
            f"activations, _ = splitter.get_activations(inputs, forward_kwargs={forward_kwargs!r})"
        )
    else:
        lines.append("activations, _ = splitter.get_activations(inputs)")
    lines.append("")

    lines.append(f"concept_explainer = {explainer_cls.__name__}(")
    lines.append("    splitter,")
    lines.extend(init_lines)
    lines.append(")")

    if explainer_cls is not NeuronsAsConcepts:
        lines.append("")
        if fit_lines:
            lines.append("concept_explainer.fit(")
            lines.append("    activations,")
            lines.extend(fit_lines)
            lines.append(")")
        else:
            lines.append("concept_explainer.fit(activations)")

    lines.append("")
    lines.append("topk = TopKInputs(")
    lines.append("    concept_explainer=concept_explainer,")
    lines.append(f"    k={TOPK_WORDS},")
    lines.append("    use_unique_words=3,")
    lines.append(
        f'    unique_words_kwargs={{"count_min_threshold": {count_min_threshold}, "lemmatize": True}},'
    )
    lines.append(")")
    lines.append(
        'labels = {k: list(v.keys()) for k, v in topk.interpret(inputs=inputs, concepts_indices="all").items()}'
    )
    lines.append("")
    lines.append("gradients = concept_explainer.concept_output_gradient(")
    lines.append("    inputs=activations,")
    lines.append("    targets=None,")
    lines.append(f"    batch_size={GRADIENT_BATCH_SIZE},")
    lines.append(")")
    lines.append("mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)")
    lines.append("")
    lines.append("plot_concepts(")
    lines.append("    classes_names=classes_names,")
    lines.append("    concepts_importances=mean_gradients,")
    lines.append("    concepts_labels=labels,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def render_llm_labels_snippet(
    method_name: str,
    explainer_cls: type,
    hf_model_id: str,
    hf_dataset_id: str,
    hf_dataset_config: str | None,
    classes_names: list[str],
    init_params: dict,
    fit_params: dict,
    forward_kwargs: dict | None,
) -> str:
    """Return a self-contained snippet reproducing one ``_llm_labels`` HTML."""
    init_lines, init_imports = format_kwargs_lines(init_params, indent="    ")
    fit_lines, fit_imports = format_kwargs_lines(fit_params, indent="    ")
    extra_imports = sorted(init_imports | fit_imports)

    lines: list[str] = [
        "import torch",
        "from datasets import load_dataset",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from interpreto import SplitterForClassification, plot_concepts",
        f"from interpreto.concepts import {explainer_cls.__name__}, LLMLabels",
        "",
        "# ``HuggingFaceLLM`` is not (yet) shipped by interpreto — this import will",
        "# start working once the class lands upstream. In the meantime, see the",
        '# "Using your own LLM interface" section of the classification concept',
        "# tutorial for the reference implementation you can paste in here.",
        "from interpreto.commons import HuggingFaceLLM",
    ]
    if extra_imports:
        lines.append(
            "from interpreto.concepts.methods.overcomplete import "
            + ", ".join(extra_imports)
        )
    lines.append("")
    lines.append('device = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("")
    lines.append(
        f"splitter = SplitterForClassification({hf_model_id!r}, "
        f"batch_size={BATCH_SIZE}, device_map=device)"
    )

    if hf_dataset_config is None:
        load_call = f"load_dataset({hf_dataset_id!r})"
    else:
        load_call = f"load_dataset({hf_dataset_id!r}, {hf_dataset_config!r})"
    lines.append(f'inputs = {load_call}[{DATASET_SPLIT!r}]["text"]')
    lines.append(f"classes_names = {classes_names!r}")
    lines.append("")

    if forward_kwargs:
        lines.append(
            f"activations, _ = splitter.get_activations(inputs, forward_kwargs={forward_kwargs!r})"
        )
    else:
        lines.append("activations, _ = splitter.get_activations(inputs)")
    lines.append("")

    lines.append(f"concept_explainer = {explainer_cls.__name__}(")
    lines.append("    splitter,")
    lines.extend(init_lines)
    lines.append(")")

    if explainer_cls is not NeuronsAsConcepts:
        lines.append("")
        if fit_lines:
            lines.append("concept_explainer.fit(")
            lines.append("    activations,")
            lines.extend(fit_lines)
            lines.append(")")
        else:
            lines.append("concept_explainer.fit(activations)")

    lines.append("")
    lines.append("# Load a small local causal LM as the labeler.")
    lines.append(f"labeler_tokenizer = AutoTokenizer.from_pretrained({LABELER_HF_ID!r})")
    lines.append(
        f"labeler_model = AutoModelForCausalLM.from_pretrained("
        f"{LABELER_HF_ID!r}, torch_dtype=torch.bfloat16).to(device)"
    )
    lines.append("llm_interface = HuggingFaceLLM(labeler_model, labeler_tokenizer)")
    lines.append("")
    lines.append("llm_labels = LLMLabels(")
    lines.append("    concept_explainer=concept_explainer,")
    lines.append("    llm_interface=llm_interface,")
    lines.append(f"    k_examples={LLM_LABELS_K_EXAMPLES},")
    lines.append(")")
    lines.append(
        'labels = {k: v for k, v in llm_labels.interpret('
        'inputs=inputs, latent_activations=activations, concepts_indices="all"'
        ').items() if v}'
    )
    lines.append("")
    lines.append("gradients = concept_explainer.concept_output_gradient(")
    lines.append("    inputs=activations,")
    lines.append("    targets=None,")
    lines.append(f"    batch_size={GRADIENT_BATCH_SIZE},")
    lines.append(")")
    lines.append("mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)")
    lines.append("")
    lines.append("plot_concepts(")
    lines.append("    classes_names=classes_names,")
    lines.append("    concepts_importances=mean_gradients,")
    lines.append("    concepts_labels=labels,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def load_inputs(config: dict) -> list[str]:
    num_samples = int(os.environ.get("DEBUG_SAMPLES", NUM_SAMPLES))
    dataset_kwargs = (
        {"name": config["hf_dataset_config"]}
        if config["hf_dataset_config"] is not None
        else {}
    )
    dataset = load_dataset(config["hf_dataset_id"], **dataset_kwargs)[DATASET_SPLIT]
    dataset = dataset.shuffle(seed=SEED)
    n = min(num_samples, len(dataset))
    print(f"Using {n} inputs (out of {len(dataset)} in {DATASET_SPLIT} split)")
    return list(dataset.select(range(n))["text"])


def _try_load_labeler(hf_id: str):
    """Attempt to load ``(tokenizer, model)`` for ``hf_id``.

    Returns ``(tokenizer, model)`` on success. Returns ``(None, None)``
    if the local ``transformers`` install doesn't know the checkpoint's
    architecture (which is the usual failure mode for very-recent
    Qwen3.5 releases).
    """
    print(f"\nLoading labeler {hf_id} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=LABELER_TORCH_DTYPE,
        ).to(device)
        return tokenizer, model
    except ValueError as exc:
        if "model type" in str(exc) or "model_type" in str(exc):
            print(f"   ! {hf_id} is not supported by the installed transformers ({exc})")
            return None, None
        raise


def _load_labeler():
    """Load the labeler, falling back to :data:`LABELER_FALLBACK_HF_ID`
    on unknown-architecture errors. Returns ``(hf_id, tokenizer, model)``
    or ``(None, None, None)`` if both attempts fail."""
    tokenizer, model = _try_load_labeler(LABELER_HF_ID)
    if model is not None:
        return LABELER_HF_ID, tokenizer, model

    if LABELER_FALLBACK_HF_ID and LABELER_FALLBACK_HF_ID != LABELER_HF_ID:
        print(f"   falling back to {LABELER_FALLBACK_HF_ID}")
        tokenizer, model = _try_load_labeler(LABELER_FALLBACK_HF_ID)
        if model is not None:
            return LABELER_FALLBACK_HF_ID, tokenizer, model

    print("   ! labeler load failed; skipping LLM-Labels branch for this run.")
    return None, None, None


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    classes_names = config["classes_names"]

    torch.manual_seed(SEED)

    skip_llm_labels = os.environ.get("SKIP_LLM_LABELS") == "1"
    only_llm_labels = os.environ.get("ONLY_LLM_LABELS") == "1"
    if skip_llm_labels and only_llm_labels:
        raise SystemExit(
            "Both SKIP_LLM_LABELS=1 and ONLY_LLM_LABELS=1 were set; "
            "these are mutually exclusive."
        )

    output_root = OUTPUT_ROOT / model_id / "concept" / "general"
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Splitter + inputs + cached activations
    # ------------------------------------------------------------------
    splitter = SplitterForClassification(
        config["hf_model_id"],
        batch_size=BATCH_SIZE,
        device_map=device,
    )
    inputs = load_inputs(config)
    count_min_threshold = max(1, round(len(inputs) * 0.002))

    activations, _predictions = cache_activations(
        splitter,
        inputs,
        activations_cache_path(model_id),
        tqdm_bar=True,
        forward_kwargs=config["forward_kwargs"] or {},
    )

    # ------------------------------------------------------------------
    # 1b. Optional labeler for the LLM-Labels variant (loaded once).
    # ------------------------------------------------------------------
    my_llm = None
    if skip_llm_labels:
        print("\nSKIP_LLM_LABELS=1: skipping labeler load, TopK HTMLs only.")
    else:
        labeler_id, labeler_tokenizer, labeler_model = _load_labeler()
        if labeler_model is not None:
            print(f"   labeler ready ({labeler_id}, device={labeler_model.device})")
            my_llm = HuggingFaceLLM(labeler_model, labeler_tokenizer)

    # ------------------------------------------------------------------
    # 2. Iterate over concept methods
    # ------------------------------------------------------------------
    for method_name, explainer_cls in METHODS.items():
        html_path = output_root / f"{method_name}.html"
        code_path = html_path.with_suffix(".py")
        html_path_llm = output_root / f"{method_name}_llm_labels.html"
        code_path_llm = html_path_llm.with_suffix(".py")
        init_params = INIT_PARAMETERS.get(method_name, {})
        fit_params = FIT_PARAMETERS.get(method_name, {})

        print(f"\n== {method_name}")
        concept_explainer = explainer_cls(splitter, **init_params)

        explainer_path = explainers_cache_dir(model_id) / f"{method_name}.pt"
        if explainer_cls is NeuronsAsConcepts:
            pass  # nothing to fit
        elif explainer_path.exists():
            print(f"   loading fitted explainer from {explainer_path}")
            load_concept_model(concept_explainer, explainer_path, device)
        else:
            print(f"   fitting on {activations.shape[0]} activations")
            concept_explainer.fit(activations, **fit_params)
            try:
                save_concept_model(concept_explainer, explainer_path)
                print(f"   saved to {explainer_path}")
            except NotImplementedError as exc:
                print(f"   (skipped save: {exc})")

        # ---- importance (shared by TopK and LLM-Labels) --------------
        gradients = concept_explainer.concept_output_gradient(
            inputs=activations,
            targets=None,
            batch_size=GRADIENT_BATCH_SIZE,
        )
        mean_gradients = torch.stack(gradients).abs().squeeze().mean(0)

        # ---- TopK interpretation + HTML ------------------------------
        if not only_llm_labels:
            topk = TopKInputs(
                concept_explainer=concept_explainer,
                k=TOPK_WORDS,
                use_unique_words=3,
                unique_words_kwargs={
                    "count_min_threshold": count_min_threshold,
                    "lemmatize": True,
                },
            )
            topk_words = topk.interpret(inputs=inputs, concepts_indices="all")
            topk_labels = {k: list(v.keys()) for k, v in topk_words.items() if v}

            print(f"   saving {html_path}")
            plot_concepts(
                classes_names=classes_names,
                concepts_importances=mean_gradients,
                concepts_labels=topk_labels,
                top_k=TOP_K,
                save_path=str(html_path),
            )

            code_path.write_text(
                render_code_snippet(
                    method_name=method_name,
                    explainer_cls=explainer_cls,
                    hf_model_id=config["hf_model_id"],
                    hf_dataset_id=config["hf_dataset_id"],
                    hf_dataset_config=config["hf_dataset_config"],
                    classes_names=classes_names,
                    init_params=init_params,
                    fit_params=fit_params,
                    count_min_threshold=count_min_threshold,
                    forward_kwargs=config["forward_kwargs"],
                ),
                encoding="utf-8",
            )
        else:
            print(f"   ONLY_LLM_LABELS=1: skipping {html_path.name}")

        # ---- LLM-Labels interpretation + HTML ------------------------
        if my_llm is not None:
            llm_labels_method = LLMLabels(
                concept_explainer=concept_explainer,
                llm_interface=my_llm,
                k_examples=LLM_LABELS_K_EXAMPLES,
            )
            # Pass the pre-computed activations so LLMLabels doesn't
            # re-tokenize/re-run the base model. Without this, inputs
            # longer than the base model's context window (e.g. IMDB
            # reviews on DistilBERT's 512-token limit) crash even when
            # ``forward_kwargs={"truncation": True}`` was used at
            # activation-caching time — ``interpret`` doesn't take
            # ``forward_kwargs``.
            llm_interpretations = llm_labels_method.interpret(
                inputs=inputs,
                latent_activations=activations,
                concepts_indices="all",
            )
            llm_labels = {
                k: v for k, v in llm_interpretations.items() if v
            }

            print(f"   saving {html_path_llm}")
            plot_concepts(
                classes_names=classes_names,
                concepts_importances=mean_gradients,
                concepts_labels=llm_labels,
                top_k=TOP_K,
                save_path=str(html_path_llm),
            )

            code_path_llm.write_text(
                render_llm_labels_snippet(
                    method_name=method_name,
                    explainer_cls=explainer_cls,
                    hf_model_id=config["hf_model_id"],
                    hf_dataset_id=config["hf_dataset_id"],
                    hf_dataset_config=config["hf_dataset_config"],
                    classes_names=classes_names,
                    init_params=init_params,
                    fit_params=fit_params,
                    forward_kwargs=config["forward_kwargs"],
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
