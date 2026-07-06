#!/usr/bin/env python3
"""Generate generation concept HTML files and minimal .py snippets.

Interpreto 0.5.0 API. For each generation model, this script:

1. Loads a ``SplitterForGeneration`` at a configured split point.
2. Loads the IMDB training corpus (capped at ``NUM_SAMPLES``) and computes
   the per-token activations once, flattened as ``(n * l, d)``. The
   ``(activations, _)`` tuple is cached under ``data/<model_id>/activations.pt``.
3. For every concept method, trains (or loads) the concept explainer,
   interprets it with ``TopKInputs`` and, for each demo ``SAMPLES``,
   plots the local concepts view + writes a minimal ``.py`` snippet.

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

from interpreto import SplitterForGeneration, plot_concepts
from interpreto.concepts import (
    BatchTopKSAEConcepts,
    LLMLabels,
    MpSAEConcepts,
    NeuronsAsConcepts,
    TopKInputs,
    VanillaSAEConcepts,
)
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
model_id = "gen:gpt2"

MODEL_CONFIGS = {
    "gen:gpt2": {
        "hf_model_id": "gpt2",
        "split_point": 8,
        "batch_size": 8,
        "nb_concepts": 1000,
        "concept_model_batch_size": 512,
        "top_k_ratio": 10,
        "max_tokens": 1024,  # GPT-2 context window
        "torch_dtype": None,  # default (float32 for gpt2)
        "num_samples": 10_000,
    },
    "gen:qwen3-0.6b": {
        "hf_model_id": "Qwen/Qwen3-0.6B",
        "split_point": 5,
        "batch_size": 8,
        "nb_concepts": 1000,
        "concept_model_batch_size": 512,
        "top_k_ratio": 10,
        "max_tokens": 4096,
        "torch_dtype": None,
        "num_samples": 2_000,
    },
    "gen:llama3.1-8b": {
        "hf_model_id": "meta-llama/Llama-3.1-8B",
        "split_point": 10,
        "batch_size": 1,
        "nb_concepts": 2000,
        "concept_model_batch_size": 256,
        "top_k_ratio": 10,
        "max_tokens": 4096,
        "torch_dtype": torch.bfloat16,
        "num_samples": 500,
    },
}

DATASET_HF_ID = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
DATASET_SPLIT = "train"
SEED = 0

TOP_K = 10
TOPK_WORDS = 10
EPOCHS = 5

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
# For generation we keep ``k_context`` non-zero: the concept explainer
# sits on token-level activations, so surrounding tokens give the LLM
# real linguistic context.
LABELER_HF_ID = "Qwen/Qwen3.5-2B"
LABELER_FALLBACK_HF_ID = "Qwen/Qwen3-1.7B"
LABELER_TORCH_DTYPE = torch.bfloat16
LLM_LABELS_K_EXAMPLES = 20
LLM_LABELS_K_CONTEXT = 5

# Activations are saved on disk in float16 to halve the footprint and
# loaded back as float32 (Overcomplete SAE training expects float32).
ACTIVATIONS_SAVE_DTYPE = torch.float16
ACTIVATIONS_LOAD_DTYPE = torch.float32

# Fixed samples used to produce per-sample HTMLs
SAMPLES = [
    "Alice and Bob enter the bar, then Alice offers a drink to Bob.",
    "Interpreto ships interpretable concept visualizations for language models.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
]

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "explanations"

METHODS = {
    "neurons_as_concepts": NeuronsAsConcepts,
    "vanilla_sae": VanillaSAEConcepts,
    "mp_sae": MpSAEConcepts,
    "batch_top_k_sae": BatchTopKSAEConcepts,
}

SAES_FIT_PARAMETERS_TEMPLATE = {
    "criterion": DeadNeuronsReanimationLoss,
    "optimizer_class": torch.optim.Adam,
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {"T_max": EPOCHS, "eta_min": 1e-6},
    "lr": 1e-3,
    "nb_epochs": EPOCHS,
    "monitoring": 0,
}


def build_init_parameters(config: dict) -> dict[str, dict]:
    """Init kwargs per method, given the model config."""
    common = {"nb_concepts": config["nb_concepts"], "device": device}
    return {
        "neurons_as_concepts": {},
        "vanilla_sae": common.copy(),
        "mp_sae": common.copy(),
        "batch_top_k_sae": {
            **common,
            "top_k": config["top_k_ratio"] * config["concept_model_batch_size"],
        },
    }


def build_fit_parameters(config: dict) -> dict[str, dict]:
    """Fit kwargs per method, given the model config."""
    sae_kwargs = {**SAES_FIT_PARAMETERS_TEMPLATE, "batch_size": config["concept_model_batch_size"]}
    return {
        "vanilla_sae": sae_kwargs.copy(),
        "mp_sae": {**sae_kwargs, "criterion": MSELoss},
        "batch_top_k_sae": sae_kwargs.copy(),
    }


# ----------------------------------------------------------------------
# Snippet rendering
# ----------------------------------------------------------------------


def render_code_snippet(
    method_name: str,
    explainer_cls: type,
    hf_model_id: str,
    split_point: int,
    batch_size: int,
    concept_model_batch_size: int,
    max_tokens: int,
    torch_dtype: torch.dtype | None,
    num_samples: int,
    sample_text: str,
    init_params: dict,
    fit_params: dict,
) -> str:
    """Return a self-contained snippet reproducing one HTML file."""
    init_lines, init_imports = format_kwargs_lines(init_params, indent="    ")
    fit_lines, fit_imports = format_kwargs_lines(fit_params, indent="    ")
    extra_imports = sorted(init_imports | fit_imports)

    lines: list[str] = [
        "import torch",
        "from datasets import load_dataset",
        "from interpreto import SplitterForGeneration, plot_concepts",
        f"from interpreto.concepts import {explainer_cls.__name__}, TopKInputs",
    ]
    if extra_imports:
        lines.append(
            "from interpreto.concepts.methods.overcomplete import "
            + ", ".join(extra_imports)
        )
    lines.append("")
    lines.append('device = "cuda" if torch.cuda.is_available() else "cpu"')
    lines.append("")

    splitter_args = (
        f"{hf_model_id!r}, split_point={split_point}, "
        f"batch_size={batch_size}, device_map=device"
    )
    if torch_dtype is not None:
        splitter_args += f", torch_dtype={torch_dtype}"

    lines.append(f"splitter = SplitterForGeneration({splitter_args})")
    lines.append(
        f'raw_inputs = load_dataset({DATASET_HF_ID!r}, {DATASET_CONFIG!r})[{DATASET_SPLIT!r}]["text"][:{num_samples}]'
    )
    lines.append(
        "# Truncate to the model's context window so `TopKInputs.interpret` and"
    )
    lines.append("# `get_activations` agree on the same tokenization.")
    lines.append(
        f"inputs = [splitter.tokenizer.decode("
        f'splitter.tokenizer(t, truncation=True, max_length={max_tokens}, add_special_tokens=False)["input_ids"], '
        f"skip_special_tokens=True) for t in raw_inputs]"
    )
    lines.append("")
    lines.append("activations, _ = splitter.get_activations(inputs, tqdm_bar=True)")
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
    lines.append("interpretations = TopKInputs(")
    lines.append("    concept_explainer=concept_explainer,")
    lines.append(f"    k={TOPK_WORDS},")
    lines.append(f"    concept_encoding_batch_size={concept_model_batch_size},")
    lines.append(").interpret(")
    lines.append("    inputs=inputs,")
    lines.append("    latent_activations=activations,")
    lines.append('    concepts_indices="all",')
    lines.append(")")
    lines.append(
        'labels = {k: [t.lstrip("\u0120") for t in v.keys()] if v else None for k, v in interpretations.items()}'
    )
    lines.append("")
    lines.append(f"sample = {sample_text!r}")
    lines.append(
        "encoded = splitter.tokenizer(sample, add_special_tokens=True)"
    )
    lines.append(
        'sample_tokens = [t.replace("\u0120", " ") for t in '
        "splitter.tokenizer.convert_ids_to_tokens(encoded['input_ids'])]"
    )
    lines.append("")
    lines.append("local_importances = concept_explainer.concept_output_gradient(")
    lines.append("    inputs=[sample], targets=None,")
    lines.append(")[0].sum(dim=1)")
    lines.append("")
    lines.append(
        "local_activations, _ = splitter.get_activations([sample], include_special_tokens=True)"
    )
    lines.append(
        "concepts_activations = concept_explainer.activations_to_concepts(local_activations)"
    )
    lines.append("")
    lines.append("plot_concepts(")
    lines.append("    concepts_activations=concepts_activations,")
    lines.append("    concepts_importances=local_importances,")
    lines.append("    concepts_labels=labels,")
    lines.append("    sample=sample_tokens,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


def render_llm_labels_snippet(
    method_name: str,
    explainer_cls: type,
    hf_model_id: str,
    split_point: int,
    batch_size: int,
    concept_model_batch_size: int,
    max_tokens: int,
    torch_dtype: torch.dtype | None,
    num_samples: int,
    sample_text: str,
    init_params: dict,
    fit_params: dict,
) -> str:
    """Return a self-contained snippet reproducing one ``_llm_labels`` HTML."""
    init_lines, init_imports = format_kwargs_lines(init_params, indent="    ")
    fit_lines, fit_imports = format_kwargs_lines(fit_params, indent="    ")
    extra_imports = sorted(init_imports | fit_imports)

    lines: list[str] = [
        "import torch",
        "from datasets import load_dataset",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from interpreto import SplitterForGeneration, plot_concepts",
        f"from interpreto.concepts import {explainer_cls.__name__}, LLMLabels",
        "",
        "# ``HuggingFaceLLM`` is not (yet) shipped by interpreto — this import will",
        "# start working once the class lands upstream. In the meantime, see the",
        '# "Using your own LLM interface" section of the generation concept',
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

    splitter_args = (
        f"{hf_model_id!r}, split_point={split_point}, "
        f"batch_size={batch_size}, device_map=device"
    )
    if torch_dtype is not None:
        splitter_args += f", torch_dtype={torch_dtype}"

    lines.append(f"splitter = SplitterForGeneration({splitter_args})")
    lines.append(
        f'raw_inputs = load_dataset({DATASET_HF_ID!r}, {DATASET_CONFIG!r})[{DATASET_SPLIT!r}]["text"][:{num_samples}]'
    )
    lines.append(
        "# Truncate to the model's context window so `LLMLabels.interpret` and"
    )
    lines.append("# `get_activations` agree on the same tokenization.")
    lines.append(
        f"inputs = [splitter.tokenizer.decode("
        f'splitter.tokenizer(t, truncation=True, max_length={max_tokens}, add_special_tokens=False)["input_ids"], '
        f"skip_special_tokens=True) for t in raw_inputs]"
    )
    lines.append("")
    lines.append("activations, _ = splitter.get_activations(inputs, tqdm_bar=True)")
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
    lines.append("interpretations = LLMLabels(")
    lines.append("    concept_explainer=concept_explainer,")
    lines.append("    llm_interface=llm_interface,")
    lines.append(f"    k_examples={LLM_LABELS_K_EXAMPLES},")
    lines.append(f"    k_context={LLM_LABELS_K_CONTEXT},")
    lines.append(f"    concept_encoding_batch_size={concept_model_batch_size},")
    lines.append(").interpret(")
    lines.append("    inputs=inputs,")
    lines.append("    latent_activations=activations,")
    lines.append('    concepts_indices="all",')
    lines.append(")")
    lines.append("labels = {k: v for k, v in interpretations.items() if v}")
    lines.append("")
    lines.append(f"sample = {sample_text!r}")
    lines.append(
        "encoded = splitter.tokenizer(sample, add_special_tokens=True)"
    )
    lines.append(
        'sample_tokens = [t.replace("\u0120", " ") for t in '
        "splitter.tokenizer.convert_ids_to_tokens(encoded['input_ids'])]"
    )
    lines.append("")
    lines.append("local_importances = concept_explainer.concept_output_gradient(")
    lines.append("    inputs=[sample], targets=None,")
    lines.append(")[0].sum(dim=1)")
    lines.append("")
    lines.append(
        "local_activations, _ = splitter.get_activations([sample], include_special_tokens=True)"
    )
    lines.append(
        "concepts_activations = concept_explainer.activations_to_concepts(local_activations)"
    )
    lines.append("")
    lines.append("plot_concepts(")
    lines.append("    concepts_activations=concepts_activations,")
    lines.append("    concepts_importances=local_importances,")
    lines.append("    concepts_labels=labels,")
    lines.append("    sample=sample_tokens,")
    lines.append(f"    top_k={TOP_K},")
    lines.append(")")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def load_inputs(num_samples: int) -> list[str]:
    debug = os.environ.get("DEBUG_SAMPLES")
    if debug:
        num_samples = int(debug)
    dataset = load_dataset(DATASET_HF_ID, DATASET_CONFIG)[DATASET_SPLIT]
    n = min(num_samples, len(dataset))
    print(f"Using {n} inputs (out of {len(dataset)} in {DATASET_SPLIT} split)")
    return list(dataset.select(range(n))["text"])


def truncate_inputs(inputs: list[str], tokenizer, max_tokens: int) -> list[str]:
    """Truncate every input to ``max_tokens`` tokens and return re-decoded strings.

    Guarantees that the same tokenization is applied later by both
    ``splitter.get_activations`` and ``TopKInputs.interpret``, so that
    the granular-input count matches the flattened activation count.
    """
    out: list[str] = []
    for text in inputs:
        ids = tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False)["input_ids"]
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def _clean_token(token: str) -> str:
    return token.replace("\u0120", " ")  # GPT-2 / Qwen space marker


def _pick_labeler_device() -> str:
    """Return a torch device for the labeler.

    When more than one CUDA device is available, use ``cuda:1`` so the
    labeler doesn't compete with the base generation model for VRAM on
    ``cuda:0`` (matters for llama3.1-8b). Otherwise fall back to
    ``device`` (which is ``cuda:0`` or ``cpu``).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "cuda:1"
    return device


def _try_load_labeler(hf_id: str, target_device: str):
    """Attempt to load ``(tokenizer, model)`` for ``hf_id``.

    Returns ``(tokenizer, model)`` on success. Returns ``(None, None)``
    if the local ``transformers`` install doesn't know the checkpoint's
    architecture (which is the usual failure mode for very-recent
    Qwen3.5 releases).
    """
    print(f"\nLoading labeler {hf_id} on {target_device} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=LABELER_TORCH_DTYPE,
        ).to(target_device)
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
    target_device = _pick_labeler_device()

    tokenizer, model = _try_load_labeler(LABELER_HF_ID, target_device)
    if model is not None:
        return LABELER_HF_ID, tokenizer, model

    if LABELER_FALLBACK_HF_ID and LABELER_FALLBACK_HF_ID != LABELER_HF_ID:
        print(f"   falling back to {LABELER_FALLBACK_HF_ID}")
        tokenizer, model = _try_load_labeler(LABELER_FALLBACK_HF_ID, target_device)
        if model is not None:
            return LABELER_FALLBACK_HF_ID, tokenizer, model

    print("   ! labeler load failed; skipping LLM-Labels branch for this run.")
    return None, None, None


def main() -> None:
    config = MODEL_CONFIGS[model_id]
    hf_model_id = config["hf_model_id"]
    split_point = config["split_point"]
    batch_size = config["batch_size"]
    concept_model_batch_size = config["concept_model_batch_size"]
    torch_dtype = config.get("torch_dtype")

    torch.manual_seed(SEED)

    skip_llm_labels = os.environ.get("SKIP_LLM_LABELS") == "1"
    only_llm_labels = os.environ.get("ONLY_LLM_LABELS") == "1"
    if skip_llm_labels and only_llm_labels:
        raise SystemExit(
            "Both SKIP_LLM_LABELS=1 and ONLY_LLM_LABELS=1 were set; "
            "these are mutually exclusive."
        )

    output_root = OUTPUT_ROOT / model_id / "concept" / "local"
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Splitter + inputs + cached activations
    # ------------------------------------------------------------------
    splitter_kwargs = dict(
        split_point=split_point,
        batch_size=batch_size,
        device_map=device,
    )
    if torch_dtype is not None:
        splitter_kwargs["torch_dtype"] = torch_dtype

    splitter = SplitterForGeneration(hf_model_id, **splitter_kwargs)
    inputs = load_inputs(config["num_samples"])
    inputs = truncate_inputs(inputs, splitter.tokenizer, config["max_tokens"])

    activations, _ = cache_activations(
        splitter,
        inputs,
        activations_cache_path(model_id),
        tqdm_bar=True,
        activation_dtype=ACTIVATIONS_SAVE_DTYPE,
        load_dtype=ACTIVATIONS_LOAD_DTYPE,
    )
    print(f"activations shape: {tuple(activations.shape)}")

    init_parameters = build_init_parameters(config)
    fit_parameters = build_fit_parameters(config)

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
        init_params = init_parameters.get(method_name, {})
        fit_params = fit_parameters.get(method_name, {})

        print(f"\n== {method_name}")
        concept_explainer = explainer_cls(splitter, **init_params)

        explainer_path = explainers_cache_dir(model_id) / f"{method_name}.pt"
        # BatchTopKSAE stores a "running_threshold" that is populated during
        # training; saving the state_dict alone loses it, so we always refit.
        skip_cache = explainer_cls is BatchTopKSAEConcepts
        if explainer_cls is NeuronsAsConcepts:
            pass
        elif explainer_path.exists() and not skip_cache:
            print(f"   loading fitted explainer from {explainer_path}")
            try:
                load_concept_model(concept_explainer, explainer_path, device)
            except Exception as exc:  # noqa: BLE001
                print(f"   load failed ({exc}); refitting")
                concept_explainer.fit(activations, **fit_params)
        else:
            print(f"   fitting on {activations.shape[0]} activations")
            concept_explainer.fit(activations, **fit_params)
            try:
                save_concept_model(concept_explainer, explainer_path)
                print(f"   saved to {explainer_path}")
            except NotImplementedError as exc:
                print(f"   (skipped save: {exc})")

        # ---- TopK interpretation ------------------------------------
        topk_labels = None
        if not only_llm_labels:
            interpretations = TopKInputs(
                concept_explainer=concept_explainer,
                k=TOPK_WORDS,
                concept_encoding_batch_size=concept_model_batch_size,
            ).interpret(
                inputs=inputs,
                latent_activations=activations,
                concepts_indices="all",
            )
            topk_labels = {
                k: [_clean_token(t) for t in v.keys()] if v else None
                for k, v in interpretations.items()
            }

        # ---- LLM-Labels interpretation -------------------------------
        llm_labels_map = None
        if my_llm is not None:
            llm_interpretations = LLMLabels(
                concept_explainer=concept_explainer,
                llm_interface=my_llm,
                k_examples=LLM_LABELS_K_EXAMPLES,
                k_context=LLM_LABELS_K_CONTEXT,
                concept_encoding_batch_size=concept_model_batch_size,
            ).interpret(
                inputs=inputs,
                latent_activations=activations,
                concepts_indices="all",
            )
            llm_labels_map = {
                k: v for k, v in llm_interpretations.items() if v
            }

        # ---- per-sample HTML -----------------------------------------
        for i, sample in enumerate(SAMPLES):
            sample_dir = output_root / f"sample-{i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Match the sample tokenization to what the splitter sees:
            # ``get_activations`` calls the tokenizer with its default
            # ``add_special_tokens=True``. Some models (e.g. Llama) auto-prepend
            # a BOS, so ``.tokenize(sample)`` alone would be off by one.
            encoded = splitter.tokenizer(sample, add_special_tokens=True)
            sample_tokens = [
                _clean_token(t)
                for t in splitter.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            ]

            local_importances = concept_explainer.concept_output_gradient(
                inputs=[sample], targets=None,
            )[0].sum(dim=1)

            local_activations, _ = splitter.get_activations(
                [sample], include_special_tokens=True
            )
            concepts_activations = concept_explainer.activations_to_concepts(
                local_activations
            )

            # ---- TopK variant --------------------------------------
            if topk_labels is not None:
                html_path = sample_dir / f"{method_name}.html"
                code_path = html_path.with_suffix(".py")

                print(f"   saving {html_path}")
                plot_concepts(
                    concepts_activations=concepts_activations,
                    concepts_importances=local_importances,
                    concepts_labels=topk_labels,
                    sample=sample_tokens,
                    top_k=TOP_K,
                    save_path=str(html_path),
                )

                code_path.write_text(
                    render_code_snippet(
                        method_name=method_name,
                        explainer_cls=explainer_cls,
                        hf_model_id=hf_model_id,
                        split_point=split_point,
                        batch_size=batch_size,
                        concept_model_batch_size=concept_model_batch_size,
                        max_tokens=config["max_tokens"],
                        torch_dtype=torch_dtype,
                        num_samples=config["num_samples"],
                        sample_text=sample,
                        init_params=init_params,
                        fit_params=fit_params,
                    ),
                    encoding="utf-8",
                )

            # ---- LLM-Labels variant --------------------------------
            if llm_labels_map is not None:
                html_path_llm = sample_dir / f"{method_name}_llm_labels.html"
                code_path_llm = html_path_llm.with_suffix(".py")

                print(f"   saving {html_path_llm}")
                plot_concepts(
                    concepts_activations=concepts_activations,
                    concepts_importances=local_importances,
                    concepts_labels=llm_labels_map,
                    sample=sample_tokens,
                    top_k=TOP_K,
                    save_path=str(html_path_llm),
                )

                code_path_llm.write_text(
                    render_llm_labels_snippet(
                        method_name=method_name,
                        explainer_cls=explainer_cls,
                        hf_model_id=hf_model_id,
                        split_point=split_point,
                        batch_size=batch_size,
                        concept_model_batch_size=concept_model_batch_size,
                        max_tokens=config["max_tokens"],
                        torch_dtype=torch_dtype,
                        num_samples=config["num_samples"],
                        sample_text=sample,
                        init_params=init_params,
                        fit_params=fit_params,
                    ),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
