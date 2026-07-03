"""Shared helpers for the explanation-generation scripts.

Everything in this module is script-only. It must never be imported by the
snippets we render out into ``explanations/`` — those snippets stay minimal
and self-contained.

The main features are:

* ``cache_activations`` — compute or load ``(activations, predictions)``
  from a splitter, caching them under ``data/<model_id>/activations.pt``.
* ``save_concept_model`` / ``load_concept_model`` — persist and restore a
  fitted concept explainer's ``concept_model`` weights. Copied and
  hardened from ``references/save_load_concepts.py``.
* ``format_value`` / ``format_kwargs_lines`` — turn Python objects back
  into source strings for the rendered ``.py`` snippets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------


def model_data_dir(model_id: str) -> Path:
    """Return the ``data/<model_id>`` folder, creating it on demand."""
    path = DATA_ROOT / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def activations_cache_path(model_id: str, tag: str = "activations") -> Path:
    return model_data_dir(model_id) / f"{tag}.pt"


def explainers_cache_dir(model_id: str) -> Path:
    path = model_data_dir(model_id) / "explainers"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# Activations caching
# ----------------------------------------------------------------------


def cache_activations(
    splitter,
    inputs: list[str],
    cache_path: Path,
    *,
    force: bool = False,
    activation_dtype: torch.dtype | None = None,
    load_dtype: torch.dtype | None = None,
    **get_activations_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute or load a splitter's ``(activations, predictions)`` tuple.

    * ``force`` — bypass cache and recompute.
    * ``activation_dtype`` — if given, cast activations before saving to
      shrink the on-disk footprint (e.g. ``torch.float16`` for generation).
    * ``load_dtype`` — if given, cast activations back to this dtype after
      loading from disk (typically ``torch.float32`` so downstream code
      doesn't have to worry about low-precision surprises).
    * ``get_activations_kwargs`` — forwarded to ``splitter.get_activations``.
    """
    if cache_path.exists() and not force:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        activations = payload["activations"]
        if load_dtype is not None and activations.dtype != load_dtype:
            activations = activations.to(load_dtype)
        return activations, payload.get("predictions")

    activations, predictions = splitter.get_activations(
        inputs=inputs, **get_activations_kwargs
    )

    to_save = (
        activations.to(activation_dtype)
        if activation_dtype is not None
        else activations
    )
    payload = {"activations": to_save, "predictions": predictions}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return activations, predictions


# ----------------------------------------------------------------------
# Concept model save/load
#
# Copied and hardened from references/save_load_concepts.py.
# ----------------------------------------------------------------------


def save_concept_model(concept_explainer, model_path: Path) -> None:
    """Persist a fitted concept explainer's concept_model to disk.

    Handles:
    * ``SemiNMFConcepts`` — save the dictionary ``D`` tensor.
    * Any concept_model exposing ``state_dict`` (SAEs, sklearn-wrapped
      PCA/ICA/SVD via torch modules): save the state dict.

    Raises ``NotImplementedError`` for concept models that do not fit
    either mold (e.g. ``NeuronsAsConcepts`` has no state and does not
    need saving anyway; the caller should skip it before calling this).
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    concept_model = concept_explainer.concept_model
    cls_name = concept_explainer.__class__.__name__

    if cls_name == "SemiNMFConcepts":
        torch.save(concept_model.D, model_path)
        return
    if hasattr(concept_model, "state_dict"):
        torch.save(concept_model.state_dict(), str(model_path))
        return

    raise NotImplementedError(
        f"Saving not implemented for concept explainer {cls_name!r}"
    )


def load_concept_model(concept_explainer, model_path: Path, device: str):
    """Restore a fitted concept explainer's concept_model from disk.

    The passed ``concept_explainer`` must have been instantiated with the
    same splitter, ``nb_concepts``, and other init kwargs used at
    training time. This function only refills the weights and marks the
    explainer as fitted.
    """
    concept_model = concept_explainer.concept_model
    cls_name = concept_explainer.__class__.__name__

    if hasattr(concept_model, "_set_fitted"):
        concept_model._set_fitted()

    if cls_name == "SemiNMFConcepts":
        D = torch.load(str(model_path), map_location=device, weights_only=False)
        concept_model.D = D
        return concept_explainer

    if hasattr(concept_model, "load_state_dict"):
        concept_model.load_state_dict(
            torch.load(str(model_path), map_location=device, weights_only=True)
        )
        concept_explainer.to(device)
        if hasattr(concept_model, "training"):
            concept_model.training = False
        return concept_explainer

    raise NotImplementedError(
        f"Loading not implemented for concept explainer {cls_name!r}"
    )


# ----------------------------------------------------------------------
# Snippet rendering helpers
# ----------------------------------------------------------------------


def format_value(
    value: Any,
    param_name: str = "",
    *,
    device_variable: str = "device",
) -> tuple[str, set[str]]:
    """Turn ``value`` back into source code for a rendered snippet.

    Returns ``(source, extra_imports)`` where ``extra_imports`` is a set
    of interpreto-namespaced class names the snippet must import from
    ``interpreto.concepts.methods.overcomplete``.
    """
    imports: set[str] = set()

    if param_name == "device":
        return device_variable, imports

    if isinstance(value, type):
        module = value.__module__
        if module.startswith("torch.optim.lr_scheduler"):
            return f"torch.optim.lr_scheduler.{value.__name__}", imports
        if module.startswith("torch.optim"):
            return f"torch.optim.{value.__name__}", imports
        if module.startswith("interpreto."):
            imports.add(value.__name__)
            return value.__name__, imports
        return value.__name__, imports

    return repr(value), imports


def format_kwargs_lines(
    params: dict[str, Any],
    indent: str = "    ",
    *,
    device_variable: str = "device",
) -> tuple[list[str], set[str]]:
    """Turn a kwargs dict into a list of ``key=value,`` source lines."""
    lines: list[str] = []
    imports: set[str] = set()
    for key, value in params.items():
        value_str, value_imports = format_value(
            value, key, device_variable=device_variable
        )
        imports.update(value_imports)
        lines.append(f"{indent}{key}={value_str},")
    return lines, imports


def dedupe(names: Iterable[str]) -> list[str]:
    """Preserve order, drop duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out
