"""Post-training int8 dynamic quantization for Chronos-Bolt on CPU.

Weights of ``nn.Linear`` layers are stored as int8; activations are
quantized on the fly per batch, so no calibration data is needed.
Chronos-Bolt is a T5 encoder-decoder, which puts almost all of its compute in
Linear layers (attention projections and feed-forward blocks).

The input and output patch embeddings are kept in float32 by default: they
map raw (scaled) values in and quantile values out, and are where precision
loss shows up most directly in forecast accuracy, while being a small share
of the compute.
"""

from __future__ import annotations

import copy
import io
import platform
from collections.abc import Iterable
from typing import Any

SENSITIVE_LAYERS = ("input_patch_embedding", "output_patch_embedding")


def select_engine() -> str:
    """Pick the quantized kernel backend for this CPU (qnnpack on ARM)."""
    import torch

    supported = torch.backends.quantized.supported_engines
    arm = platform.machine().lower() in {"arm64", "aarch64"}
    preference = ("qnnpack", "x86", "fbgemm") if arm else ("x86", "fbgemm", "qnnpack")
    for engine in preference:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"no quantized engine available (supported: {supported})")


def _skipped(name: str, skip: tuple[str, ...]) -> bool:
    return any(name == s or name.startswith(f"{s}.") or f".{s}." in f".{name}." for s in skip)


def quantize_int8(
    model: Any, skip: Iterable[str] = SENSITIVE_LAYERS, per_channel: bool = False
) -> Any:
    """Return an int8 dynamically quantized copy of ``model``.

    ``skip`` lists module names (or name prefixes) to keep in float32.
    Per-channel weight scales are usually a little more accurate but are not
    supported by every backend.
    """
    import torch
    from torch.ao.quantization import (
        default_dynamic_qconfig,
        per_channel_dynamic_qconfig,
        quantize_dynamic,
    )

    select_engine()
    skip = tuple(skip)
    qconfig = per_channel_dynamic_qconfig if per_channel else default_dynamic_qconfig
    targets = {
        name: qconfig
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and not _skipped(name, skip)
    }
    quantized = quantize_dynamic(model.float().eval(), qconfig_spec=targets, dtype=torch.qint8)
    return quantized.eval()


def quantize_forecaster(forecaster: Any, name: str | None = None, **kwargs: Any) -> Any:
    """Copy of a :class:`~daf.forecasting.model.ChronosBolt` with an int8 model.

    The original forecaster keeps its float32 model, so both can be
    benchmarked side by side.
    """
    from daf.forecasting.model import ChronosBolt

    pipeline = copy.copy(forecaster.pipeline)
    pipeline.model = quantize_int8(forecaster.pipeline.model, **kwargs)
    if hasattr(pipeline, "inner_model"):
        pipeline.inner_model = pipeline.model
    return ChronosBolt(
        model_id=forecaster.model_id,
        pipeline=pipeline,
        name=name or f"{forecaster.name}-int8",
        context_transform=forecaster.context_transform,
    )


def model_size_mb(model: Any) -> float:
    """Serialized size of the model's state dict, in MB."""
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


def count_linear_layers(model: Any) -> dict[str, int]:
    """How many Linear layers are float vs. dynamically quantized."""
    import torch
    from torch.ao.nn.quantized import dynamic as qdynamic

    counts = {"float": 0, "int8": 0}
    for module in model.modules():
        if isinstance(module, qdynamic.Linear):
            counts["int8"] += 1
        elif isinstance(module, torch.nn.Linear):
            counts["float"] += 1
    return counts
