"""Load through the upstream runtime without staging model casts on the GPU."""

from __future__ import annotations

import logging
from dataclasses import replace

import torch

from irodori_tts import inference_runtime as upstream
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

logger = logging.getLogger(__name__)


def _move_model(
    model: torch.nn.Module, *, device: torch.device, dtype: torch.dtype
) -> torch.nn.Module:
    # Convert on the source device before transferring. Do not use model.to(dtype=...),
    # which would also cast complex buffers. Tensor-subclass .to() preserves the
    # torchao quantized storage, just as in the upstream inference mover.
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(dtype=dtype)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(dtype=dtype)
        for child in model.modules():
            for name, buffer in child._buffers.items():
                if buffer is not None and buffer.is_floating_point() and buffer.dtype != dtype:
                    child._buffers[name] = buffer.to(dtype=dtype)
    return model.to(device=device)


def load_runtime(key: RuntimeKey) -> InferenceRuntime:
    model_device = upstream.resolve_runtime_device(key.model_device)
    model_dtype = upstream.resolve_runtime_dtype(precision=key.model_precision, device=model_device)

    # Reuse upstream checkpoint deserialization, strict state loading, tokenizers,
    # inference defaults, codec and watermark setup. Only the TTS model is staged
    # on CPU; the codec keeps the requested device/precision. Compile after moving.
    cpu_key = replace(key, model_device="cpu", model_precision="fp32", compile_model=False)
    runtime = InferenceRuntime.from_key(cpu_key)
    runtime.model = _move_model(runtime.model, device=model_device, dtype=model_dtype)
    runtime.model = upstream._maybe_compile_inference_model(
        runtime.model, enabled=bool(key.compile_model), dynamic=bool(key.compile_dynamic)
    )
    runtime.key = key
    runtime.model_device = model_device
    runtime._model_dtype = next(runtime.model.parameters()).dtype
    _release_loading_cache(runtime)
    return runtime


def _release_loading_cache(runtime: InferenceRuntime) -> None:
    """Return unused loading buffers; resident model weights remain allocated."""
    devices = {runtime.model_device, runtime.codec_device}
    cuda_devices = {
        device.index if device.index is not None else torch.cuda.current_device()
        for device in devices
        if device.type == "cuda"
    }
    for index in sorted(cuda_devices):
        with torch.cuda.device(index):
            before = torch.cuda.memory_reserved(index)
            torch.cuda.empty_cache()
            logger.info(
                "CUDA memory after runtime load (cuda:%d): allocated=%.1f MiB, "
                "reserved=%.1f MiB, released=%.1f MiB",
                index,
                torch.cuda.memory_allocated(index) / 2**20,
                torch.cuda.memory_reserved(index) / 2**20,
                (before - torch.cuda.memory_reserved(index)) / 2**20,
            )
    for device_type in {device.type for device in devices} - {"cuda", "cpu"}:
        backend = getattr(torch, device_type, None)
        if backend is not None and hasattr(backend, "empty_cache"):
            backend.empty_cache()
