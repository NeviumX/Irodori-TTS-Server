from __future__ import annotations

import gc
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from irodori_openai_tts import loading
from irodori_tts.config import ModelConfig
from irodori_tts.inference_runtime import RuntimeKey


def test_precision_conversion_preserves_complex_integer_buffers_and_tied_weights():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    model.alias = model.weight
    model.weight.grad = torch.ones_like(model.weight)
    model.register_buffer("floating", torch.ones(8, dtype=torch.bfloat16))
    model.register_buffer("positions", torch.arange(8))
    model.register_buffer("rotary", torch.ones(8, dtype=torch.complex64) * (1 + 2j))
    expected = model.weight.detach().float()

    loading._move_model(model, device=torch.device("cpu"), dtype=torch.float32)

    torch.testing.assert_close(model.weight, expected)
    assert model.alias is model.weight
    assert model.weight.grad.dtype == torch.float32
    assert model.floating.dtype == torch.float32
    assert model.positions.dtype == torch.int64
    assert model.rotary.dtype == torch.complex64
    torch.testing.assert_close(model.rotary, torch.ones(8, dtype=torch.complex64) * (1 + 2j))


@pytest.mark.parametrize("quantization", [None, "int8_weight_only", "int8_dynamic"])
@pytest.mark.parametrize("device, precision", [("cpu", "fp32"), ("cuda", "bf16")])
def test_load_runtime_roundtrip_preserves_quantization_and_upstream_setup(
    tmp_path, monkeypatch, quantization, device, precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )

    from irodori_tts.quantization import flatten_quantized_state_dict

    def tiny_model(*args, **kwargs):
        return torch.nn.Sequential(torch.nn.Linear(64, 64, bias=False))

    source = tiny_model().to(dtype=torch.bfloat16)
    if quantization:
        config_type = (
            Int8WeightOnlyConfig
            if quantization == "int8_weight_only"
            else Int8DynamicActivationInt8WeightConfig
        )
        quantize_(source, config_type(version=2))
    config = ModelConfig(text_encoder_type="pretrained", use_caption_condition=True)
    config_dict = asdict(config)
    config_dict.update(max_text_len=123, max_caption_len=234, ref_max_seconds=45.0)
    metadata = {"config_json": json.dumps(config_dict)}
    state = source.state_dict()
    if quantization:
        state, metadata = flatten_quantized_state_dict(
            state,
            base_metadata=metadata,
            quantization_type=(
                "int8_weight_only"
                if quantization == "int8_weight_only"
                else "int8_dynamic_activation_int8_weight"
            ),
            profile="core",
            compute_dtype=torch.bfloat16,
            quantized_modules=1,
        )
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, str(checkpoint), metadata=metadata)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer_config.json").write_text("{}")
    tokenizer_calls = []
    codec_calls = []
    compile_calls = []
    monkeypatch.setattr(loading.upstream, "TextToLatentRFDiT", tiny_model)

    def tokenizer(**kwargs):
        tokenizer_calls.append(kwargs)
        return SimpleNamespace(vocab_size=config.text_vocab_size)

    def codec(**kwargs):
        codec_calls.append(kwargs)
        return SimpleNamespace(latent_dim=config.latent_dim, device=torch.device(kwargs["device"]))

    def compile_model(model, **kwargs):
        compile_calls.append((model[0].weight.device.type, kwargs))
        return model

    monkeypatch.setattr(loading.upstream.PretrainedTextTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(loading.upstream.DACVAECodec, "load", codec)
    monkeypatch.setattr(loading.upstream, "SilentCipherWatermarker", lambda **kw: kw)
    monkeypatch.setattr(loading.upstream, "_maybe_compile_inference_model", compile_model)
    key = RuntimeKey(
        checkpoint=str(checkpoint),
        model_device=device,
        model_precision=precision,
        codec_device=device,
        codec_precision="fp32",
        compile_model=True,
        compile_dynamic=True,
    )

    runtime = loading.load_runtime(key)

    assert runtime.key is key
    assert runtime.model_device.type == device
    assert runtime.codec_device.type == device
    assert runtime.default_text_max_len == 123
    assert runtime.default_caption_max_len == 234
    assert runtime.default_max_ref_seconds == 45.0
    assert runtime._model_dtype == runtime.model[0].weight.dtype
    assert runtime.model[0].weight.device.type == device
    assert not runtime.model.training
    assert len(tokenizer_calls) == 2
    assert all(call["local_files_only"] and call["revision"] is None for call in tokenizer_calls)
    assert codec_calls[0]["device"] == device
    assert codec_calls[0]["dtype"] == torch.float32
    assert compile_calls == [
        ("cpu", {"enabled": False, "dynamic": True}),
        (device, {"enabled": True, "dynamic": True}),
    ]
    weight = runtime.model[0].weight
    if quantization:
        assert type(weight) is type(source[0].weight)
        assert weight.qdata.dtype == torch.int8
        torch.testing.assert_close(weight.qdata.cpu(), source[0].weight.qdata)
        torch.testing.assert_close(weight.scale.cpu(), source[0].weight.scale)
        torch.testing.assert_close(weight.dequantize().cpu(), source[0].weight.dequantize())
    else:
        target_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        torch.testing.assert_close(weight.cpu(), source[0].weight.to(dtype=target_dtype))

    target_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    source.to(device=device, dtype=target_dtype)
    value = torch.randn(2, 64, device=device, dtype=target_dtype)
    with torch.no_grad():
        torch.testing.assert_close(runtime.model(value), source(value))


def test_load_runtime_rejects_invalid_precision_before_loading(monkeypatch):
    monkeypatch.setattr(
        loading.InferenceRuntime,
        "from_key",
        lambda key: pytest.fail("invalid request reached checkpoint loading"),
    )
    with pytest.raises(ValueError, match="requires CUDA or XPU"):
        loading.load_runtime(
            RuntimeKey(checkpoint="unused", model_device="cpu", model_precision="bf16")
        )


def test_loading_cache_releases_each_cuda_device_once(monkeypatch, caplog):
    from contextlib import nullcontext

    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device", lambda index: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("release"))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 100 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 80 * 2**20)
    with caplog.at_level("INFO"):
        loading._release_loading_cache(
            SimpleNamespace(model_device=torch.device("cuda"), codec_device=torch.device("cuda:1"))
        )
    assert calls == ["release"]
    assert "allocated=80.0 MiB" in caplog.text


def test_cpu_loading_does_not_touch_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("CUDA cache touched"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("CUDA initialized"))
    loading._release_loading_cache(
        SimpleNamespace(model_device=torch.device("cpu"), codec_device=torch.device("cpu"))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cast_avoids_old_and_new_weights_overlapping_on_cuda():
    # Synthetic 32 MiB BF16 weight: the old path overlaps it with its 64 MiB
    # FP32 replacement. The new path only allocates the final 64 MiB on CUDA.
    def measure(old_path):
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.zeros(4096, 4096, dtype=torch.bfloat16))
        try:
            if old_path:
                model.to("cuda")
                loading.upstream._move_inference_module(
                    model, device=torch.device("cuda"), dtype=torch.float32
                )
            else:
                loading._move_model(model, device=torch.device("cuda"), dtype=torch.float32)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - baseline
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    old_peak = measure(True)
    new_peak = measure(False)
    print(
        f"synthetic CUDA loading peak: old={old_peak / 2**20:.1f} MiB, new={new_peak / 2**20:.1f} MiB"
    )
    assert new_peak <= 65 * 2**20
    assert old_peak >= new_peak + 30 * 2**20
