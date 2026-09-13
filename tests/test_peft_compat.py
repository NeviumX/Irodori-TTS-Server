from __future__ import annotations

import warnings

import pytest

from irodori_openai_tts import peft_compat


@pytest.fixture
def unpatched_layer(monkeypatch):
    backend = pytest.importorskip("peft.tuners.lora.torchao")
    import_utils = pytest.importorskip("peft.import_utils")

    class Layer:
        def __init__(self, *args, get_apply_tensor_subclass, **kwargs):
            self.get_apply_tensor_subclass = get_apply_tensor_subclass
            self.active_adapters = ["alice"]
            self.merged_adapters = []
            self.calls = []

        @property
        def merged(self):
            return bool(self.merged_adapters)

        def merge(self, safe_merge=False, adapter_names=None):
            self.calls.append(("merge", safe_merge, adapter_names))

        def unmerge(self):
            self.calls.append(("unmerge",))

    monkeypatch.setattr(backend, "TorchaoLoraLinear", Layer)
    monkeypatch.setattr(import_utils, "is_torchao_available", lambda: True)
    monkeypatch.setattr(peft_compat, "version", lambda _: "0.19.1")
    return Layer


def test_missing_callback_is_optional_and_patch_is_idempotent(unpatched_layer):
    with pytest.raises(TypeError, match="get_apply_tensor_subclass"):
        unpatched_layer()
    assert peft_compat.ensure_peft_torchao_compatibility()
    patched_init = unpatched_layer.__init__
    assert not peft_compat.ensure_peft_torchao_compatibility()
    assert unpatched_layer.__init__ is patched_init
    with pytest.warns(UserWarning, match="merge/unmerge"):
        layer = unpatched_layer()
    assert layer.get_apply_tensor_subclass is None


def test_merge_and_unmerge_fail_before_mutation(unpatched_layer):
    peft_compat.ensure_peft_torchao_compatibility()
    with pytest.warns(UserWarning):
        layer = unpatched_layer()
    with pytest.raises(ValueError, match="get_apply_tensor_subclass"):
        layer.merge()
    assert layer.calls == []
    assert layer.merged_adapters == []
    # No work to do remains a no-op, including already unmerged layers.
    layer.merge(adapter_names=[])
    assert layer.calls == []
    layer.unmerge()
    assert layer.calls == [("unmerge",)]
    layer.calls.clear()
    layer.merged_adapters.append("alice")
    with pytest.raises(ValueError, match="get_apply_tensor_subclass"):
        layer.unmerge()
    assert layer.calls == []
    assert layer.merged_adapters == ["alice"]


def test_existing_callback_and_merge_arguments_are_preserved(unpatched_layer):
    peft_compat.ensure_peft_torchao_compatibility()
    callback = object()
    with warnings.catch_warnings(record=True) as caught:
        layer = unpatched_layer(get_apply_tensor_subclass=callback)
    assert not caught
    assert layer.get_apply_tensor_subclass is callback
    layer.merge(True, ["alice"])
    layer.unmerge()
    assert layer.calls == [("merge", True, ["alice"]), ("unmerge",)]


@pytest.mark.parametrize("installed_version", ["0.18.0", "0.19.2", "0.20.0"])
def test_other_versions_are_untouched(unpatched_layer, monkeypatch, installed_version):
    original = unpatched_layer.__init__
    monkeypatch.setattr(peft_compat, "version", lambda _: installed_version)
    assert not peft_compat.ensure_peft_torchao_compatibility()
    assert unpatched_layer.__init__ is original


def test_existing_backport_is_untouched(unpatched_layer, monkeypatch):
    def already_fixed(self, *args, get_apply_tensor_subclass=None, **kwargs):
        pass

    monkeypatch.setattr(unpatched_layer, "__init__", already_fixed)
    original_merge = unpatched_layer.merge
    assert not peft_compat.ensure_peft_torchao_compatibility()
    assert unpatched_layer.__init__ is already_fixed
    assert unpatched_layer.merge is original_merge


def test_missing_optional_packages(monkeypatch):
    def missing_version(_):
        raise peft_compat.PackageNotFoundError("peft")

    monkeypatch.setattr(peft_compat, "version", missing_version)
    assert not peft_compat.ensure_peft_torchao_compatibility()
    import_utils = pytest.importorskip("peft.import_utils")
    monkeypatch.setattr(peft_compat, "version", lambda _: "0.19.1")
    monkeypatch.setattr(import_utils, "is_torchao_available", lambda: False)
    assert not peft_compat.ensure_peft_torchao_compatibility()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("quantization", ["int8_weight_only", "int8_dynamic"])
def test_real_int8_lora_load_switch_and_base_restoration(tmp_path, device, quantization):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    ao = pytest.importorskip("torchao.quantization")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    peft_compat.ensure_peft_torchao_compatibility()
    torch.manual_seed(321)
    model = torch.nn.Sequential(torch.nn.Linear(64, 64, bias=False)).to(
        device=device, dtype=torch.bfloat16
    )
    config_type = (
        ao.Int8WeightOnlyConfig
        if quantization == "int8_weight_only"
        else ao.Int8DynamicActivationInt8WeightConfig
    )
    ao.quantize_(model, config_type(version=2))
    value = torch.randn(2, 64, device=device, dtype=torch.bfloat16)
    config = peft.LoraConfig(r=4, lora_alpha=4, target_modules=["0"])
    with torch.no_grad():
        expected_base = model(value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            wrapped = peft.get_peft_model(model, config, adapter_name="alice")
        layer = wrapped.base_model.model[0]
        layer.lora_A["alice"].weight.fill_(0.1)
        layer.lora_B["alice"].weight.fill_(0.1)
        wrapped.save_pretrained(tmp_path)
        wrapped.load_adapter(tmp_path / "alice", adapter_name="bob")
        layer.lora_A["bob"].weight.fill_(0.2)
        layer.lora_B["bob"].weight.fill_(0.2)
        wrapped.set_adapter("alice")
        alice = wrapped(value)
        wrapped.set_adapter("bob")
        bob = wrapped(value)
        assert not torch.allclose(alice, bob)
        with wrapped.disable_adapter():
            torch.testing.assert_close(wrapped(value), expected_base)
        wrapped.set_adapter("alice")
        torch.testing.assert_close(wrapped(value), alice)
        weight = layer.get_base_layer().weight
        with pytest.raises(ValueError, match="get_apply_tensor_subclass"):
            layer.merge()
        assert layer.get_base_layer().weight is weight
        assert layer.merged_adapters == []
        # Exercise the unmerge guard without changing any quantized weights.
        layer.merged_adapters.append("alice")
        with pytest.raises(ValueError, match="get_apply_tensor_subclass"):
            layer.unmerge()
        assert layer.get_base_layer().weight is weight
        assert layer.merged_adapters == ["alice"]
        layer.merged_adapters.clear()
        with wrapped.disable_adapter():
            torch.testing.assert_close(wrapped(value), expected_base)
