"""Process-local backport of https://github.com/huggingface/peft/pull/3234.

PEFT 0.19.1 requires a requantization callback even for dynamic LoRA
inference on directly quantized torchao models. Only merge/unmerge need it.
Installed packages are never edited by this compatibility shim.
"""

from __future__ import annotations

import inspect
import logging
import threading
import warnings
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)
_patch_lock = threading.Lock()
_MERGE_ERROR = (
    "Quantized LoRA merge/unmerge requires get_apply_tensor_subclass. "
    "Use dynamic adapters or merge into a full-precision base first."
)


def ensure_peft_torchao_compatibility() -> bool:
    """Patch the affected PEFT release once, before constructing a runtime.

    Return whether this call installed the shim. Newer releases, environments
    without torchao, and installations with the earlier file backport are left
    alone.
    """
    try:
        if version("peft") != "0.19.1":
            return False
    except PackageNotFoundError:
        return False

    from peft.import_utils import is_torchao_available

    if not is_torchao_available():
        return False

    from peft.tuners.lora.torchao import TorchaoLoraLinear
    from peft.tuners.tuners_utils import check_adapters_to_merge

    with _patch_lock:
        cls = TorchaoLoraLinear
        if getattr(cls, "_irodori_optional_torchao_callback", False):
            return False
        parameter = inspect.signature(cls.__init__).parameters.get("get_apply_tensor_subclass")
        if parameter is None or parameter.default is not inspect.Parameter.empty:
            return False

        original_init = cls.__init__
        original_merge = cls.merge
        original_unmerge = cls.unmerge

        @wraps(original_init)
        def compatible_init(self, *args, get_apply_tensor_subclass=None, **kwargs):
            original_init(
                self, *args, get_apply_tensor_subclass=get_apply_tensor_subclass, **kwargs
            )
            if get_apply_tensor_subclass is None:
                warnings.warn(
                    "Torchao LoRA supports inference without a requantization callback; "
                    "merge/unmerge requires get_apply_tensor_subclass.",
                    stacklevel=2,
                )

        @wraps(original_merge)
        def compatible_merge(self, safe_merge=False, adapter_names=None):
            if self.get_apply_tensor_subclass is None:
                if not check_adapters_to_merge(self, adapter_names):
                    return
                raise ValueError(_MERGE_ERROR)
            return original_merge(self, safe_merge=safe_merge, adapter_names=adapter_names)

        @wraps(original_unmerge)
        def compatible_unmerge(self):
            if self.merged and self.get_apply_tensor_subclass is None:
                raise ValueError(_MERGE_ERROR)
            return original_unmerge(self)

        cls.__init__ = compatible_init
        cls.merge = compatible_merge
        cls.unmerge = compatible_unmerge
        cls._irodori_optional_torchao_callback = True
        logger.info("Applied PEFT 0.19.1 torchao dynamic LoRA compatibility fix")
        return True
