from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Dict, Type

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_hip, mxfp_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


class _LazyQuantConfigProxy:
    def __init__(self, module_path: str, class_name: str):
        self.module_path = module_path
        self.class_name = class_name

    def _resolve(self):
        module = importlib.import_module(self.module_path)
        return getattr(module, self.class_name)

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"_LazyQuantConfigProxy(module_path={self.module_path!r}, "
            f"class_name={self.class_name!r})"
        )


_CLASS_SPECS = {
    "AutoRoundConfig": (
        "sglang.srt.layers.quantization.auto_round",
        "AutoRoundConfig",
    ),
    "AWQConfig": ("sglang.srt.layers.quantization.awq", "AWQConfig"),
    "AWQCPUConfig": ("sglang.srt.layers.quantization.awq", "AWQCPUConfig"),
    "AWQMarlinConfig": (
        "sglang.srt.layers.quantization.awq",
        "AWQMarlinConfig",
    ),
    "BitsAndBytesConfig": (
        "sglang.srt.layers.quantization.bitsandbytes",
        "BitsAndBytesConfig",
    ),
    "BlockInt8Config": (
        "sglang.srt.layers.quantization.blockwise_int8",
        "BlockInt8Config",
    ),
    "CompressedTensorsConfig": (
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
        "CompressedTensorsConfig",
    ),
    "CPUGPTQConfig": (
        "sglang.srt.layers.quantization.gptq_cpu",
        "CPUGPTQConfig",
    ),
    "FBGEMMFp8Config": (
        "sglang.srt.layers.quantization.fpgemm_fp8",
        "FBGEMMFp8Config",
    ),
    "Fp8Config": ("sglang.srt.layers.quantization.fp8", "Fp8Config"),
    "GGUFConfig": ("sglang.srt.layers.quantization.gguf", "GGUFConfig"),
    "GPTQConfig": ("sglang.srt.layers.quantization.gptq", "GPTQConfig"),
    "GPTQMarlinConfig": (
        "sglang.srt.layers.quantization.gptq",
        "GPTQMarlinConfig",
    ),
    "ModelOptFp4Config": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp4Config",
    ),
    "ModelOptFp8Config": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp8Config",
    ),
    "ModelOptMixedPrecisionConfig": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptMixedPrecisionConfig",
    ),
    "ModelSlimConfig": (
        "sglang.srt.layers.quantization.modelslim.modelslim",
        "ModelSlimConfig",
    ),
    "MoeWNA16Config": (
        "sglang.srt.layers.quantization.moe_wna16",
        "MoeWNA16Config",
    ),
    "Mxfp4Config": ("sglang.srt.layers.quantization.mxfp4", "Mxfp4Config"),
    "PetitNvFp4Config": (
        "sglang.srt.layers.quantization.petit",
        "PetitNvFp4Config",
    ),
    "QoQConfig": ("sglang.srt.layers.quantization.qoq", "QoQConfig"),
    "QuarkConfig": (
        "sglang.srt.layers.quantization.quark.quark",
        "QuarkConfig",
    ),
    "QuarkInt4Fp8Config": (
        "sglang.srt.layers.quantization.quark_int4fp8_moe",
        "QuarkInt4Fp8Config",
    ),
    "W4AFp8Config": (
        "sglang.srt.layers.quantization.w4afp8",
        "W4AFp8Config",
    ),
    "W8A8Fp8Config": (
        "sglang.srt.layers.quantization.w8a8_fp8",
        "W8A8Fp8Config",
    ),
    "W8A8Int8Config": (
        "sglang.srt.layers.quantization.w8a8_int8",
        "W8A8Int8Config",
    ),
}

_METHOD_TO_CLASS = {
    "fp8": "Fp8Config",
    "mxfp8": "Fp8Config",
    "blockwise_int8": "BlockInt8Config",
    "modelopt": "ModelOptFp8Config",
    "modelopt_fp8": "ModelOptFp8Config",
    "modelopt_fp4": "ModelOptFp4Config",
    "modelopt_mixed": "ModelOptMixedPrecisionConfig",
    "w8a8_int8": "W8A8Int8Config",
    "w8a8_fp8": "W8A8Fp8Config",
    "awq": "AWQConfig",
    "awq_marlin": "AWQMarlinConfig",
    "bitsandbytes": "BitsAndBytesConfig",
    "gguf": "GGUFConfig",
    "gptq": "GPTQConfig",
    "gptq_marlin": "GPTQMarlinConfig",
    "moe_wna16": "MoeWNA16Config",
    "compressed-tensors": "CompressedTensorsConfig",
    "qoq": "QoQConfig",
    "w4afp8": "W4AFp8Config",
    "petit_nvfp4": "PetitNvFp4Config",
    "fbgemm_fp8": "FBGEMMFp8Config",
    "quark": "QuarkConfig",
    "auto-round": "AutoRoundConfig",
    "modelslim": "ModelSlimConfig",
    "quark_int4fp8_moe": "QuarkInt4Fp8Config",
}

if is_cuda() or (mxfp_supported() and is_hip()):
    _METHOD_TO_CLASS["mxfp4"] = "Mxfp4Config"


def _make_proxy(class_name: str) -> _LazyQuantConfigProxy:
    module_path, resolved_class_name = _CLASS_SPECS[class_name]
    return _LazyQuantConfigProxy(module_path, resolved_class_name)


def _resolve_entry(entry):
    if isinstance(entry, _LazyQuantConfigProxy):
        return entry._resolve()
    return entry


QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    name: _make_proxy(class_name) for name, class_name in _METHOD_TO_CLASS.items()
}

CPU_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": _make_proxy("Fp8Config"),
    "w8a8_int8": _make_proxy("W8A8Int8Config"),
    "compressed-tensors": _make_proxy("CompressedTensorsConfig"),
    "awq": _make_proxy("AWQCPUConfig"),
    "gptq": _make_proxy("CPUGPTQConfig"),
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    if is_cpu() and cpu_has_amx_support():
        if quantization not in CPU_QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization method on CPU: {quantization}. "
                f"Available methods on CPU: {list(QUANTIZATION_METHODS.keys())}"
            )
        return _resolve_entry(CPU_QUANTIZATION_METHODS[quantization])

    return _resolve_entry(QUANTIZATION_METHODS[quantization])


def __getattr__(name: str):
    if name == "QuantizationConfig":
        return QuantizationConfig
    if name in _CLASS_SPECS:
        return _make_proxy(name)._resolve()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CPU_QUANTIZATION_METHODS",
    "QUANTIZATION_METHODS",
    "QuantizationConfig",
    "get_quantization_config",
    *_CLASS_SPECS.keys(),
]
