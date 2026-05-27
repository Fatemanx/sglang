# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

Keep stage exports lazy so importing a single pipeline path does not eagerly
pull optional 3D/video dependencies for unrelated models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_STAGE_EXPORTS = {
    "PipelineStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base",
        "PipelineStage",
    ),
    "CausalDMDDenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising",
        "CausalDMDDenoisingStage",
    ),
    "ComfyUILatentPreparationStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation",
        "ComfyUILatentPreparationStage",
    ),
    "DecodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding",
        "DecodingStage",
    ),
    "LTX2AVDecodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av",
        "LTX2AVDecodingStage",
    ),
    "DenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising",
        "DenoisingStage",
    ),
    "LTX2AVDenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av",
        "LTX2AVDenoisingStage",
    ),
    "LTX2RefinementStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av",
        "LTX2RefinementStage",
    ),
    "DmdDenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd",
        "DmdDenoisingStage",
    ),
    "EncodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.encoding",
        "EncodingStage",
    ),
    "Hunyuan3DPaintPostprocessStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_paint",
        "Hunyuan3DPaintPostprocessStage",
    ),
    "Hunyuan3DPaintPreprocessStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_paint",
        "Hunyuan3DPaintPreprocessStage",
    ),
    "Hunyuan3DPaintTexGenStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_paint",
        "Hunyuan3DPaintTexGenStage",
    ),
    "Hunyuan3DShapeBeforeDenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape",
        "Hunyuan3DShapeBeforeDenoisingStage",
    ),
    "Hunyuan3DShapeDenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape",
        "Hunyuan3DShapeDenoisingStage",
    ),
    "Hunyuan3DShapeExportStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape",
        "Hunyuan3DShapeExportStage",
    ),
    "Hunyuan3DShapeSaveStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape",
        "Hunyuan3DShapeSaveStage",
    ),
    "ImageEncodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
        "ImageEncodingStage",
    ),
    "ImageVAEEncodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
        "ImageVAEEncodingStage",
    ),
    "LTX2ImageEncodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
        "LTX2ImageEncodingStage",
    ),
    "InputValidationStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation",
        "InputValidationStage",
    ),
    "LatentPreparationStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation",
        "LatentPreparationStage",
    ),
    "LTX2AVLatentPreparationStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation_av",
        "LTX2AVLatentPreparationStage",
    ),
    "LTX2DenoisingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising",
        "LTX2DenoisingStage",
    ),
    "LTX2TextConnectorStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.text_connector",
        "LTX2TextConnectorStage",
    ),
    "TextEncodingStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding",
        "TextEncodingStage",
    ),
    "TimestepPreparationStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation",
        "TimestepPreparationStage",
    ),
    "LTX2HalveResolutionStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.upsampling",
        "LTX2HalveResolutionStage",
    ),
    "LTX2LoRASwitchStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.upsampling",
        "LTX2LoRASwitchStage",
    ),
    "LTX2UpsampleStage": (
        "sglang.multimodal_gen.runtime.pipelines_core.stages.upsampling",
        "LTX2UpsampleStage",
    ),
}

__all__ = list(_STAGE_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _STAGE_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
