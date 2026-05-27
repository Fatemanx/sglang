# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs import PipelineConfig
from sglang.multimodal_gen.configs.sample import SamplingParams

__all__ = ["DiffGenerator", "PipelineConfig", "SamplingParams"]


def __getattr__(name: str) -> Any:
    if name == "DiffGenerator":
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        return DiffGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
