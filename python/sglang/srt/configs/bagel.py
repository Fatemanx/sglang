from typing import Any, Optional

from transformers import (
    BatchFeature,
    PretrainedConfig,
    ProcessorMixin,
    Qwen2TokenizerFast,
    SiglipImageProcessor,
)

from sglang.srt.configs.utils import register_image_processor, register_processor


def _to_config(config: Optional[Any]) -> PretrainedConfig:
    if isinstance(config, PretrainedConfig):
        return config
    if config is None:
        return PretrainedConfig()
    return PretrainedConfig(**config)


class BagelConfig(PretrainedConfig):
    model_type = "bagel"

    def __init__(
        self,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        image_token_id: int = 151655,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        architectures=None,
        visual_und: bool = True,
        visual_gen: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.llm_config = _to_config(llm_config)
        self.text_config = self.llm_config
        self.vit_config = _to_config(vit_config)
        self.vision_config = self.vit_config
        self.vae_config = _to_config(vae_config)

        normalized_architectures = architectures or ["BagelForConditionalGeneration"]
        normalized_architectures = [
            "BagelForConditionalGeneration" if arch == "Bagel" else arch
            for arch in normalized_architectures
        ]

        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.visual_und = visual_und
        self.visual_gen = visual_gen

        super().__init__(
            architectures=normalized_architectures,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class BagelImageProcessor(SiglipImageProcessor):
    model_input_names = ["pixel_values"]


class BagelProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BagelImageProcessor,
        tokenizer: Qwen2TokenizerFast,
        image_token: str = "<|vision_start|><|image_pad|><|vision_end|>",
        image_start_token: str = "<|vision_start|>",
        image_pad_token: str = "<|image_pad|>",
        image_end_token: str = "<|vision_end|>",
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_pad_token = image_pad_token
        self.image_end_token = image_end_token

        super().__init__(image_processor, tokenizer, **kwargs)

    @property
    def model_input_names(self):
        names = list(self.tokenizer.model_input_names)
        for name in getattr(self.image_processor, "model_input_names", []):
            if name not in names:
                names.append(name)
        return names

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_pad_token)

    @property
    def image_start_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_start_token)

    @property
    def image_end_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_end_token)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("BagelProcessor requires at least one of text or images.")

        processor_outputs = {}
        if text is not None:
            processor_outputs.update(
                self.tokenizer(text=text, return_tensors=return_tensors, **kwargs)
            )
        if images is not None:
            processor_outputs.update(
                self.image_processor(
                    images=images, return_tensors=return_tensors, **kwargs
                )
            )

        return BatchFeature(data=processor_outputs, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


register_processor(BagelConfig, BagelProcessor)
register_image_processor(BagelConfig, BagelImageProcessor)
