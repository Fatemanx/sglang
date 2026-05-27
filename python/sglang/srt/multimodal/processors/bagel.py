from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.bagel import BagelForConditionalGeneration
from sglang.srt.multimodal.processors.transformers_auto import (
    TransformersAutoMultimodalProcessor,
)


class BagelMultimodalProcessor(TransformersAutoMultimodalProcessor):
    supports_transformers_backend = True
    models = [BagelForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        if self.mm_tokens.image_token_id is None and hasattr(
            _processor, "image_token_id"
        ):
            self.mm_tokens.image_token_id = _processor.image_token_id
        if self.mm_tokens.image_token is None and hasattr(_processor, "image_token"):
            self.mm_tokens.image_token = _processor.image_token
        self.mm_tokens.build(_processor)

    def _build_mm_items(self, processor_output, input_ids):
        items = super()._build_mm_items(processor_output, input_ids)
        start_id = getattr(self.hf_config, "vision_start_token_id", None)
        end_id = getattr(self.hf_config, "vision_end_token_id", None)
        if start_id is None or end_id is None:
            return items

        offsets = self.get_mm_items_offset_by_pair(input_ids, start_id, end_id)
        if not offsets:
            return items

        for item in items:
            if item.modality in (Modality.IMAGE, Modality.MULTI_IMAGES):
                item.offsets = offsets
        return items
