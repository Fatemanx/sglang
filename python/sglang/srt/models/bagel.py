from sglang.srt.models.transformers import TransformersMultiModalForCausalLM
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper


class BagelForConditionalGeneration(TransformersMultiModalForCausalLM):
    """Understanding-only BAGEL support via the transformers backend."""

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=self.skip_prefixes,
            skip_substrs=self.skip_substrs,
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
        )

        def renamed_weights():
            for name, weight in weights:
                mapped_name = self.hf_to_sglang_mapper._map_name(name)
                if mapped_name is None:
                    continue
                yield mapped_name, weight

        return loader.load_weights(renamed_weights(), mapper=None)


BagelForConditionalGeneration.hf_to_sglang_mapper = (
    BagelForConditionalGeneration.hf_to_sglang_mapper
    | WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.language_model.model.",
            "language_model.lm_head.": "lm_head.",
            "vit_model.": "model.vit_model.",
            "connector.": "model.connector.",
            "time_embedder.": None,
            "vae2llm.": None,
            "llm2vae.": None,
            "latent_pos_embed.": None,
            "vit_pos_embed.": "model.vit_pos_embed.",
            "model.layers.": "model.language_model.model.layers.",
            "model.embed_tokens.": "model.language_model.model.embed_tokens.",
            "model.norm.": "model.language_model.model.norm.",
            "model.rotary_emb.": "model.language_model.model.rotary_emb.",
            "decoder.": None,
            "encoder.": None,
        }
    )
)


EntryClass = BagelForConditionalGeneration
