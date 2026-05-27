"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
import importlib.machinery
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

partial_json_parser = types.ModuleType("partial_json_parser")
partial_json_parser_core = types.ModuleType("partial_json_parser.core")
partial_json_parser_exceptions = types.ModuleType("partial_json_parser.core.exceptions")
partial_json_parser_options = types.ModuleType("partial_json_parser.core.options")
torchcodec = types.ModuleType("torchcodec")
torchcodec_decoders = types.ModuleType("torchcodec.decoders")
torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
torchcodec_decoders.__spec__ = importlib.machinery.ModuleSpec(
    "torchcodec.decoders", loader=None
)


class _MalformedJSON(Exception):
    pass


class _Allow:
    STR = 1
    OBJ = 2
    ARR = 4
    ALL = STR | OBJ | ARR


def _loads(input_str, _flags=None):
    return json.loads(input_str)


partial_json_parser_exceptions.MalformedJSON = _MalformedJSON
partial_json_parser_options.Allow = _Allow
partial_json_parser.loads = _loads
sys.modules.setdefault("partial_json_parser", partial_json_parser)
sys.modules.setdefault("partial_json_parser.core", partial_json_parser_core)
sys.modules.setdefault(
    "partial_json_parser.core.exceptions", partial_json_parser_exceptions
)
sys.modules.setdefault("partial_json_parser.core.options", partial_json_parser_options)
# Keep these unit tests isolated from optional torchcodec native libraries.
sys.modules.setdefault("torchcodec", torchcodec)
sys.modules.setdefault("torchcodec.decoders", torchcodec_decoders)
try:
    import flashinfer

    if not hasattr(flashinfer, "mm_mxfp8"):
        flashinfer.mm_mxfp8 = lambda *args, **kwargs: None
except Exception:
    pass

from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    _prepare_nvfp4_weight_bytes,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _filter_duplicate_precision_variant_safetensors,
    _Flux2Nvfp4FallbackAdapter,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.models.dits.flux import FluxSingleTransformerBlock
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
)
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _updated_quant_config,
)


class _FakeFluxTransformer:
    pass


class _FakeQuantConfig:
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"


class TestTransformerQuantHelpers(unittest.TestCase):
    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(param_names_mapping={})
                ),
            ),
            nunchaku_config=None,
            tp_size=1,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _write_safetensors_file(
        self,
        directory: str,
        filename: str,
        tensors: dict[str, torch.Tensor],
        *,
        metadata: dict[str, str] | None = None,
    ) -> str:
        path = f"{directory}/{filename}"
        save_file(tensors, path, metadata=metadata)
        return path

    def test_resolve_transformer_safetensors_to_load_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [f.name])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_safetensors_to_load_prefers_mixed_export(
        self, _mock_download
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed = f"{tmpdir}/flux2-dev-nvfp4-mixed.safetensors"
            full = f"{tmpdir}/flux2-dev-nvfp4.safetensors"
            open(mixed, "a").close()
            open(full, "a").close()

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [mixed])

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(
            resolved,
            [
                "/tmp/transformer/diffusion_pytorch_model.safetensors",
                "/tmp/transformer/other.safetensors",
            ],
        )

    def test_filter_transformer_precision_variants_keeps_precision_only_family(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.bf16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(resolved, files)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16
        ),
    )
    def test_resolve_transformer_quant_load_spec_prefers_component_config_over_weights_metadata(
        self,
        mock_quant_metadata,
        _mock_nvfp4,
    ):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/override-transformer.safetensors"
        )

        spec = resolve_transformer_quant_load_spec(
            hf_config={
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": ["proj_out"],
                }
            },
            server_args=server_args,
            safetensors_list=["/tmp/override-transformer.safetensors"],
            component_model_path="/unused/component/path",
            model_cls=_FakeFluxTransformer,
            cls_name=_FakeFluxTransformer.__name__,
        )

        self.assertEqual(type(spec.quant_config).get_name(), "modelopt_fp8")
        self.assertEqual(spec.quant_config.exclude_modules, ["proj_out"])
        mock_quant_metadata.assert_not_called()

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_quant_load_spec_reads_override_directory_config(
        self,
        _mock_download,
        _mock_nvfp4,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "modelopt",
                            "quant_algo": "FP8",
                            "ignore": ["proj_out"],
                        }
                    },
                    f,
                )

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            spec = resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=[f"{tmpdir}/unused.safetensors"],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
            )

        self.assertEqual(type(spec.quant_config).get_name(), "modelopt_fp8")
        self.assertEqual(spec.quant_config.exclude_modules, ["proj_out"])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["fresh.module"],
            checkpoint_uses_packed_qkv=True,
        ),
    )
    def test_resolve_transformer_quant_load_spec_prefers_safetensors_inferred_nvfp4_layout(
        self,
        _mock_nvfp4,
    ):
        spec = resolve_transformer_quant_load_spec(
            hf_config={
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "ignore": ["stale.module"],
                    "swap_weight_nibbles": False,
                }
            },
            server_args=self._make_server_args(),
            safetensors_list=["/tmp/quantized-transformer.safetensors"],
            component_model_path="/unused/component/path",
            model_cls=_FakeFluxTransformer,
            cls_name=_FakeFluxTransformer.__name__,
        )

        self.assertEqual(type(spec.quant_config).get_name(), "modelopt_fp4")
        self.assertEqual(spec.quant_config.exclude_modules, ["fresh.module"])
        self.assertTrue(spec.quant_config.checkpoint_uses_packed_qkv)
        self.assertFalse(spec.quant_config.swap_weight_nibbles)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        _mock_download,
        mock_metadata,
        _mock_quant_metadata,
        mock_maybe_download,
        _mock_nvfp4,
    ):
        mock_maybe_download.side_effect = AssertionError(
            "local safetensors path should not trigger maybe_download_model"
        )
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            nunchaku_config = NunchakuConfig(transformer_weights_path=f.name)
            server_args = self._make_server_args(
                transformer_weights_path=nunchaku_config.transformer_weights_path,
                nunchaku_config=nunchaku_config,
            )

            spec = resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=[nunchaku_config.transformer_weights_path],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
            )

        self.assertIsNone(spec.quant_config)
        self.assertIs(spec.nunchaku_config, nunchaku_config)
        self.assertIsNone(spec.param_dtype)
        self.assertEqual(len(spec.post_load_hooks), 1)
        self.assertIs(nunchaku_config.model_cls, _FakeFluxTransformer)
        mock_maybe_download.assert_not_called()

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads(self):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4-mixed.safetensors",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads_for_directory_override(
        self,
    ):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
            safetensors_list=["/tmp/flux2-dev-nvfp4/flux2-dev-nvfp4-mixed.safetensors"],
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)

    def test_prepare_nvfp4_weight_bytes_swaps_nibbles(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=True)

        self.assertEqual(prepared.tolist(), [[0xBA, 0x01]])

    def test_prepare_nvfp4_weight_bytes_can_skip_nibble_swap(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=False)

        self.assertEqual(prepared.tolist(), [[0xAB, 0x10]])

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_flat_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "swap_weight_nibbles": False,
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_nested_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "swap_weight_nibbles": False,
                },
                "config_groups": {"default": {"weights": {"group_size": 16}}},
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_accepts_fp4_quant_algo_alias(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "FP4",
                "group_size": 16,
                "ignore": [],
            }
        )

        self.assertEqual(config.group_size, 16)
        self.assertEqual(config.exclude_modules, [])

    def test_build_nvfp4_config_from_safetensors_list_aggregates_fallback_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            quantized = self._write_safetensors_file(
                tmpdir,
                "quantized.safetensors",
                {
                    "transformer.layers.0.weight": torch.zeros(
                        (2, 4), dtype=torch.uint8
                    ),
                    "transformer.layers.0.weight_scale": torch.ones(
                        (2, 2), dtype=torch.float32
                    ),
                },
            )
            fallback = self._write_safetensors_file(
                tmpdir,
                "fallback.safetensors",
                {
                    "transformer.layers.1.weight": torch.ones(
                        (4, 8), dtype=torch.bfloat16
                    )
                },
            )

            config = build_nvfp4_config_from_safetensors_list([quantized, fallback])

        self.assertIsNotNone(config)
        self.assertEqual(type(config).get_name(), "modelopt_fp4")
        self.assertEqual(config.group_size, 4)
        self.assertEqual(config.exclude_modules, ["transformer.layers.1"])
        self.assertFalse(config.checkpoint_uses_packed_qkv)

    def test_build_nvfp4_config_from_safetensors_list_uses_fallback_group_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = self._write_safetensors_file(
                tmpdir,
                "quantized.safetensors",
                {
                    "transformer.layers.0.weight": torch.zeros(
                        (2, 16), dtype=torch.uint8
                    ),
                    "transformer.layers.0.weight_scale": torch.ones(
                        (2, 3), dtype=torch.float32
                    ),
                },
            )

            config = build_nvfp4_config_from_safetensors_list(
                [shard], fallback_group_size=16
            )

        self.assertIsNotNone(config)
        self.assertEqual(type(config).get_name(), "modelopt_fp4")
        self.assertEqual(config.group_size, 16)
        self.assertEqual(config.exclude_modules, [])

    def test_builder_adds_diffusers_quant_type_for_nvfp4(self):
        updated = _updated_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                }
            },
            fallback_patterns=["single_transformer_blocks.*.proj_mlp*"],
            swap_weight_nibbles=False,
        )

        self.assertEqual(updated["quantization_config"]["quant_type"], "NVFP4")
        self.assertEqual(
            updated["quantization_config"]["ignore"],
            ["single_transformer_blocks.*.proj_mlp*"],
        )

    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    def test_flux_single_transformer_block_modelopt_excludes_use_full_prefix(
        self,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
    ):
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "single_transformer_blocks.*.proj_mlp*",
                "single_transformer_blocks.*.proj_out*",
                "single_transformer_blocks.*.attn.to_q",
            ],
        )

        block = FluxSingleTransformerBlock(
            dim=64,
            num_attention_heads=4,
            attention_head_dim=16,
            mlp_ratio=2.0,
            quant_config=quant_config,
            prefix="single_transformer_blocks.0",
        )

        self.assertEqual(block.proj_mlp.prefix, "single_transformer_blocks.0.proj_mlp")
        self.assertEqual(block.proj_out.prefix, "single_transformer_blocks.0.proj_out")
        self.assertEqual(
            block.attn.to_q.prefix, "single_transformer_blocks.0.attn.to_q"
        )
        self.assertIsInstance(block.proj_mlp.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.proj_out.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
