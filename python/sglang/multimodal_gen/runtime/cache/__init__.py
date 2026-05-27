# SPDX-License-Identifier: Apache-2.0
"""
Cache acceleration module for SGLang-diffusion.

TeaCache is always available. cache-dit integration stays lazy so importing the
cache package does not require the optional `cache_dit` dependency unless those
APIs are actually used.
"""

from typing import Any

from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheContext, TeaCacheMixin

__all__ = [
    "TeaCacheContext",
    "TeaCacheMixin",
    "CacheDitConfig",
    "enable_cache_on_transformer",
    "enable_cache_on_dual_transformer",
    "get_scm_mask",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CacheDitConfig",
        "enable_cache_on_transformer",
        "enable_cache_on_dual_transformer",
        "get_scm_mask",
    }:
        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            CacheDitConfig,
            enable_cache_on_dual_transformer,
            enable_cache_on_transformer,
            get_scm_mask,
        )

        return {
            "CacheDitConfig": CacheDitConfig,
            "enable_cache_on_transformer": enable_cache_on_transformer,
            "enable_cache_on_dual_transformer": enable_cache_on_dual_transformer,
            "get_scm_mask": get_scm_mask,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
