"""Preferred/fallback scheduling across query and head boundaries."""

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute import interface, utils


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("request_clc", [False, True])
@pytest.mark.parametrize("head_dim,kv_stage", [(128, 4), (256, 4), (256, 5), (256, 12)])
def test_mixed_cluster_nonpersistent(monkeypatch, dtype, request_clc, head_dim, kv_stage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("The preferred-cluster specialization requires SM103")
    if dtype == torch.bfloat16 and kv_stage > 5:
        pytest.skip("D256 BF16 cannot fit twelve KV stages in shared memory")
    monkeypatch.setattr(utils, "_fa_fwd_cluster_mode", "8cta+2cta/2cta")
    monkeypatch.setattr(utils, "_fa_clc_enabled", request_clc)
    monkeypatch.setattr(utils, "_fa_disable_2cta_enabled", False)
    monkeypatch.setattr(utils, "_fa_hd256_kv_stage", kv_stage)
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
    kernel_cls = (
        interface.FlashAttentionForwardSm100 if head_dim == 128
        else interface.BlackwellFusedMultiHeadAttentionForward
    )
    original_init = kernel_cls.__init__
    specializations = []

    def check_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        specializations.append(self)
        assert self.fwd_cluster_mode == "8cta+2cta/2cta"
        assert not self.use_clc_scheduler and not self.is_persistent

    monkeypatch.setattr(kernel_cls, "__init__", check_init)
    torch.manual_seed(1234)
    with torch.inference_mode():
        # Three and five 2CTA tasks per head: preferred clusters cross heads,
        # and the final task has an incomplete query tile.
        for seqlen in (1025, 2049):
            q, k, v = [
                torch.randn(1, seqlen, 24, head_dim, device="cuda", dtype=torch.bfloat16).to(dtype)
                for _ in range(3)
            ]
            ref = F.scaled_dot_product_attention(
                q.to(torch.bfloat16).transpose(1, 2),
                k.to(torch.bfloat16).transpose(1, 2),
                v.to(torch.bfloat16).transpose(1, 2),
            ).transpose(1, 2)
            for _ in range(3):
                out, lse, *_ = interface._flash_attn_fwd(q, k, v, return_lse=True, num_splits=1)
                torch.testing.assert_close(out, ref, rtol=0.05, atol=0.02 if dtype.itemsize == 1 else 0.002)
                assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    assert specializations, "The test must compile and check the selected kernel"


@pytest.mark.parametrize("head_dim,causal", [(64, False), (128, True), (192, False), (256, True)])
def test_preferred_cluster_preserves_other_shapes(monkeypatch, head_dim, causal):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires SM103")
    monkeypatch.setattr(utils, "_fa_fwd_cluster_mode", "8cta+2cta/2cta")
    monkeypatch.setattr(utils, "_fa_clc_enabled", False)
    monkeypatch.setattr(utils, "_fa_hd256_kv_stage", 4)
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
    with torch.inference_mode():
        q, k = [torch.randn(2, 513, 4, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        v = torch.randn(2, 513, 4, 128 if head_dim == 192 else head_dim, device="cuda", dtype=torch.bfloat16)
        out, *_ = interface._flash_attn_fwd(q, k, v, causal=causal, num_splits=1)
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
        ).transpose(1, 2)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.02)


def test_d256_kv_depth_cache_key(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires SM103")
    monkeypatch.setattr(utils, "_fa_fwd_cluster_mode", "8cta+2cta/2cta")
    monkeypatch.setattr(utils, "_fa_clc_enabled", False)
    cache = {}
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", cache)
    with torch.inference_mode():
        q, k, v = [torch.randn(1, 513, 24, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        for depth in (4, 5, 4):
            monkeypatch.setattr(utils, "_fa_hd256_kv_stage", depth)
            out, *_ = interface._flash_attn_fwd(q, k, v, num_splits=1)
            torch.testing.assert_close(out, ref, rtol=0.05, atol=0.02)
    assert len(cache) == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_d256_preferred_mqa(monkeypatch, dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires SM103")
    monkeypatch.setattr(utils, "_fa_fwd_cluster_mode", "8cta+2cta/2cta")
    monkeypatch.setattr(utils, "_fa_hd256_kv_stage", 5)
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", {})
    with torch.inference_mode():
        q = torch.randn(1, 513, 128, 256, device="cuda", dtype=torch.bfloat16).to(dtype)
        k, v = [torch.randn(1, 513, 1, 256, device="cuda", dtype=torch.bfloat16).to(dtype) for _ in range(2)]
        out, *_ = interface._flash_attn_fwd(q, k, v, num_splits=1)
        ref = F.scaled_dot_product_attention(
            q.to(torch.bfloat16).transpose(1, 2), k.to(torch.bfloat16).transpose(1, 2),
            v.to(torch.bfloat16).transpose(1, 2), enable_gqa=True,
        ).transpose(1, 2)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.02)
