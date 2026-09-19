"""
Unit tests for model forward passes and loss functions on synthetic data.
Run: pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from losses.tagcl import CombinedLoss, LMMConstraintLoss, TAGCLLoss
from models.architectures import (
    GC_SUAE,
    PooledAttnFusion,
    EarlyFusionAE,
    LateFusionAE,
    SpatialCrossAttention,
    SpectralAngleMapperLoss,
    Unimodal2DCNN,
    Unimodal3DCNN,
    build_model,
)

# Fixtures

N_BANDS    = 86
PATCH_SIZE = 32   # small for test speed
BATCH_SIZE = 2
LATENT_DIM = 32
N_MINERALS = 6


def make_batch(B=BATCH_SIZE, n_bands=N_BANDS, ps=PATCH_SIZE) -> dict:
    return {
        "iirs":       torch.rand(B, n_bands, ps, ps),
        "dem":        torch.rand(B, 1, ps, ps),
        "feo":        torch.rand(B, 1, ps, ps),
        "slope":      torch.rand(B, 1, ps, ps),
        "aspect":     torch.rand(B, 1, ps, ps),
        "feo_mean":   torch.rand(B),
        "slope_mean": torch.rand(B),
        "patch_idx":  torch.arange(B),
    }


# Model shape tests

class TestBaselineModels:
    """Test that all ablation baselines produce correct output shapes."""

    def _test_model(self, model):
        batch = make_batch()
        recon, z = model(batch)
        assert recon.shape == (BATCH_SIZE, N_BANDS, PATCH_SIZE, PATCH_SIZE), \
            f"Reconstruction shape mismatch: {recon.shape}"
        assert z.shape == (BATCH_SIZE, LATENT_DIM), \
            f"Latent shape mismatch: {z.shape}"
        assert not torch.isnan(recon).any(), "NaN in reconstruction"
        assert not torch.isnan(z).any(), "NaN in latent"

    def test_unimodal_2d(self):
        model = Unimodal2DCNN(N_BANDS, LATENT_DIM, PATCH_SIZE)
        self._test_model(model)

    def test_unimodal_3d(self):
        model = Unimodal3DCNN(N_BANDS, LATENT_DIM, PATCH_SIZE)
        self._test_model(model)

    def test_early_fusion(self):
        model = EarlyFusionAE(N_BANDS, LATENT_DIM, PATCH_SIZE)
        self._test_model(model)

    def test_late_fusion(self):
        model = LateFusionAE(N_BANDS, LATENT_DIM, PATCH_SIZE)
        self._test_model(model)

    def test_pooled_attn_fusion(self):
        model = PooledAttnFusion(N_BANDS, LATENT_DIM, PATCH_SIZE)
        self._test_model(model)


class TestGCSUAE:
    """Test GC-SUAE specific architecture."""

    def test_forward_shapes(self):
        model = GC_SUAE(N_BANDS, LATENT_DIM, N_MINERALS, PATCH_SIZE, d_model=64, n_heads=4)
        batch = make_batch()
        output = model(batch)

        assert "recon_iirs"       in output
        assert "abundances"       in output
        assert "endmember_matrix" in output
        assert "recon_feo"        in output
        assert "recon_dem"        in output
        assert "latent"           in output

        assert output["recon_iirs"].shape        == (BATCH_SIZE, N_BANDS, PATCH_SIZE, PATCH_SIZE)
        assert output["abundances"].shape        == (BATCH_SIZE, N_MINERALS, PATCH_SIZE, PATCH_SIZE)
        assert output["endmember_matrix"].shape  == (N_MINERALS, N_BANDS)
        assert output["recon_feo"].shape         == (BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
        assert output["recon_dem"].shape         == (BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE)
        assert output["latent"].shape            == (BATCH_SIZE, LATENT_DIM)

    def test_abundance_constraints(self):
        """Test that LMM decoder output satisfies ANC and ASC."""
        model = GC_SUAE(N_BANDS, LATENT_DIM, N_MINERALS, PATCH_SIZE, d_model=64, n_heads=4)
        batch = make_batch()
        output = model(batch)
        abundances = output["abundances"]

        # ANC: all values >= 0 (softplus guarantees this)
        assert (abundances >= 0).all(), "Abundance non-negativity violated"

        # ASC: sum over minerals ≈ 1.0 at each pixel
        abundance_sum = abundances.sum(dim=1)  # (B, H, W)
        assert torch.allclose(abundance_sum, torch.ones_like(abundance_sum), atol=1e-4), \
            f"ASC violated; max deviation: {(abundance_sum - 1).abs().max():.6f}"

        # endmember_matrix must be present and bounded
        E = output["endmember_matrix"]
        assert E.shape == (N_MINERALS, N_BANDS)
        assert (E >= 0).all() and (E <= 1).all(), "Endmember matrix out of [0,1]"

    def test_lmm_output_range(self):
        """LMM reconstruction must lie in [0,1] without explicit sigmoid."""
        model = GC_SUAE(N_BANDS, LATENT_DIM, N_MINERALS, PATCH_SIZE, d_model=64, n_heads=4)
        batch = make_batch()
        output = model(batch)
        recon = output["recon_iirs"]
        assert recon.min() >= -1e-5, "Reconstruction below 0"
        assert recon.max() <= 1 + 1e-5, "Reconstruction above 1"

    def test_no_nan(self):
        model = GC_SUAE(N_BANDS, LATENT_DIM, N_MINERALS, PATCH_SIZE, d_model=64, n_heads=4)
        batch = make_batch()
        output = model(batch)
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                assert not torch.isnan(val).any(), f"NaN in output['{key}']"
                assert not torch.isinf(val).any(), f"Inf in output['{key}']"

    def test_encode_method(self):
        model = GC_SUAE(N_BANDS, LATENT_DIM, N_MINERALS, PATCH_SIZE, d_model=64, n_heads=4)
        batch = make_batch()
        z, f_fused = model.encode(batch)
        assert z.shape[0] == BATCH_SIZE
        assert z.shape[1] == LATENT_DIM
        assert f_fused.dim() == 4  # (B, C, h, w)


# Loss function tests

class TestLosses:

    def test_sam_loss(self):
        sam = SpectralAngleMapperLoss()
        pred   = torch.rand(BATCH_SIZE, N_BANDS, PATCH_SIZE, PATCH_SIZE)
        target = torch.rand(BATCH_SIZE, N_BANDS, PATCH_SIZE, PATCH_SIZE)
        loss = sam(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Perfect reconstruction = SAM near 0 (small numerical error from acos is expected)
        perfect_loss = sam(target, target)
        assert perfect_loss.item() < 1e-3

    def test_sam_invariant_to_scale(self):
        """SAM should be scale-invariant."""
        sam = SpectralAngleMapperLoss()
        pred   = torch.rand(2, N_BANDS, 4, 4)
        target = pred * 2.5  # scaled version
        loss = sam(pred, target)
        assert loss.item() < 1e-3, "SAM should be ~0 for scaled-same spectra"

    def test_tagcl_loss(self):
        tagcl = TAGCLLoss(
            feo_threshold=0.1, slope_threshold=0.2,
            temperature=0.07, queue_size=32, d_model=LATENT_DIM
        )
        z_q = torch.randn(BATCH_SIZE, LATENT_DIM)
        z_k = torch.randn(BATCH_SIZE, LATENT_DIM)
        feo   = torch.rand(BATCH_SIZE) * 0.05   # close → positives
        slope = torch.rand(BATCH_SIZE) * 0.1
        loss = tagcl(z_q, z_k, feo, slope)
        assert not torch.isnan(loss), "TAGCL loss is NaN"
        assert loss.item() >= 0

    def test_lmm_loss(self):
        lmm = LMMConstraintLoss()
        abundances = torch.rand(BATCH_SIZE, N_MINERALS, PATCH_SIZE, PATCH_SIZE)
        # Normalize to sum-to-one
        abundances = abundances / abundances.sum(dim=1, keepdim=True)
        endmembers = torch.rand(N_MINERALS, N_BANDS)
        target = torch.rand(BATCH_SIZE, N_BANDS, PATCH_SIZE, PATCH_SIZE)
        loss = lmm(abundances, endmembers, target)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_combined_loss(self):
        model = GC_SUAE(N_BANDS, LATENT_DIM, N_MINERALS, PATCH_SIZE, d_model=64, n_heads=4)
        batch = make_batch()
        output = model(batch)

        combined = CombinedLoss(
            lambda_sam=0.1, lambda_tagcl=0.0,
            lambda_dem=0.05, lambda_feo=0.1, lambda_lmm=0.2,
            tagcl_cfg={"d_model": LATENT_DIM, "queue_size": 32},
        )
        total, components = combined(output, batch, z_momentum=None)
        assert not torch.isnan(total)
        assert total.item() >= 0
        assert "loss_total" in components
        assert "loss_mse"   in components
        assert "loss_sam"   in components


# Build factory tests

class TestSpatialCrossAttention:
    def test_output_shape(self):
        attn = SpatialCrossAttention(d_model=64, n_heads=4)
        q = torch.rand(2, 64, 8, 8)
        ctx = torch.rand(2, 64, 8, 8)
        out = attn(q, ctx)
        assert out.shape == q.shape, f"Shape mismatch: {out.shape}"

    def test_residual_preserves_info(self):
        """Output should not collapse to zero: residual connects query to output."""
        torch.manual_seed(0)
        attn = SpatialCrossAttention(d_model=64, n_heads=4)
        # Use random (not all-ones) query so that learned projections produce non-zero output
        q = torch.randn(2, 64, 4, 4)
        ctx = torch.randn(2, 64, 4, 4)
        out = attn(q, ctx)
        # After training initialisation, GroupNorm weight=1 bias=0, so output = norm(proj(attn)+q)
        # The residual q is non-zero, so output must be non-zero
        assert not torch.allclose(out, torch.zeros_like(out)), "Output is exactly zero"
    def test_all_models_build(self):
        cfg = {"n_bands": N_BANDS, "latent_dim": LATENT_DIM, "patch_size": PATCH_SIZE, "n_minerals": N_MINERALS, "d_model": 64, "n_heads": 4}
        for name in ["Unimodal2DCNN", "Unimodal3DCNN", "EarlyFusion", "LateFusion", "PooledAttnFusion", "GC_SUAE"]:
            model = build_model(name, cfg)
            assert model is not None, f"Failed to build {name}"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            build_model("UnknownModel", {})
