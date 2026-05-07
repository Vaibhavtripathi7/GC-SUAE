"""
Unit tests for model forward passes and loss functions on synthetic data.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np

from models.architectures import (
    Unimodal2DCNN, Unimodal3DCNN, EarlyFusionAE, LateFusionAE, TRIAD, GC_SUAE,
    SpectralAngleMapperLoss, SpatialCrossAttention, build_model,
)
from losses.tagcl import TAGCLLoss, CombinedLoss, LMMConstraintLoss


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

    def test_triad(self):
        model = TRIAD(N_BANDS, LATENT_DIM, PATCH_SIZE)
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
