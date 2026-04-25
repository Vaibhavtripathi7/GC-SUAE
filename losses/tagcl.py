"""
Terrain-Aware Geochemical Contrastive Loss (TAGCL).

Positive pairs are defined by geochemical + topographic proximity (similar FeO
abundance and terrain slope) instead of augmentation, regularizing the latent
space toward mineralogical structure without labels. Uses a MoCo-style momentum
queue with an InfoNCE supervised-contrastive objective; lambda is annealed from
0 to its max so the autoencoder stabilizes first.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TAGCLLoss(nn.Module):
    """
    Terrain-Aware Geochemical Contrastive Loss.

    Args:
        feo_threshold:   FeO proximity threshold for positive pair construction
                         (in normalized [0,1] units; ~0.05 ≈ 5% FeO wt%)
        slope_threshold: Slope proximity threshold (normalized units; ~0.1)
        temperature:     InfoNCE temperature (default 0.07, standard for MoCo)
        queue_size:      Number of negative samples in the memory queue
        d_model:         Dimension of latent representations
        momentum:        Momentum encoder update rate (default 0.999)
    """

    def __init__(
        self,
        feo_threshold: float = 0.05,
        slope_threshold: float = 0.1,
        temperature: float = 0.07,
        queue_size: int = 4096,
        d_model: int = 128,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.feo_threshold   = feo_threshold
        self.slope_threshold = slope_threshold
        self.T               = temperature
        self.queue_size      = queue_size
        self.momentum        = momentum
        self.d_model         = d_model

        # Memory queue (negative bank)
        self.register_buffer("queue",       F.normalize(torch.randn(queue_size, d_model), dim=1))
        self.register_buffer("queue_feo",   torch.zeros(queue_size))
        self.register_buffer("queue_slope", torch.zeros(queue_size))
        self.register_buffer("queue_ptr",   torch.zeros(1, dtype=torch.long))
        # Track how many slots are actually filled (for warmup correctness)
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        keys: torch.Tensor,
        feo_means: torch.Tensor,
        slope_means: torch.Tensor,
    ):
        """Update the memory queue with the current batch."""
        B = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Circular buffer - overwrite oldest entries
        end = min(ptr + B, self.queue_size)
        n   = end - ptr
        self.queue[ptr:end]       = keys[:n]
        self.queue_feo[ptr:end]   = feo_means[:n]
        self.queue_slope[ptr:end] = slope_means[:n]

        if n < B:
            # Wrap around
            remainder = B - n
            self.queue[:remainder]       = keys[n:]
            self.queue_feo[:remainder]   = feo_means[n:]
            self.queue_slope[:remainder] = slope_means[n:]
            self.queue_ptr[0] = remainder
        else:
            self.queue_ptr[0] = end % self.queue_size

        # Track actual filled size (saturates at queue_size)
        self.queue_filled[0] = min(
            int(self.queue_filled) + B, self.queue_size
        )

    def _build_pair_mask(
        self,
        feo_q: torch.Tensor,   # (B,) query FeO means
        slope_q: torch.Tensor, # (B,) query slope means
        feo_k: torch.Tensor,   # (N,) key FeO means (queue + current batch)
        slope_k: torch.Tensor, # (N,) key slope means
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs positive and negative masks based on physical proximity.

        A pair (i, j) is POSITIVE if:
            |FeO_i - FeO_j| < feo_threshold  AND
            |slope_i - slope_j| < slope_threshold

        Returns:
            pos_mask: (B, N) bool - True where pair is positive
            neg_mask: (B, N) bool - True where pair is negative
        """
        feo_dist   = (feo_q.unsqueeze(1) - feo_k.unsqueeze(0)).abs()    # (B, N)
        slope_dist = (slope_q.unsqueeze(1) - slope_k.unsqueeze(0)).abs() # (B, N)

        pos_mask = (feo_dist < self.feo_threshold) & (slope_dist < self.slope_threshold)
        neg_mask = (feo_dist >= self.feo_threshold) | (slope_dist >= self.slope_threshold)

        return pos_mask, neg_mask

    def forward(
        self,
        z_query: torch.Tensor,      # (B, d_model) - current batch latents
        z_key: torch.Tensor,        # (B, d_model) - momentum encoder latents
        feo_means: torch.Tensor,    # (B,) - batch FeO means
        slope_means: torch.Tensor,  # (B,) - batch slope means
    ) -> torch.Tensor:
        """
        Computes TAGCL loss.

        Uses supervised contrastive formulation: for each query, positives are
        geochemically similar patches in the queue; negatives are dissimilar patches.
        """
        B = z_query.shape[0]

        q = F.normalize(z_query, dim=1)            # (B, d)
        k = F.normalize(z_key, dim=1).detach()     # (B, d)

        # Use only filled portion of queue (avoids phantom negatives during warmup)
        q_filled = int(self.queue_filled)
        queue_k  = self.queue[:q_filled].clone().detach()      # (Q_actual, d)
        queue_f  = self.queue_feo[:q_filled].clone()
        queue_s  = self.queue_slope[:q_filled].clone()

        # All keys = current batch keys + filled queue
        all_keys   = torch.cat([k, queue_k], dim=0)            # (B+Q_actual, d)
        all_feo    = torch.cat([feo_means.detach(), queue_f], dim=0)
        all_slopes = torch.cat([slope_means.detach(), queue_s], dim=0)

        N = all_keys.shape[0]  # B + Q_actual (varies during warmup)

        sim = torch.mm(q, all_keys.T) / self.T     # (B, N)

        # Self-mask: exclude query_i vs key_i (within current batch only)
        self_mask = torch.zeros(B, N, dtype=torch.bool, device=q.device)
        self_mask[:, :B] = torch.eye(B, dtype=torch.bool, device=q.device)

        # Build pair masks (applied AFTER self-mask to avoid false positives)
        pos_mask, _ = self._build_pair_mask(
            feo_means, slope_means, all_feo, all_slopes
        )                                                       # (B, N)
        pos_mask = pos_mask & ~self_mask

        # Only compute loss for queries that have at least one valid positive
        has_positive = pos_mask.any(dim=1)                      # (B,)
        if not has_positive.any():
            self._dequeue_and_enqueue(k, feo_means, slope_means)
            return torch.tensor(0.0, device=z_query.device, requires_grad=True)

        # Numerator: log-sum-exp over positives
        sim_pos = sim.masked_fill(~pos_mask, float("-inf"))
        log_num = torch.logsumexp(sim_pos, dim=1)               # (B,)

        # Denominator: log-sum-exp over all non-self keys
        sim_all = sim.masked_fill(self_mask, float("-inf"))
        log_den = torch.logsumexp(sim_all, dim=1)               # (B,)

        # InfoNCE per-sample loss; clamp to avoid -inf - (-inf) = nan
        loss_per = -(log_num - log_den).clamp(min=-100.0)
        loss     = loss_per[has_positive].mean()

        self._dequeue_and_enqueue(k, feo_means, slope_means)
        return loss
