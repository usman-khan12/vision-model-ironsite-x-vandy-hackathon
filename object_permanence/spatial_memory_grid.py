"""
Spatial Memory Grid with Predictive Occupancy Maps.
Novel technique for maintaining object presence in spatial grid even when occluded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialMemoryGrid(nn.Module):
    """
    Maintains a 2D spatial grid that tracks object presence even when occluded.
    Each grid cell stores object features, IDs, and confidence scores.

    Grid dimensions:
        grid_state:      [batch, H, W, num_objects, feature_dim]
        grid_confidence:  [batch, H, W, num_objects]
        grid_temporal:    [batch, H, W, num_objects]
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (32, 32),
        feature_dim: int = 512,
        num_objects: int = 10,
        hidden_dim: int = 256,
        decay_factor: float = 0.95,
    ):
        super().__init__()
        self.grid_height, self.grid_width = grid_size
        self.feature_dim = feature_dim
        self.num_objects = num_objects
        self.decay_factor = decay_factor

        # Occupancy predictor: features + current_pos + future_pos + velocity
        self.occupancy_predictor = nn.Sequential(
            nn.Linear(feature_dim + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_height * self.grid_width),
            nn.Sigmoid(),
        )

        # Cross-reference matcher: obs_features + grid_features + position
        self.cross_reference = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects),
            nn.Softmax(dim=-1),
        )

        # Grid state (buffers, not parameters)
        self.grid_state: Optional[torch.Tensor] = None
        self.grid_confidence: Optional[torch.Tensor] = None
        self.grid_temporal: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Initialisation / reset
    # ------------------------------------------------------------------

    def initialize_grid(self, batch_size: int, device: torch.device):
        self.grid_state = torch.zeros(
            batch_size,
            self.grid_height,
            self.grid_width,
            self.num_objects,
            self.feature_dim,
            device=device,
        )
        self.grid_confidence = torch.zeros(
            batch_size,
            self.grid_height,
            self.grid_width,
            self.num_objects,
            device=device,
        )
        self.grid_temporal = torch.zeros(
            batch_size,
            self.grid_height,
            self.grid_width,
            self.num_objects,
            device=device,
        )

    def reset(self):
        self.grid_state = None
        self.grid_confidence = None
        self.grid_temporal = None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def position_to_grid(
        self,
        positions: torch.Tensor,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Convert (x, y) positions to integer (grid_w, grid_h) coordinates.

        Args:
            positions: [batch, num_objects, 2]  (x, y) in [0, 1] or pixels
            image_shape: (height, width) if positions are in pixels

        Returns:
            [batch, num_objects, 2] as long — column 0 = grid_w, column 1 = grid_h
        """
        pos = positions.clone().float()
        if image_shape is not None:
            h, w = image_shape
            pos[..., 0] /= w
            pos[..., 1] /= h

        out = pos.clone()
        out[..., 0] = pos[..., 0] * (self.grid_width - 1)
        out[..., 1] = pos[..., 1] * (self.grid_height - 1)
        out[..., 0].clamp_(0, self.grid_width - 1)
        out[..., 1].clamp_(0, self.grid_height - 1)
        return out.long()

    def _flat_index(self, grid_coords: torch.Tensor):
        """
        Convert grid coords [batch, num_objects, 2] (w, h) to flat indices
        into the [H, W] plane — useful for advanced indexing.

        Returns:
            grid_h: [batch, num_objects]   (long)
            grid_w: [batch, num_objects]   (long)
        """
        return grid_coords[..., 1], grid_coords[..., 0]

    # ------------------------------------------------------------------
    # update_grid  (vectorised — no Python loops over batch / objects)
    # ------------------------------------------------------------------

    def update_grid(
        self,
        object_features: torch.Tensor,
        positions: torch.Tensor,
        occlusion_factors: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None,
    ):
        """
        Update grid with current observations.

        Args:
            object_features:  [batch, num_objects, feature_dim]
            positions:         [batch, num_objects, 2]
            occlusion_factors: [batch, num_objects]   0 = visible, 1 = occluded
        """
        B, N, D = object_features.shape
        device = object_features.device

        if self.grid_state is None:
            self.initialize_grid(B, device)

        grid_coords = self.position_to_grid(positions)  # [B, N, 2]
        gh, gw = self._flat_index(grid_coords)  # each [B, N]

        # Build batch + object index tensors for scatter
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(gh)  # [B, N]
        o_idx = torch.arange(N, device=device).unsqueeze(0).expand_as(gh)  # [B, N]

        # Gather current grid values at the addressed cells
        cur_feat = self.grid_state[b_idx, gh, gw, o_idx, :]  # [B, N, D]
        cur_conf = self.grid_confidence[b_idx, gh, gw, o_idx]  # [B, N]

        # Per-element alpha depending on visibility
        visible = (occlusion_factors < 0.5).float()  # [B, N]
        alpha = visible * 0.8 + (1.0 - visible) * 0.3  # [B, N]
        alpha_f = alpha.unsqueeze(-1)  # [B, N, 1]

        new_feat = alpha_f * object_features + (1.0 - alpha_f) * cur_feat

        # Confidence: visible -> increase,  occluded -> decay
        new_conf = visible * (cur_conf * 0.9 + 0.5).clamp(max=1.0) + (1.0 - visible) * (
            cur_conf * self.decay_factor
        )

        # Temporal increment
        temporal_inc = visible * 1.0 + (1.0 - visible) * 0.5

        # Write back
        self.grid_state[b_idx, gh, gw, o_idx, :] = new_feat
        self.grid_confidence[b_idx, gh, gw, o_idx] = new_conf
        self.grid_temporal[b_idx, gh, gw, o_idx] += temporal_inc

        # Global confidence decay
        self.grid_confidence *= self.decay_factor

    # ------------------------------------------------------------------
    # predict_occupancy  (batched over objects via reshape)
    # ------------------------------------------------------------------

    def predict_occupancy(
        self,
        object_features: torch.Tensor,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        num_frames_ahead: int = 1,
    ) -> torch.Tensor:
        """
        Predict spatial occupancy maps for every object.

        Returns:
            [batch, num_objects, H, W]
        """
        B, N, D = object_features.shape
        if velocities is None:
            velocities = torch.zeros_like(positions)

        future_pos = positions + velocities * num_frames_ahead

        # [B, N, D+6]
        inp = torch.cat([object_features, positions, future_pos, velocities], dim=-1)
        # Flatten batch*objects -> single batch for the MLP
        flat = inp.reshape(B * N, -1)
        occ_flat = self.occupancy_predictor(flat)  # [B*N, H*W]
        return occ_flat.view(B, N, self.grid_height, self.grid_width)

    # ------------------------------------------------------------------
    # reidentify_object  (vectorised inner loop)
    # ------------------------------------------------------------------

    def reidentify_object(
        self,
        new_features: torch.Tensor,
        new_position: torch.Tensor,
        threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-identify objects by matching new observations to grid state.

        Returns:
            matched_ids:        [batch, num_objects]
            match_confidence:   [batch, num_objects]
        """
        B, N, D = new_features.shape
        device = new_features.device

        if self.grid_state is None:
            ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            return ids, torch.ones(B, N, device=device)

        grid_coords = self.position_to_grid(new_position)
        gh, gw = self._flat_index(grid_coords)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        o_all = torch.arange(self.num_objects, device=device)

        matched_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        match_conf = torch.zeros(B, N, device=device)

        for oi in range(N):
            # Grid features at each batch element's cell for ALL tracked objects
            # grid_state[b, gh[b,oi], gw[b,oi], :, :]  -> [B, num_objects, D]
            g_feats = self.grid_state[b_idx[:, oi], gh[:, oi], gw[:, oi], :, :]
            g_conf = self.grid_confidence[b_idx[:, oi], gh[:, oi], gw[:, oi], :]

            obs = new_features[:, oi, :].unsqueeze(1).expand(-1, self.num_objects, -1)
            pos = (
                grid_coords[:, oi, :]
                .unsqueeze(1)
                .expand(-1, self.num_objects, -1)
                .float()
            )

            match_in = torch.cat([obs, g_feats, pos], dim=-1)  # [B, num_obj, D*2+2]
            flat_in = match_in.reshape(B * self.num_objects, -1)
            match_out = self.cross_reference(flat_in)  # [B*num_obj, num_obj]
            match_out = match_out.view(B, self.num_objects, self.num_objects)

            diag = match_out[:, o_all, o_all]  # [B, num_objects]
            scores = diag * g_conf  # [B, num_objects]

            best = scores.argmax(dim=-1)  # [B]
            matched_ids[:, oi] = best
            match_conf[:, oi] = scores[torch.arange(B, device=device), best]

        return matched_ids, match_conf

    # ------------------------------------------------------------------
    # get_grid_features_at_position  (vectorised)
    # ------------------------------------------------------------------

    def get_grid_features_at_position(
        self,
        positions: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retrieve grid features at given positions.

        Returns:
            [batch, num_objects, feature_dim]
        """
        B, N, _ = positions.shape
        device = positions.device

        if self.grid_state is None:
            return torch.zeros(B, N, self.feature_dim, device=device)

        grid_coords = self.position_to_grid(positions)
        gh, gw = self._flat_index(grid_coords)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

        if object_ids is not None:
            oid = object_ids.long()
            return self.grid_state[b_idx, gh, gw, oid, :]  # [B, N, D]

        # No ids: average across all object slots at each cell
        all_feats = self.grid_state[b_idx, gh, gw, :, :]  # [B, N, num_obj, D]
        return all_feats.mean(dim=2)
