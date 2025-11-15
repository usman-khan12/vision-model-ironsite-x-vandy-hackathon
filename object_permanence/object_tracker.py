"""
Object Tracker Module
Maintains temporal memory and tracks objects across frames
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ObjectTracker:
    """
    Maintains state for tracked objects across frames.
    Stores previous features, positions, velocities, and occlusion history.
    """
    
    def __init__(self, max_objects=50, feature_dim=512):
        """
        Args:
            max_objects: Maximum number of objects to track
            feature_dim: Dimension of object features
        """
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        
        # State storage
        self.previous_features: Optional[torch.Tensor] = None
        self.previous_positions: Optional[torch.Tensor] = None
        self.velocities: Optional[torch.Tensor] = None
        self.occlusion_history: Optional[torch.Tensor] = None
        self.object_ids: Optional[torch.Tensor] = None
        
    def initialize(self, batch_size, num_objects, device):
        """Initialize tracker state for a new sequence."""
        self.previous_features = torch.zeros(
            batch_size, num_objects, self.feature_dim, device=device
        )
        self.previous_positions = torch.zeros(
            batch_size, num_objects, 2, device=device  # [x, y] positions
        )
        self.velocities = torch.zeros(
            batch_size, num_objects, self.feature_dim, device=device
        )
        self.occlusion_history = torch.zeros(
            batch_size, num_objects, device=device
        )
        self.object_ids = torch.arange(num_objects, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    def update(self, fused_features, positions=None, occlusion_factors=None):
        """
        Update tracker state with new fused features.
        
        Args:
            fused_features: [batch, num_objects, feature_dim] - Fused features
            positions: [batch, num_objects, 2] - Object positions (optional)
            occlusion_factors: [batch, num_objects] - Occlusion factors (optional)
        """
        batch_size, num_objects, feature_dim = fused_features.shape
        
        # Update velocities
        if self.previous_features is not None:
            self.velocities = fused_features - self.previous_features
        else:
            self.velocities = torch.zeros_like(fused_features)
        
        # Update positions if provided
        if positions is not None:
            if self.previous_positions is not None:
                # Update position velocities
                position_velocities = positions - self.previous_positions
            self.previous_positions = positions
        else:
            # Estimate positions from features (first 2 dims as proxy)
            estimated_positions = fused_features[:, :, :2]
            if self.previous_positions is not None:
                position_velocities = estimated_positions - self.previous_positions
            self.previous_positions = estimated_positions
        
        # Update occlusion history
        if occlusion_factors is not None:
            if self.occlusion_history is not None:
                # Exponential moving average
                alpha = 0.3
                self.occlusion_history = (
                    alpha * occlusion_factors + 
                    (1 - alpha) * self.occlusion_history
                )
            else:
                self.occlusion_history = occlusion_factors
        
        # Update previous features
        self.previous_features = fused_features.clone()
    
    def get_temporal_context(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get temporal context for prediction and gating.
        
        Returns:
            previous_features: [batch, num_objects, feature_dim] - Previous frame features
            velocity: [batch, num_objects, feature_dim] - Feature velocity
        """
        if self.previous_features is None:
            raise ValueError("Tracker not initialized. Call initialize() first.")
        
        return self.previous_features, self.velocities
    
    def reset(self):
        """Reset tracker state."""
        self.previous_features = None
        self.previous_positions = None
        self.velocities = None
        self.occlusion_history = None
        self.object_ids = None

