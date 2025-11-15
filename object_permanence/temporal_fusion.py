"""
Temporal Fusion Module
Fuses predicted and observed features based on occlusion state using learned gates
"""
import torch
import torch.nn as nn


class TemporalFusion(nn.Module):
    """
    Fuses predicted and observed features using learned gate values.
    Implements the core gating mechanism from Loci-Looped.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, observed_features, predicted_features, gate_values):
        """
        Fuse observed and predicted features using gate values.
        
        Args:
            observed_features: [batch, num_objects, feature_dim] - Features from current observation
            predicted_features: [batch, num_objects, feature_dim] - Features predicted from previous frame
            gate_values: [batch, num_objects] - Gate values (0=use prediction, 1=use observation)
            
        Returns:
            fused_features: [batch, num_objects, feature_dim] - Fused features
        """
        # Expand gate values to match feature dimensions
        gate_expanded = gate_values.unsqueeze(-1)  # [batch, num_objects, 1]
        
        # Linear interpolation: gated_features = gate * observed + (1-gate) * predicted
        fused_features = (
            gate_expanded * observed_features + 
            (1.0 - gate_expanded) * predicted_features
        )
        
        return fused_features
    
    def fuse_with_separate_gates(self, observed_features, predicted_features, 
                                 position_gates, appearance_gates, position_dim=None):
        """
        Fuse features with separate gates for position and appearance.
        
        Args:
            observed_features: [batch, num_objects, feature_dim] - Observed features
            predicted_features: [batch, num_objects, feature_dim] - Predicted features
            position_gates: [batch, num_objects] - Gates for position features
            appearance_gates: [batch, num_objects] - Gates for appearance features
            position_dim: Dimension of position features (if None, uses first half)
            
        Returns:
            fused_features: [batch, num_objects, feature_dim] - Fused features
        """
        feature_dim = observed_features.shape[-1]
        
        if position_dim is None:
            position_dim = feature_dim // 2
        
        # Split features into position and appearance
        observed_pos = observed_features[:, :, :position_dim]
        observed_app = observed_features[:, :, position_dim:]
        predicted_pos = predicted_features[:, :, :position_dim]
        predicted_app = predicted_features[:, :, position_dim:]
        
        # Fuse separately
        fused_pos = self.forward(observed_pos, predicted_pos, position_gates)
        fused_app = self.forward(observed_app, predicted_app, appearance_gates)
        
        # Concatenate
        fused_features = torch.cat([fused_pos, fused_app], dim=-1)
        
        return fused_features

