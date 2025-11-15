"""
Occlusion Detector Module
Detects when objects are occluded or partially visible
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionDetector(nn.Module):
    """
    Detects occlusion state of objects based on features, attention, and detection confidence.
    
    Computes occlusion indicators:
    - Visibility scores (0 = fully visible, 1 = fully occluded)
    - Partial occlusion detection
    """
    
    def __init__(self, feature_dim, hidden_dim=128):
        """
        Args:
            feature_dim: Dimension of object features
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # MLP to compute occlusion from feature similarity
        self.occlusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output occlusion factor [0, 1]
        )
        
    def forward(self, current_features, previous_features, attention_weights=None, detection_confidence=None):
        """
        Compute occlusion factor for each object.
        
        Args:
            current_features: [batch, num_objects, feature_dim] - Current frame features
            previous_features: [batch, num_objects, feature_dim] - Previous frame features
            attention_weights: [batch, num_objects] - Optional attention/confidence scores
            detection_confidence: [batch, num_objects] - Optional detection confidence scores
            
        Returns:
            occlusion_factor: [batch, num_objects] - Occlusion factor (0=visible, 1=occluded)
            visibility_score: [batch, num_objects] - Visibility score (1=visible, 0=occluded)
        """
        batch_size, num_objects, feature_dim = current_features.shape
        
        # Compute feature similarity between current and previous
        # Lower similarity suggests occlusion or significant change
        feature_diff = torch.abs(current_features - previous_features)
        feature_similarity = 1.0 - torch.mean(feature_diff, dim=-1)  # [batch, num_objects]
        
        # Normalize feature similarity to [0, 1]
        feature_similarity = torch.clamp(feature_similarity, 0, 1)
        
        # Concatenate current and previous features for MLP
        feature_pair = torch.cat([current_features, previous_features], dim=-1)  # [batch, num_objects, 2*feature_dim]
        
        # Compute occlusion from feature pairs
        occlusion_from_features = self.occlusion_mlp(feature_pair).squeeze(-1)  # [batch, num_objects]
        
        # Combine multiple signals
        # 1. Feature magnitude (low magnitude might indicate occlusion)
        feature_magnitude = torch.norm(current_features, dim=-1)  # [batch, num_objects]
        feature_magnitude_norm = 1.0 - torch.clamp(feature_magnitude / (feature_dim ** 0.5), 0, 1)
        
        # 2. Feature similarity (low similarity might indicate occlusion)
        similarity_occlusion = 1.0 - feature_similarity
        
        # 3. Detection confidence (if provided)
        if detection_confidence is not None:
            confidence_occlusion = 1.0 - detection_confidence
        else:
            confidence_occlusion = torch.zeros_like(occlusion_from_features)
        
        # 4. Attention weights (if provided, low attention might indicate occlusion)
        if attention_weights is not None:
            attention_occlusion = 1.0 - attention_weights
        else:
            attention_occlusion = torch.zeros_like(occlusion_from_features)
        
        # Weighted combination of occlusion signals
        occlusion_factor = (
            0.4 * occlusion_from_features +
            0.2 * similarity_occlusion +
            0.2 * feature_magnitude_norm +
            0.1 * confidence_occlusion +
            0.1 * attention_occlusion
        )
        
        # Clamp to [0, 1]
        occlusion_factor = torch.clamp(occlusion_factor, 0, 1)
        
        # Visibility is inverse of occlusion
        visibility_score = 1.0 - occlusion_factor
        
        return occlusion_factor, visibility_score

