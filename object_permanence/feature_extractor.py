"""
Feature Extractor Module
Extracts object-centric features from video model outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ObjectFeatureExtractor(nn.Module):
    """
    Extracts object-centric features from video model outputs.
    Can work with attention-based or dense feature representations.
    """
    
    def __init__(self, model_hidden_dim, object_feature_dim=512, num_objects=10):
        """
        Args:
            model_hidden_dim: Hidden dimension of the video model
            object_feature_dim: Desired dimension for object features
            num_objects: Number of objects to extract
        """
        super().__init__()
        self.model_hidden_dim = model_hidden_dim
        self.object_feature_dim = object_feature_dim
        self.num_objects = num_objects
        
        # Project model features to object feature space
        self.feature_projection = nn.Linear(model_hidden_dim, object_feature_dim)
        
        # Object query embeddings (learnable)
        self.object_queries = nn.Parameter(
            torch.randn(1, num_objects, object_feature_dim)
        )
        
        # Cross-attention for object feature extraction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=object_feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Object feature refinement
        self.refinement = nn.Sequential(
            nn.LayerNorm(object_feature_dim),
            nn.Linear(object_feature_dim, object_feature_dim),
            nn.ReLU(),
            nn.Linear(object_feature_dim, object_feature_dim)
        )
    
    def extract_from_attention(self, model_outputs, attention_weights=None):
        """
        Extract object features from model attention outputs.
        
        Args:
            model_outputs: Model hidden states or features
            attention_weights: Optional attention weights for object detection
            
        Returns:
            object_features: [batch, num_objects, object_feature_dim]
            object_positions: [batch, num_objects, 2] - Estimated positions
            detection_confidence: [batch, num_objects] - Detection confidence
        """
        # Handle different input formats
        if isinstance(model_outputs, dict):
            hidden_states = model_outputs.get('hidden_states', model_outputs.get('last_hidden_state'))
        elif isinstance(model_outputs, tuple):
            hidden_states = model_outputs[0]
        else:
            hidden_states = model_outputs
        
        batch_size = hidden_states.shape[0]
        
        # Project to object feature dimension
        projected_features = self.feature_projection(hidden_states)  # [batch, seq_len, object_feature_dim]
        
        # Expand object queries
        queries = self.object_queries.expand(batch_size, -1, -1)  # [batch, num_objects, object_feature_dim]
        
        # Cross-attention: queries attend to projected features
        object_features, attention_scores = self.cross_attention(
            query=queries,
            key=projected_features,
            value=projected_features
        )
        
        # Refine features
        object_features = object_features + self.refinement(object_features)
        
        # Compute detection confidence from attention scores
        # Average attention across sequence length
        detection_confidence = torch.mean(attention_scores, dim=-1)  # [batch, num_objects, seq_len]
        detection_confidence = torch.mean(detection_confidence, dim=-1)  # [batch, num_objects]
        
        # Estimate positions from attention (weighted average of spatial positions)
        # For simplicity, use feature statistics as position proxy
        # In practice, you'd extract spatial positions from spatial features
        object_positions = torch.mean(object_features[:, :, :2], dim=-1, keepdim=True)  # [batch, num_objects, 1]
        object_positions = object_positions.repeat(1, 1, 2)  # [batch, num_objects, 2]
        
        return object_features, object_positions, detection_confidence
    
    def extract_from_dense_features(self, dense_features, spatial_shape=None):
        """
        Extract object features from dense spatial features.
        
        Args:
            dense_features: [batch, channels, height, width] or [batch, height*width, channels]
            spatial_shape: Optional (height, width) for reshaping
            
        Returns:
            object_features: [batch, num_objects, object_feature_dim]
            object_positions: [batch, num_objects, 2]
            detection_confidence: [batch, num_objects]
        """
        batch_size = dense_features.shape[0]
        
        # Reshape if needed
        if len(dense_features.shape) == 4:  # [batch, channels, height, width]
            channels, height, width = dense_features.shape[1:]
            dense_features = dense_features.view(batch_size, channels, height * width)
            dense_features = dense_features.permute(0, 2, 1)  # [batch, height*width, channels]
        elif len(dense_features.shape) == 3:  # [batch, seq_len, channels]
            pass
        else:
            raise ValueError(f"Unexpected dense_features shape: {dense_features.shape}")
        
        # Project to object feature dimension
        projected_features = self.feature_projection(dense_features)
        
        # Expand object queries
        queries = self.object_queries.expand(batch_size, -1, -1)
        
        # Cross-attention
        object_features, attention_scores = self.cross_attention(
            query=queries,
            key=projected_features,
            value=projected_features
        )
        
        # Refine features
        object_features = object_features + self.refinement(object_features)
        
        # Detection confidence
        detection_confidence = torch.mean(attention_scores, dim=-1)
        detection_confidence = torch.mean(detection_confidence, dim=-1)
        
        # Estimate positions from attention-weighted spatial locations
        if spatial_shape is not None:
            height, width = spatial_shape
            # Create spatial grid
            y_coords = torch.arange(height, device=dense_features.device).float()
            x_coords = torch.arange(width, device=dense_features.device).float()
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            spatial_grid = torch.stack([x_grid, y_grid], dim=-1)  # [height, width, 2]
            spatial_grid = spatial_grid.view(height * width, 2)  # [height*width, 2]
            spatial_grid = spatial_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, height*width, 2]
            
            # Weight by attention
            attention_weights = torch.mean(attention_scores, dim=1)  # [batch, num_objects, height*width]
            object_positions = torch.bmm(
                attention_weights.unsqueeze(1),  # [batch, 1, num_objects, height*width]
                spatial_grid.unsqueeze(1)  # [batch, 1, height*width, 2]
            ).squeeze(1)  # [batch, num_objects, 2]
        else:
            # Fallback: use feature statistics
            object_positions = torch.mean(object_features[:, :, :2], dim=-1, keepdim=True)
            object_positions = object_positions.repeat(1, 1, 2)
        
        return object_features, object_positions, detection_confidence


class SimpleFeatureExtractor(nn.Module):
    """
    Simplified feature extractor that works with any model output.
    Uses pooling and projection to create object features.
    """
    
    def __init__(self, input_dim, object_feature_dim=512, num_objects=10):
        super().__init__()
        self.input_dim = input_dim
        self.object_feature_dim = object_feature_dim
        self.num_objects = num_objects
        
        # Simple projection and pooling
        self.projection = nn.Linear(input_dim, object_feature_dim)
        self.object_embeddings = nn.Parameter(
            torch.randn(num_objects, object_feature_dim)
        )
        
    def forward(self, model_features):
        """
        Extract object features from model features.
        
        Args:
            model_features: [batch, seq_len, input_dim] or [batch, input_dim]
            
        Returns:
            object_features: [batch, num_objects, object_feature_dim]
            object_positions: [batch, num_objects, 2]
            detection_confidence: [batch, num_objects]
        """
        if len(model_features.shape) == 2:
            # [batch, input_dim] - add sequence dimension
            model_features = model_features.unsqueeze(1)
        
        batch_size, seq_len, input_dim = model_features.shape
        
        # Project features
        projected = self.projection(model_features)  # [batch, seq_len, object_feature_dim]
        
        # Pool across sequence (mean pooling)
        pooled = torch.mean(projected, dim=1)  # [batch, object_feature_dim]
        
        # Create object features by combining pooled features with object embeddings
        object_emb = self.object_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, self.num_objects, -1)
        
        # Combine
        object_features = object_emb + 0.1 * pooled_expanded
        
        # Simple position estimate (from feature statistics)
        object_positions = torch.mean(object_features[:, :, :2], dim=-1, keepdim=True)
        object_positions = object_positions.repeat(1, 1, 2)
        
        # Simple confidence (uniform for now)
        detection_confidence = torch.ones(batch_size, self.num_objects, device=model_features.device) * 0.8
        
        return object_features, object_positions, detection_confidence

