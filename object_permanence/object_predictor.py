"""
Object State Predictor Module
Predicts object features/positions for the next frame, enabling tracking through occlusions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectPredictor(nn.Module):
    """
    Predicts object state (features, positions) for the next frame.
    Uses temporal context to maintain object identity through occlusions.
    """
    
    def __init__(self, feature_dim, hidden_dim=256, num_layers=2, use_transformer=False):
        """
        Args:
            feature_dim: Dimension of object features
            hidden_dim: Hidden dimension for MLP/Transformer
            num_layers: Number of layers
            use_transformer: Whether to use Transformer instead of MLP
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer
        
        if use_transformer:
            # Transformer-based predictor
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.predict_head = nn.Linear(feature_dim, feature_dim)
        else:
            # MLP-based predictor with residual connections
            layers = []
            input_dim = feature_dim
            
            for i in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            
            self.mlp = nn.Sequential(*layers)
            self.predict_head = nn.Linear(hidden_dim, feature_dim)
            
            # Residual connection
            self.residual = nn.Linear(feature_dim, feature_dim)
        
        # Velocity/acceleration predictor (for position prediction)
        self.velocity_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim)
        )
        
    def forward(self, current_features, previous_features=None, temporal_context=None):
        """
        Predict next frame features.
        
        Args:
            current_features: [batch, num_objects, feature_dim] - Current frame features
            previous_features: [batch, num_objects, feature_dim] - Previous frame features (optional)
            temporal_context: [batch, num_objects, context_dim] - Additional temporal context (optional)
            
        Returns:
            predicted_features: [batch, num_objects, feature_dim] - Predicted features for next frame
            confidence: [batch, num_objects] - Prediction confidence
        """
        batch_size, num_objects, feature_dim = current_features.shape
        
        # Build input with temporal context
        if previous_features is not None:
            # Compute velocity (change in features)
            velocity = current_features - previous_features
            
            # Predict next velocity
            predicted_velocity = self.velocity_predictor(current_features)
            
            # Combine current features with predicted velocity
            input_features = current_features + predicted_velocity
        else:
            input_features = current_features
        
        # Add temporal context if provided
        if temporal_context is not None:
            # Project temporal context to feature_dim if needed
            if temporal_context.shape[-1] != feature_dim:
                context_proj = nn.Linear(temporal_context.shape[-1], feature_dim).to(temporal_context.device)
                temporal_context = context_proj(temporal_context)
            input_features = input_features + temporal_context
        
        # Predict next features
        if self.use_transformer:
            # Use Transformer
            predicted = self.transformer(input_features)
            predicted_features = self.predict_head(predicted)
        else:
            # Use MLP with residual
            mlp_out = self.mlp(input_features)
            predicted_features = self.predict_head(mlp_out) + self.residual(input_features)
        
        # Compute prediction confidence based on feature stability
        if previous_features is not None:
            # Higher confidence if features are stable
            feature_stability = 1.0 - torch.mean(torch.abs(current_features - previous_features), dim=-1)
            confidence = torch.clamp(feature_stability, 0, 1)
        else:
            # Default confidence
            confidence = torch.ones(batch_size, num_objects, device=current_features.device)
        
        return predicted_features, confidence

