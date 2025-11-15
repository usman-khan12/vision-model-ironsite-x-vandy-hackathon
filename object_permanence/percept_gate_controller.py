"""
Percept Gate Controller Module
Learns when to trust predictions vs. observations based on occlusion state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptGateController(nn.Module):
    """
    Learns gating values to blend predicted and observed features.
    Outputs gate values per object where:
    - 0 = fully occluded, use prediction
    - 1 = fully visible, use observation
    - Values in between = partial occlusion, blend
    """
    
    def __init__(self, feature_dim, num_objects=None, hidden_dim=128, num_layers=2):
        """
        Args:
            feature_dim: Dimension of object features
            num_objects: Number of objects (optional, for fixed-size)
            hidden_dim: Hidden dimension for MLP
            num_layers: Number of MLP layers
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Input: [current_features, prev_features, occlusion_signal, temporal_context]
        # Temporal context includes: position changes, velocity estimates
        input_dim = feature_dim * 2 + 1 + feature_dim  # features + occlusion + velocity
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Separate heads for position and appearance gates
        self.position_gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.appearance_gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize biases to favor observation (bias ~3.0)
        # This means gates start closer to 1 (trust observation)
        with torch.no_grad():
            for layer in [self.position_gate_head[-2], self.appearance_gate_head[-2]]:
                if isinstance(layer, nn.Linear):
                    layer.bias.fill_(3.0)
    
    def forward(self, current_features, previous_features, occlusion_signal, temporal_context=None):
        """
        Compute gate values for fusing predicted and observed features.
        
        Args:
            current_features: [batch, num_objects, feature_dim] - Current frame features
            previous_features: [batch, num_objects, feature_dim] - Previous frame features
            occlusion_signal: [batch, num_objects] - Occlusion factor (0=visible, 1=occluded)
            temporal_context: [batch, num_objects, context_dim] - Temporal context (velocity, etc.)
            
        Returns:
            position_gates: [batch, num_objects] - Gate values for position features
            appearance_gates: [batch, num_objects] - Gate values for appearance features
        """
        batch_size, num_objects, feature_dim = current_features.shape
        
        # Compute velocity (change in features) as temporal context
        if temporal_context is None:
            velocity = current_features - previous_features  # [batch, num_objects, feature_dim]
        else:
            velocity = temporal_context
        
        # Prepare input: concatenate all signals
        # [current_features, prev_features, occlusion_signal, velocity]
        occlusion_expanded = occlusion_signal.unsqueeze(-1)  # [batch, num_objects, 1]
        
        gate_input = torch.cat([
            current_features,           # [batch, num_objects, feature_dim]
            previous_features,          # [batch, num_objects, feature_dim]
            occlusion_expanded,        # [batch, num_objects, 1]
            velocity,                  # [batch, num_objects, feature_dim]
        ], dim=-1)  # [batch, num_objects, 2*feature_dim + 1 + feature_dim]
        
        # Process through MLP
        mlp_out = self.mlp(gate_input)  # [batch, num_objects, hidden_dim]
        
        # Compute separate gates for position and appearance
        position_gates = self.position_gate_head(mlp_out).squeeze(-1)  # [batch, num_objects]
        appearance_gates = self.appearance_gate_head(mlp_out).squeeze(-1)  # [batch, num_objects]
        
        # Adjust gates based on occlusion: lower occlusion -> higher gate (trust observation)
        # Higher occlusion -> lower gate (trust prediction)
        position_gates = position_gates * (1.0 - occlusion_signal) + 0.1 * occlusion_signal
        appearance_gates = appearance_gates * (1.0 - occlusion_signal) + 0.1 * occlusion_signal
        
        # Clamp to [0, 1]
        position_gates = torch.clamp(position_gates, 0, 1)
        appearance_gates = torch.clamp(appearance_gates, 0, 1)
        
        return position_gates, appearance_gates

