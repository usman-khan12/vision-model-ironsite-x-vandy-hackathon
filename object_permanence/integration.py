"""
Integration Module
Integrates object permanence components into video processing pipeline
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .occlusion_detector import OcclusionDetector
from .object_predictor import ObjectPredictor
from .percept_gate_controller import PerceptGateController
from .temporal_fusion import TemporalFusion
from .object_tracker import ObjectTracker
from .feature_extractor import SimpleFeatureExtractor


class ObjectPermanenceModule(nn.Module):
    """
    Main module that integrates all object permanence components.
    Processes video frames with object tracking through occlusions.
    """
    
    def __init__(
        self,
        feature_dim=512,
        num_objects=10,
        hidden_dim=256,
        use_transformer_predictor=False
    ):
        """
        Args:
            feature_dim: Dimension of object features
            num_objects: Number of objects to track
            hidden_dim: Hidden dimension for MLPs
            use_transformer_predictor: Whether to use Transformer for prediction
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_objects = num_objects
        
        # Core modules
        self.occlusion_detector = OcclusionDetector(feature_dim, hidden_dim)
        self.object_predictor = ObjectPredictor(
            feature_dim, 
            hidden_dim, 
            use_transformer=use_transformer_predictor
        )
        self.percept_gate_controller = PerceptGateController(
            feature_dim, 
            num_objects, 
            hidden_dim
        )
        self.temporal_fusion = TemporalFusion()
        
        # Tracker (not a module, just state)
        self.tracker = ObjectTracker(max_objects=num_objects, feature_dim=feature_dim)
        
        # Feature extractor (will be set externally or use default)
        self.feature_extractor: Optional[nn.Module] = None
    
    def set_feature_extractor(self, extractor: nn.Module):
        """Set the feature extractor for extracting object features from model outputs."""
        self.feature_extractor = extractor
    
    def forward(
        self,
        current_model_outputs,
        previous_features: Optional[torch.Tensor] = None,
        frame_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Process a frame with object permanence.
        
        Args:
            current_model_outputs: Model outputs for current frame
            previous_features: Previous frame features (optional, uses tracker if None)
            frame_idx: Current frame index
            
        Returns:
            Dictionary containing:
            - fused_features: [batch, num_objects, feature_dim] - Fused object features
            - occlusion_factors: [batch, num_objects] - Occlusion factors
            - gate_values: [batch, num_objects] - Gate values used
            - predicted_features: [batch, num_objects, feature_dim] - Predicted features
            - object_positions: [batch, num_objects, 2] - Object positions
        """
        batch_size = current_model_outputs.shape[0] if isinstance(current_model_outputs, torch.Tensor) else current_model_outputs[0].shape[0]
        device = current_model_outputs.device if isinstance(current_model_outputs, torch.Tensor) else current_model_outputs[0].device
        
        # Initialize tracker if needed
        if self.tracker.previous_features is None:
            self.tracker.initialize(batch_size, self.num_objects, device)
        
        # Extract current object features
        if self.feature_extractor is not None:
            current_features, object_positions, detection_confidence = self.feature_extractor(current_model_outputs)
        else:
            # Fallback: use model outputs directly (assumes correct shape)
            if isinstance(current_model_outputs, torch.Tensor):
                if len(current_model_outputs.shape) == 2:
                    # [batch, feature_dim] - expand to objects
                    current_features = current_model_outputs.unsqueeze(1).expand(-1, self.num_objects, -1)
                else:
                    current_features = current_features[:, :self.num_objects, :]
            else:
                # Assume tuple/list of tensors
                current_features = current_model_outputs[0][:, :self.num_objects, :]
            
            object_positions = torch.zeros(batch_size, self.num_objects, 2, device=device)
            detection_confidence = torch.ones(batch_size, self.num_objects, device=device)
        
        # Get previous features from tracker
        if previous_features is None:
            previous_features, velocity = self.tracker.get_temporal_context()
        else:
            velocity = current_features - previous_features
        
        # 1. Detect occlusion
        occlusion_factors, visibility_scores = self.occlusion_detector(
            current_features,
            previous_features,
            attention_weights=detection_confidence,
            detection_confidence=detection_confidence
        )
        
        # 2. Predict next state
        predicted_features, prediction_confidence = self.object_predictor(
            current_features,
            previous_features,
            temporal_context=velocity
        )
        
        # 3. Compute gate values
        position_gates, appearance_gates = self.percept_gate_controller(
            current_features,
            previous_features,
            occlusion_factors,
            temporal_context=velocity
        )
        
        # 4. Fuse features (use position gates for now, can use separate gates later)
        fused_features = self.temporal_fusion(
            current_features,
            predicted_features,
            position_gates  # Using position gates for all features
        )
        
        # 5. Update tracker
        self.tracker.update(fused_features, object_positions, occlusion_factors)
        
        return {
            'fused_features': fused_features,
            'occlusion_factors': occlusion_factors,
            'visibility_scores': visibility_scores,
            'position_gates': position_gates,
            'appearance_gates': appearance_gates,
            'predicted_features': predicted_features,
            'prediction_confidence': prediction_confidence,
            'object_positions': object_positions,
            'detection_confidence': detection_confidence
        }
    
    def process_sequence(self, model_outputs_sequence):
        """
        Process a sequence of frames.
        
        Args:
            model_outputs_sequence: List of model outputs for each frame
            
        Returns:
            List of output dictionaries for each frame
        """
        outputs = []
        
        # Reset tracker for new sequence
        self.tracker.reset()
        
        for frame_idx, frame_outputs in enumerate(model_outputs_sequence):
            frame_output = self.forward(frame_outputs, frame_idx=frame_idx)
            outputs.append(frame_output)
        
        return outputs
    
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()

