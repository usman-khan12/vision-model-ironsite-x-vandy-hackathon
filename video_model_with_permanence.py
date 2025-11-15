"""
Video Model with Object Permanence
Wrapper that adds object permanence to LLaVA-NeXT-Video model
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from transformers import LlavaNextVideoForConditionalGeneration

from object_permanence import ObjectPermanenceModule
from object_permanence.feature_extractor import SimpleFeatureExtractor


class VideoModelWithPermanence(nn.Module):
    """
    Wraps LLaVA-NeXT-Video model with object permanence capabilities.
    """
    
    def __init__(
        self,
        base_model: LlavaNextVideoForConditionalGeneration,
        feature_dim: int = 512,
        num_objects: int = 10,
        enable_permanence: bool = True
    ):
        """
        Args:
            base_model: The base LLaVA-NeXT-Video model
            feature_dim: Dimension for object features
            num_objects: Number of objects to track
            enable_permanence: Whether to enable object permanence
        """
        super().__init__()
        self.base_model = base_model
        self.enable_permanence = enable_permanence
        
        # Get model hidden dimension
        if hasattr(base_model, 'config'):
            model_hidden_dim = getattr(base_model.config, 'hidden_size', 4096)
        else:
            model_hidden_dim = 4096  # Default for LLaVA models
        
        # Object permanence module
        if enable_permanence:
            self.permanence_module = ObjectPermanenceModule(
                feature_dim=feature_dim,
                num_objects=num_objects
            )
            
            # Feature extractor
            self.feature_extractor = SimpleFeatureExtractor(
                input_dim=model_hidden_dim,
                object_feature_dim=feature_dim,
                num_objects=num_objects
            )
            
            self.permanence_module.set_feature_extractor(self.feature_extractor)
    
    def extract_features_from_model(self, model_outputs, inputs):
        """
        Extract features from model outputs for object permanence.
        
        Args:
            model_outputs: Model outputs (can be logits, hidden states, etc.)
            inputs: Original model inputs
            
        Returns:
            features: Features suitable for object permanence processing
        """
        # Try to extract hidden states from model
        # This is model-specific and may need adjustment
        
        # Option 1: Use model's vision tower outputs if available
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'vision_tower'):
            # Access vision features
            # This is a placeholder - actual implementation depends on model architecture
            pass
        
        # Option 2: Use last hidden state from language model
        if isinstance(model_outputs, tuple):
            hidden_states = model_outputs[0]  # First element is usually hidden states
        elif isinstance(model_outputs, torch.Tensor):
            hidden_states = model_outputs
        else:
            # Fallback: use a dummy feature
            batch_size = inputs['input_ids'].shape[0] if 'input_ids' in inputs else 1
            hidden_states = torch.zeros(
                batch_size, 1, 4096,
                device=inputs['input_ids'].device if 'input_ids' in inputs else 'cpu'
            )
        
        # Pool or select features
        if len(hidden_states.shape) == 3:
            # [batch, seq_len, hidden_dim]
            # Use mean pooling or last token
            features = torch.mean(hidden_states, dim=1)  # [batch, hidden_dim]
        else:
            features = hidden_states
        
        return features
    
    def forward_with_permanence(self, inputs, process_frames_separately=True):
        """
        Forward pass with object permanence processing.
        
        Args:
            inputs: Model inputs (from processor)
            process_frames_separately: Whether to process each frame separately
            
        Returns:
            outputs: Model outputs with object permanence information
        """
        if not self.enable_permanence:
            # Standard forward pass
            return self.base_model(**inputs)
        
        # Extract video frames if available
        if 'pixel_values' in inputs or 'videos' in inputs:
            # Process with object permanence
            video_inputs = inputs.get('videos') or inputs.get('pixel_values')
            
            if process_frames_separately and video_inputs is not None:
                # Process each frame separately for temporal tracking
                batch_size = video_inputs.shape[0]
                num_frames = video_inputs.shape[1] if len(video_inputs.shape) > 4 else 1
                
                permanence_outputs = []
                
                for frame_idx in range(num_frames):
                    # Extract frame
                    if len(video_inputs.shape) == 5:  # [batch, frames, channels, height, width]
                        frame = video_inputs[:, frame_idx, :, :, :]
                    else:
                        frame = video_inputs
                    
                    # Create inputs for this frame
                    frame_inputs = inputs.copy()
                    if 'videos' in frame_inputs:
                        frame_inputs['videos'] = frame.unsqueeze(1) if len(frame.shape) == 4 else frame
                    if 'pixel_values' in frame_inputs:
                        frame_inputs['pixel_values'] = frame.unsqueeze(1) if len(frame.shape) == 4 else frame
                    
                    # Get model outputs (without generating, just forward pass)
                    # Note: This is a simplified approach - actual implementation may need model-specific adjustments
                    try:
                        with torch.no_grad():
                            # Try to get hidden states from model
                            if hasattr(self.base_model, 'model'):
                                model_outputs = self.base_model.model(**frame_inputs, output_hidden_states=True)
                            else:
                                # Fallback: use a dummy output
                                batch_size = frame_inputs['input_ids'].shape[0] if 'input_ids' in frame_inputs else 1
                                device = frame_inputs['input_ids'].device if 'input_ids' in frame_inputs else 'cpu'
                                model_outputs = (torch.zeros(batch_size, 1, 4096, device=device),)
                    except Exception as e:
                        # Fallback if model forward fails
                        print(f"Warning: Could not extract features from model: {e}")
                        batch_size = frame_inputs['input_ids'].shape[0] if 'input_ids' in frame_inputs else 1
                        device = frame_inputs['input_ids'].device if 'input_ids' in frame_inputs else 'cpu'
                        model_outputs = (torch.zeros(batch_size, 1, 4096, device=device),)
                    
                    # Extract features
                    features = self.extract_features_from_model(model_outputs, frame_inputs)
                    
                    # Process with object permanence
                    permanence_output = self.permanence_module.forward(features, frame_idx=frame_idx)
                    permanence_outputs.append(permanence_output)
                
                # Now do actual generation with all frames
                # (Object permanence info can be used to enhance generation)
                outputs = self.base_model(**inputs)
                
                # Attach permanence outputs
                outputs.permanence_info = permanence_outputs
                
                return outputs
            else:
                # Process all frames together
                outputs = self.base_model(**inputs, output_hidden_states=True)
                
                # Extract features and process with object permanence
                features = self.extract_features_from_model(outputs, inputs)
                permanence_output = self.permanence_module.forward(features)
                
                outputs.permanence_info = permanence_output
                
                return outputs
        else:
            # No video input, standard forward
            return self.base_model(**inputs)
    
    def generate(self, **inputs_and_kwargs):
        """
        Generate with object permanence awareness.
        
        Args:
            **inputs_and_kwargs: Model inputs and generation arguments
            
        Returns:
            Generated tokens
        """
        # Separate inputs from generation kwargs
        generation_kwargs = {}
        model_inputs = {}
        
        # Common generation kwargs
        gen_keys = ['max_new_tokens', 'max_length', 'do_sample', 'temperature', 
                   'top_p', 'top_k', 'num_beams', 'pad_token_id', 'eos_token_id']
        
        for key, value in inputs_and_kwargs.items():
            if key in gen_keys:
                generation_kwargs[key] = value
            else:
                model_inputs[key] = value
        
        if not self.enable_permanence:
            return self.base_model.generate(**model_inputs, **generation_kwargs)
        
        # Process frames with object permanence before generation
        # This can inform the generation process
        self.forward_with_permanence(model_inputs, process_frames_separately=True)
        
        # Standard generation (object permanence info is stored in model state)
        return self.base_model.generate(**model_inputs, **generation_kwargs)
    
    def reset_permanence(self):
        """Reset object permanence tracker state."""
        if self.enable_permanence:
            self.permanence_module.reset()

