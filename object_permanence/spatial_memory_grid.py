"""
Spatial Memory Grid with Predictive Occupancy Maps
Novel technique for maintaining object presence in spatial grid even when occluded
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class SpatialMemoryGrid(nn.Module):
    """
    Maintains a 2D spatial grid that tracks object presence even when occluded.
    Each grid cell stores object features, IDs, and confidence scores.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (32, 32),
        feature_dim: int = 512,
        num_objects: int = 10,
        hidden_dim: int = 256,
        decay_factor: float = 0.95
    ):
        """
        Args:
            grid_size: (height, width) of the spatial grid
            feature_dim: Dimension of object features
            num_objects: Maximum number of objects to track
            hidden_dim: Hidden dimension for processing
            decay_factor: Decay factor for grid cell confidence over time
        """
        super().__init__()
        self.grid_height, self.grid_width = grid_size
        self.feature_dim = feature_dim
        self.num_objects = num_objects
        self.decay_factor = decay_factor
        
        # Grid structure: [batch, H, W, num_objects, feature_dim + metadata]
        # Metadata includes: confidence, object_id, temporal_consistency
        
        # Grid cell processor - processes object features in each cell
        self.cell_processor = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),  # +3 for metadata (confidence, id, time)
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim + 3)
        )
        
        # Occupancy predictor - predicts where objects should be
        self.occupancy_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, hidden_dim),  # features + position + velocity
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_height * self.grid_width),
            nn.Sigmoid()  # Occupancy probability
        )
        
        # Cross-reference matcher - matches observations to grid predictions
        self.cross_reference = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, hidden_dim),  # features + position
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects),
            nn.Softmax(dim=-1)  # Matching probability per object
        )
        
        # Grid state (not parameters, but state)
        self.grid_state: Optional[torch.Tensor] = None
        self.grid_confidence: Optional[torch.Tensor] = None
        self.grid_temporal: Optional[torch.Tensor] = None
    
    def initialize_grid(self, batch_size: int, device: torch.device):
        """Initialize the spatial memory grid."""
        # Grid: [batch, H, W, num_objects, feature_dim]
        self.grid_state = torch.zeros(
            batch_size, self.grid_height, self.grid_width, 
            self.num_objects, self.feature_dim, device=device
        )
        
        # Confidence: [batch, H, W, num_objects]
        self.grid_confidence = torch.zeros(
            batch_size, self.grid_height, self.grid_width, 
            self.num_objects, device=device
        )
        
        # Temporal consistency: [batch, H, W, num_objects] - tracks how long object has been at location
        self.grid_temporal = torch.zeros(
            batch_size, self.grid_height, self.grid_width, 
            self.num_objects, device=device
        )
    
    def position_to_grid(self, positions: torch.Tensor, image_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Convert object positions to grid coordinates.
        
        Args:
            positions: [batch, num_objects, 2] - (x, y) positions (normalized 0-1 or pixel coords)
            image_shape: (height, width) of original image (if positions are in pixels)
            
        Returns:
            grid_coords: [batch, num_objects, 2] - (grid_h, grid_w) coordinates
        """
        batch_size, num_objects, _ = positions.shape
        
        # Normalize positions to [0, 1] if needed
        if image_shape is not None:
            h, w = image_shape
            positions = positions.clone()
            positions[:, :, 0] = positions[:, :, 0] / w  # x
            positions[:, :, 1] = positions[:, :, 1] / h  # y
        
        # Convert to grid coordinates
        grid_coords = positions.clone()
        grid_coords[:, :, 0] = grid_coords[:, :, 0] * (self.grid_width - 1)  # x -> grid_w
        grid_coords[:, :, 1] = grid_coords[:, :, 1] * (self.grid_height - 1)  # y -> grid_h
        
        # Clamp to valid grid range
        grid_coords = torch.clamp(grid_coords, 0, max(self.grid_height, self.grid_width) - 1)
        
        return grid_coords.long()
    
    def update_grid(
        self,
        object_features: torch.Tensor,
        positions: torch.Tensor,
        occlusion_factors: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None
    ):
        """
        Update the spatial memory grid with current observations.
        
        Args:
            object_features: [batch, num_objects, feature_dim] - Current object features
            positions: [batch, num_objects, 2] - Object positions
            occlusion_factors: [batch, num_objects] - Occlusion factors (0=visible, 1=occluded)
            object_ids: [batch, num_objects] - Object IDs (optional)
        """
        batch_size, num_objects, feature_dim = object_features.shape
        device = object_features.device
        
        # Initialize grid if needed
        if self.grid_state is None:
            self.initialize_grid(batch_size, device)
        
        # Convert positions to grid coordinates
        grid_coords = self.position_to_grid(positions)  # [batch, num_objects, 2]
        
        # Update grid for each object
        for obj_idx in range(num_objects):
            for b_idx in range(batch_size):
                grid_h = grid_coords[b_idx, obj_idx, 1].item()
                grid_w = grid_coords[b_idx, obj_idx, 0].item()
                occlusion = occlusion_factors[b_idx, obj_idx].item()
                
                # Get current cell state
                current_features = self.grid_state[b_idx, grid_h, grid_w, obj_idx, :]
                current_confidence = self.grid_confidence[b_idx, grid_h, grid_w, obj_idx].item()
                
                # Update based on visibility
                if occlusion < 0.5:  # Object is visible
                    # Strong update with observed features
                    alpha = 0.8  # High update rate for visible objects
                    new_features = (
                        alpha * object_features[b_idx, obj_idx, :] +
                        (1 - alpha) * current_features
                    )
                    new_confidence = min(1.0, current_confidence * 0.9 + 0.5)  # Increase confidence
                    temporal_inc = 1.0
                else:  # Object is occluded
                    # Maintain "ghost" presence with decay
                    alpha = 0.3  # Lower update rate for occluded objects
                    new_features = (
                        alpha * object_features[b_idx, obj_idx, :] +
                        (1 - alpha) * current_features
                    )
                    new_confidence = current_confidence * self.decay_factor  # Decay confidence
                    temporal_inc = 0.5
                
                # Update grid
                self.grid_state[b_idx, grid_h, grid_w, obj_idx, :] = new_features
                self.grid_confidence[b_idx, grid_h, grid_w, obj_idx] = new_confidence
                self.grid_temporal[b_idx, grid_h, grid_w, obj_idx] += temporal_inc
        
        # Decay all cells (reduce confidence for cells not updated)
        self.grid_confidence = self.grid_confidence * self.decay_factor
    
    def predict_occupancy(
        self,
        object_features: torch.Tensor,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        num_frames_ahead: int = 1
    ) -> torch.Tensor:
        """
        Predict spatial occupancy maps for objects.
        
        Args:
            object_features: [batch, num_objects, feature_dim] - Object features
            positions: [batch, num_objects, 2] - Current positions
            velocities: [batch, num_objects, 2] - Object velocities (optional)
            num_frames_ahead: Number of frames to predict ahead
            
        Returns:
            occupancy_maps: [batch, num_objects, H, W] - Predicted occupancy probabilities
        """
        batch_size, num_objects, feature_dim = object_features.shape
        device = object_features.device
        
        # Default velocities if not provided
        if velocities is None:
            velocities = torch.zeros_like(positions)
        
        # Predict occupancy for each object
        occupancy_maps = []
        
        for obj_idx in range(num_objects):
            obj_features = object_features[:, obj_idx, :]  # [batch, feature_dim]
            obj_pos = positions[:, obj_idx, :]  # [batch, 2]
            obj_vel = velocities[:, obj_idx, :]  # [batch, 2]
            
            # Predict future position
            future_pos = obj_pos + obj_vel * num_frames_ahead
            
            # Prepare input: features + current_pos + future_pos + velocity
            predictor_input = torch.cat([
                obj_features,
                obj_pos,
                future_pos,
                obj_vel
            ], dim=-1)  # [batch, feature_dim + 2 + 2 + 2]
            
            # Predict occupancy map
            occupancy_flat = self.occupancy_predictor(predictor_input)  # [batch, H*W]
            occupancy_map = occupancy_flat.view(batch_size, self.grid_height, self.grid_width)
            occupancy_maps.append(occupancy_map)
        
        # Stack: [batch, num_objects, H, W]
        occupancy_maps = torch.stack(occupancy_maps, dim=1)
        
        return occupancy_maps
    
    def reidentify_object(
        self,
        new_features: torch.Tensor,
        new_position: torch.Tensor,
        threshold: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-identify objects by matching new observations to grid predictions.
        
        Args:
            new_features: [batch, num_objects, feature_dim] - Newly observed features
            new_position: [batch, num_objects, 2] - New positions
            threshold: Confidence threshold for matching
            
        Returns:
            matched_ids: [batch, num_objects] - Matched object IDs
            match_confidence: [batch, num_objects] - Match confidence scores
        """
        batch_size, num_objects, feature_dim = new_features.shape
        device = new_features.device
        
        if self.grid_state is None:
            # No grid history, return default IDs
            return torch.arange(num_objects, device=device).unsqueeze(0).repeat(batch_size, 1), \
                   torch.ones(batch_size, num_objects, device=device)
        
        matched_ids = []
        match_confidences = []
        
        # Convert positions to grid coordinates
        grid_coords = self.position_to_grid(new_position)
        
        for obj_idx in range(num_objects):
            obj_features = new_features[:, obj_idx, :]  # [batch, feature_dim]
            obj_grid_h = grid_coords[:, obj_idx, 1]  # [batch]
            obj_grid_w = grid_coords[:, obj_idx, 0]  # [batch]
            
            # Get grid features at this location for all tracked objects
            batch_matches = []
            batch_confs = []
            
            for b_idx in range(batch_size):
                grid_h = obj_grid_h[b_idx].item()
                grid_w = obj_grid_w[b_idx].item()
                
                # Get grid features for all objects at this location
                grid_features = self.grid_state[b_idx, grid_h, grid_w, :, :]  # [num_objects, feature_dim]
                grid_conf = self.grid_confidence[b_idx, grid_h, grid_w, :]  # [num_objects]
                
                # Match new features to grid features
                match_input = torch.cat([
                    obj_features[b_idx:b_idx+1, :].expand(self.num_objects, -1),  # [num_objects, feature_dim]
                    grid_features,  # [num_objects, feature_dim]
                    grid_coords[b_idx:b_idx+1, obj_idx:obj_idx+1, :].expand(self.num_objects, -1)  # [num_objects, 2]
                ], dim=-1)  # [num_objects, feature_dim*2 + 2]
                
                match_probs = self.cross_reference(match_input)  # [num_objects, num_objects]
                match_scores = match_probs.diag() * grid_conf  # Weight by grid confidence
                
                # Find best match
                best_match = torch.argmax(match_scores)
                best_conf = match_scores[best_match]
                
                batch_matches.append(best_match)
                batch_confs.append(best_conf)
            
            matched_ids.append(torch.stack(batch_matches))
            match_confidences.append(torch.stack(batch_confs))
        
        matched_ids = torch.stack(matched_ids, dim=1)  # [batch, num_objects]
        match_confidences = torch.stack(match_confidences, dim=1)  # [batch, num_objects]
        
        return matched_ids, match_confidences
    
    def get_grid_features_at_position(
        self,
        positions: torch.Tensor,
        object_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve grid features at specific positions.
        
        Args:
            positions: [batch, num_objects, 2] - Positions to query
            object_ids: [batch, num_objects] - Object IDs (optional, uses all if None)
            
        Returns:
            grid_features: [batch, num_objects, feature_dim] - Features from grid
        """
        if self.grid_state is None:
            batch_size = positions.shape[0]
            num_objects = positions.shape[1]
            device = positions.device
            return torch.zeros(batch_size, num_objects, self.feature_dim, device=device)
        
        batch_size, num_objects, _ = positions.shape
        grid_coords = self.position_to_grid(positions)
        
        grid_features = []
        for obj_idx in range(num_objects):
            obj_features = []
            for b_idx in range(batch_size):
                grid_h = grid_coords[b_idx, obj_idx, 1].item()
                grid_w = grid_coords[b_idx, obj_idx, 0].item()
                
                if object_ids is not None:
                    obj_id = object_ids[b_idx, obj_idx].item()
                    features = self.grid_state[b_idx, grid_h, grid_w, obj_id, :]
                else:
                    # Average over all objects at this location
                    features = self.grid_state[b_idx, grid_h, grid_w, :, :].mean(dim=0)
                
                obj_features.append(features)
            
            grid_features.append(torch.stack(obj_features))
        
        return torch.stack(grid_features, dim=1)  # [batch, num_objects, feature_dim]
    
    def reset(self):
        """Reset grid state."""
        self.grid_state = None
        self.grid_confidence = None
        self.grid_temporal = None

