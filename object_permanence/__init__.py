"""
Object Permanence Module for Video Models
Implements Loci-Looped techniques for tracking objects through occlusions
"""

from .occlusion_detector import OcclusionDetector
from .object_predictor import ObjectPredictor
from .percept_gate_controller import PerceptGateController
from .temporal_fusion import TemporalFusion
from .object_tracker import ObjectTracker
from .spatial_memory_grid import SpatialMemoryGrid
from .integration import ObjectPermanenceModule

__all__ = [
    'OcclusionDetector',
    'ObjectPredictor',
    'PerceptGateController',
    'TemporalFusion',
    'ObjectTracker',
    'SpatialMemoryGrid',
    'ObjectPermanenceModule',
]

