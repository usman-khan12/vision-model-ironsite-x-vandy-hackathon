# Object Permanence Module

This module implements object permanence capabilities for video models using techniques inspired by Loci-Looped. It enables tracking objects through occlusions by fusing predicted object states with current observations using a learned gating mechanism.

## Overview

The object permanence module consists of several key components:

1. **OcclusionDetector**: Detects when objects are occluded or partially visible
2. **ObjectPredictor**: Predicts object features/positions for the next frame
3. **PerceptGateController**: Learns when to trust predictions vs. observations
4. **TemporalFusion**: Fuses predicted and observed features based on occlusion state
5. **ObjectTracker**: Maintains temporal memory for tracked objects
6. **SpatialMemoryGrid** (Novel): Maintains 2D spatial grid with "ghost" representations of occluded objects

## Architecture

```
Current Frame → Feature Extraction → Object Features
                                          ↓
Previous Frame → Object Tracker → Previous Features
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                             ↓
         Spatial Memory Grid                            ObjectPredictor
         (re-identify, get grid features)              (predict features)
                    ↓                                             ↓
         Enhanced Features                          Predicted Features
                    ↓                                             ↓
         OcclusionDetector                                      ↓
                    ↓                                             ↓
         Occlusion Factors                          Predictive Occupancy Maps
                    ↓                                             ↓
                    └─────────────────────┬─────────────────────┘
                                          ↓
                            PerceptGateController
                                          ↓
                                    Gate Values
                                          ↓
                                    TemporalFusion
                                          ↓
                                  Fused Features
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                             ↓
         Object Tracker (update)                    Spatial Grid (update)
```

## Key Components

### SpatialMemoryGrid (Novel Technique)

**Purpose**: Maintains a 2D spatial grid that tracks object presence even when occluded.

**Key Features**:
- **Ghost Representations**: Maintains object features in grid cells even when occluded
- **Predictive Occupancy Maps**: Predicts where objects should be spatially
- **Cross-Reference Matching**: Re-identifies objects using spatial context
- **Spatial Memory**: Explicit 2D representation of object locations

**Benefits**:
- Better re-identification after occlusion
- Long-term spatial memory
- Predictive capability for object locations
- Multi-object handling in same spatial region

See `SPATIAL_GRID_TECHNIQUE.md` for detailed documentation.

### OcclusionDetector

Detects occlusion state based on:
- Feature similarity between frames
- Feature magnitude
- Detection confidence
- Attention weights

**Output**: Occlusion factor [0, 1] where 0 = fully visible, 1 = fully occluded

### ObjectPredictor

Predicts next frame state using:
- Current object features
- Previous frame features
- Temporal context (velocity, acceleration)

**Output**: Predicted features and confidence scores

### PerceptGateController

Learns gating values to blend predictions and observations:
- Input: Current features, previous features, occlusion signal, temporal context
- Output: Gate values [0, 1] where:
  - 0 = fully occluded, use prediction
  - 1 = fully visible, use observation
  - Values in between = partial occlusion, blend

### TemporalFusion

Fuses features using learned gates:
```
fused_features = gate * observed + (1 - gate) * predicted
```

## Usage

### Basic Usage

```python
from object_permanence import ObjectPermanenceModule
from video_model_with_permanence import VideoModelWithPermanence

# Wrap your video model
model = VideoModelWithPermanence(
    base_model=your_video_model,
    feature_dim=512,
    num_objects=10,
    enable_permanence=True
)

# Process video frames
outputs = model.forward_with_permanence(inputs)
```

### With Test Script

```bash
python test_video_with_permanence.py \
    --video path/to/video.mp4 \
    --prompt "What is happening in this video?" \
    --enable-permanence \
    --show-permanence-info \
    --num-objects 10
```

## Integration Steps

1. **Extract Object Features**: The module extracts object-centric features from your model outputs
2. **Track Objects**: Maintains state for each tracked object across frames
3. **Detect Occlusion**: Computes occlusion factors for each object
4. **Predict State**: Predicts next frame features for occluded objects
5. **Gate and Fuse**: Blends predicted and observed features based on occlusion
6. **Update Tracking**: Updates object tracker with fused features

## Training Strategy

The module can be trained in phases:

1. **Phase 1**: Train predictor on visible objects (supervised)
2. **Phase 2**: Train gate controller with occlusion scenarios
3. **Phase 3**: End-to-end fine-tuning on videos with occlusions

### Loss Functions

- **Prediction Loss**: `L_pred = ||predicted_features - actual_features||` (only on visible objects)
- **Consistency Loss**: `L_cons = ||fused_features - actual_features||`
- **Gate Regularization**: Encourage gates to be binary (0 or 1) when appropriate

## Testing & Validation

Test on videos where objects:
- Move behind obstacles
- Leave and re-enter frame
- Are partially occluded

### Metrics

- Tracking continuity (ID switches)
- Position accuracy during occlusion
- Recovery after reappearance

## Configuration

Key parameters:

- `feature_dim`: Dimension of object features (default: 512)
- `num_objects`: Number of objects to track (default: 10)
- `hidden_dim`: Hidden dimension for MLPs (default: 256)
- `use_transformer_predictor`: Use Transformer instead of MLP for prediction

## Notes

- The gate controller learns automatically - you don't need to manually specify when to use predictions
- Works best with object-centric representations, but can be adapted to dense features
- Computational efficiency: predictor only activates during occlusion if needed
- Start with a minimal version (simple predictor + basic gating) and scale up as needed

## References

Based on techniques from Loci-Looped:
- Percept Gate Controller: `model/nn/percept_gate_controller.py`
- Predictor: `model/nn/predictor.py` (LociPredictor)
- Occlusion Tracking: `model/utils/slot_utils.py` (OcclusionTracker)
- Main Integration: `model/loci.py` (inner loop gating)

