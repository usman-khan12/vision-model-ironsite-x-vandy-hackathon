# Spatial Memory Grid with Predictive Occupancy Maps

## Novel Technique for Enhanced Object Permanence

### Overview

The **Spatial Memory Grid** is a novel technique that adds explicit spatial reasoning to object permanence tracking. Unlike traditional approaches that only track object features temporally, this technique maintains a 2D spatial representation of the scene where objects are tracked even when occluded.

### Key Innovation

**Main Idea**: Create a persistent 2D grid that maintains "ghost" representations of objects even when they're not visible. This provides:

1. **Explicit Spatial Reasoning**: Knows exactly where objects are in space
2. **Ghost Representations**: Maintains object presence in grid cells even when occluded
3. **Predictive Occupancy**: Predicts where objects should be based on spatial history
4. **Better Re-identification**: Uses spatial context to match objects after occlusion

### Architecture

```
Spatial Memory Grid (H x W cells)
├── Grid State: [batch, H, W, num_objects, feature_dim]
│   └── Stores object features at each spatial location
├── Grid Confidence: [batch, H, W, num_objects]
│   └── Confidence scores for each object at each location
└── Grid Temporal: [batch, H, W, num_objects]
    └── Temporal consistency (how long object has been at location)
```

### Core Components

#### 1. Grid Update Mechanism
- **Visible Objects**: Strong update (α=0.8) with observed features
- **Occluded Objects**: Maintain "ghost" presence with decay (α=0.3, decay=0.95)
- **Confidence Decay**: Grid cells decay over time if not updated

#### 2. Predictive Occupancy Maps
- Predicts spatial occupancy probability for each object
- Uses object features, positions, and velocities
- Outputs: [batch, num_objects, H, W] probability maps

#### 3. Cross-Reference Matching
- Matches new observations to grid predictions
- Uses spatial context + feature similarity
- Returns matched object IDs and confidence scores

#### 4. Grid Feature Retrieval
- Retrieves features from grid at specific positions
- Includes "ghost" features for occluded objects
- Enhances current observations with spatial memory

### Integration with Object Permanence Pipeline

```
Current Frame → Feature Extraction → Object Features
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                             ↓
         Spatial Memory Grid                            ObjectPredictor
         (re-identify objects)                        (predict features)
         (get grid features)                                   ↓
                    ↓                                    Predicted Features
         Enhanced Features                                    ↓
                    └─────────────────────┬─────────────────────┘
                                          ↓
                              Predictive Occupancy Maps
                              (where objects should be)
                                          ↓
                              Enhanced Predictions
                              (for occluded objects)
                                          ↓
                                    Temporal Fusion
                                          ↓
                                    Updated Grid
```

### Benefits

1. **Spatial Awareness**: Explicitly knows where objects are in 2D space
2. **Long-term Memory**: Maintains object presence across long occlusions
3. **Multi-object Handling**: Tracks multiple objects in same spatial region
4. **Re-identification**: Better matching after occlusion using spatial context
5. **Predictive Capability**: Predicts where objects will reappear

### Usage

The Spatial Memory Grid is automatically enabled in `ObjectPermanenceModule`:

```python
from object_permanence import ObjectPermanenceModule

module = ObjectPermanenceModule(
    feature_dim=512,
    num_objects=10,
    use_spatial_grid=True,  # Enable spatial grid
    grid_size=(32, 32)      # Grid resolution
)
```

### Outputs

When spatial grid is enabled, the module returns additional outputs:

- `occupancy_maps`: [batch, num_objects, H, W] - Predictive occupancy maps
- `object_ids`: [batch, num_objects] - Re-identified object IDs
- `grid_match_confidence`: [batch, num_objects] - Match confidence scores

### Novel Aspects

1. **Explicit Spatial Representation**: Not just features, but WHERE objects are
2. **Ghost Cells**: Maintains presence even when not visible
3. **Predictive Occupancy**: Predicts future locations spatially
4. **Spatial Re-identification**: Uses location to match objects

### Performance Considerations

- Grid size: Default (32x32) balances memory and resolution
- Decay factor: 0.95 provides good balance between memory and forgetting
- Update rates: Different for visible (0.8) vs occluded (0.3) objects

### Future Enhancements

- 3D spatial grids for depth-aware tracking
- Hierarchical grids (multi-scale)
- Attention-based grid updates
- Learned grid cell interactions

