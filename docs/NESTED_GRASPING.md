# Nested-Grasping Model (36-DoF)

## Overview

The Nested-Grasping adaptation extends the original 6-DoF GraspGen model to a hierarchical 36-DoF architecture. In this approach, each of the 6 degrees of freedom (DoF) has its own 6-DoF GraspGen implementation, resulting in a **Grasping-of-Grasping** model with 6 × 6 = 36 DoF.

## Architecture

### Original GraspGen (6-DoF)
- **Translation**: 3 DoF (x, y, z)
- **Rotation**: 3 DoF (represented as 3D rotation)
- **Total**: 6 DoF

### Nested-GraspGen (36-DoF)
The nested architecture processes each of the 6 original DoF through its own complete 6-DoF GraspGen:

```
Input Point Cloud
      |
      v
+------------------+
| Sub-Generator 1  |  (processes x-dimension)     → 6 DoF output
+------------------+
      |
      v
+------------------+
| Sub-Generator 2  |  (processes y-dimension)     → 6 DoF output
+------------------+
      |
      v
+------------------+
| Sub-Generator 3  |  (processes z-dimension)     → 6 DoF output
+------------------+
      |
      v
+------------------+
| Sub-Generator 4  |  (processes rot_1-dimension) → 6 DoF output
+------------------+
      |
      v
+------------------+
| Sub-Generator 5  |  (processes rot_2-dimension) → 6 DoF output
+------------------+
      |
      v
+------------------+
| Sub-Generator 6  |  (processes rot_3-dimension) → 6 DoF output
+------------------+
      |
      v
Combined 36-DoF Output
```

### Key Features

1. **Hierarchical Processing**: Each sub-generator can process the output of the previous one, allowing for progressive refinement of grasp predictions.

2. **Parallel Processing**: Alternatively, all sub-generators can process the input independently and their outputs combined.

3. **Flexible Combination**: The model supports two strategies:
   - `hierarchical`: Each sub-generator processes the refined output from the previous one
   - `parallel`: All sub-generators process the original input independently

## Output Format

For a grasp representation `r3_6d` (9-dimensional base):
- Each sub-generator produces: 9 values (3 for position + 6 for rotation in 6D representation)
- Total nested output: 6 sub-generators × 9 = **54 dimensions**

For a grasp representation `r3_so3` (6-dimensional base):
- Each sub-generator produces: 6 values (3 for position + 3 for rotation)
- Total nested output: 6 sub-generators × 6 = **36 dimensions**

The output is structured as:
```python
outputs = {
    "grasps_pred": [
        # For each object in batch:
        tensor of shape [num_grasps, 6_sub_generators, 4, 4]
        # Where each 4×4 matrix is a homogeneous transformation
    ],
    "sub_generator_outputs": [
        # Individual outputs from each sub-generator
    ],
    ...
}
```

## Usage

### Basic Inference

```python
from grasp_gen.models.nested_generator import NestedGraspGenGenerator
from omegaconf import DictConfig

# Configure sub-generators (use same config for all 6)
base_config = DictConfig({
    "num_embed_dim": 256,
    "num_obs_dim": 512,
    "diffusion_embed_dim": 512,
    "num_diffusion_iters": 100,
    "num_diffusion_iters_eval": 100,
    "obs_backbone": "pointnet",
    "grasp_repr": "r3_6d",
    "gripper_name": "franka_panda",
    "num_grasps_per_object": 20,
    # ... other parameters
})

# Create nested model
nested_model = NestedGraspGenGenerator(
    sub_generator_configs=[base_config] * 6,
    combine_strategy="hierarchical",
    grasp_repr="r3_6d"
)

# Run inference
data = {"points": torch.randn(1, 1024, 3)}  # Point cloud
outputs, losses, stats = nested_model.infer(data, return_metrics=False)

# Access results
grasps = outputs["grasps_pred"]  # List of [num_grasps, 6, 4, 4] tensors
```

### With Discriminator

```python
from grasp_gen.models.nested_grasp_gen import NestedGraspGen

# Create combined model (generator + discriminator)
model = NestedGraspGen(nested_generator_cfg, discriminator_cfg)

# Run inference with scoring
outputs, losses, stats = model.infer(data)

# Access scored grasps
grasps = outputs["grasps_pred"]
scores = outputs.get("scores", None)  # Discriminator scores
```

### Using the Inference Script

```bash
python scripts/inference_nested_graspgen.py \
    --sample_data_dir /path/to/sample_data \
    --gripper_config /path/to/gripper_config.yml \
    --num_grasps 20 \
    --combine_strategy hierarchical \
    --device cuda
```

## Implementation Details

### Classes

1. **`NestedGraspGenGenerator`** (`grasp_gen/models/nested_generator.py`)
   - Main nested generator class
   - Manages 6 sub-generators
   - Implements hierarchical and parallel combination strategies
   - Handles training and inference

2. **`NestedGraspGen`** (`grasp_gen/models/nested_grasp_gen.py`)
   - Wrapper combining nested generator with discriminator
   - Similar to original `GraspGen` class
   - Adds scoring and ranking capabilities

### Key Methods

- `forward_train(data)`: Training forward pass through all sub-generators
- `forward_inference(data, return_metrics)`: Inference with optional metrics
- `infer(data, return_metrics)`: Main inference method
- `_combine_outputs(all_outputs, data)`: Combines outputs from all sub-generators
- `from_config(cfg)`: Creates model from configuration

## Training

To train the nested model, you would need to:

1. Prepare grasp datasets for each sub-generator
2. Configure training parameters for hierarchical or parallel training
3. Run training similar to standard GraspGen but with nested architecture

```python
# Example training loop (conceptual)
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs, losses, stats = nested_model.forward(batch, eval=False)
        
        # Losses from all sub-generators
        total_loss = sum(weight * loss for weight, loss in losses.values())
        total_loss.backward()
        optimizer.step()
```

## Advantages

1. **Higher Expressiveness**: 36-DoF provides much more flexibility in grasp representation
2. **Hierarchical Refinement**: Sequential processing allows each sub-generator to refine previous predictions
3. **Modular Design**: Each sub-generator can be trained and fine-tuned independently
4. **Backward Compatible**: Can still extract standard 6-DoF grasps from output

## Considerations

- **Computational Cost**: 6× more parameters and computation than standard GraspGen
- **Memory Requirements**: Requires more GPU memory for training and inference
- **Training Complexity**: More challenging to train effectively than single-level model
- **Data Requirements**: May benefit from larger training datasets

## File Structure

```
grasp_gen/models/
├── nested_generator.py      # NestedGraspGenGenerator implementation
└── nested_grasp_gen.py       # NestedGraspGen wrapper class

scripts/
└── inference_nested_graspgen.py  # Example inference script

tests/
└── test_nested_model.py      # Test suite for nested model

docs/
└── NESTED_GRASPING.md        # This documentation file
```

## Testing

Run the test suite:
```bash
pytest tests/test_nested_model.py -v
```

Or validate the structure:
```bash
python validate_nested_model.py
```

## Future Extensions

Possible enhancements to the nested-grasping model:

1. **Variable Nesting Depth**: Support N-level nesting instead of just 2 levels
2. **Attention Mechanisms**: Add cross-attention between sub-generators
3. **Adaptive Combination**: Learn weights for combining sub-generator outputs
4. **Specialized Sub-Generators**: Train different sub-generators for different DoF aspects
5. **Pruning Strategies**: Dynamically select which sub-generators to use

## References

- Original GraspGen paper: [arXiv:2507.13097](https://arxiv.org/abs/2507.13097)
- Project page: [https://graspgen.github.io](https://graspgen.github.io)

## Citation

If you use the nested-grasping adaptation, please cite the original GraspGen paper:

```bibtex
@article{murali2025graspgen,
  title={GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training},
  author={Murali, Adithyavairavan and Sundaralingam, Balakumar and Chao, Yu-Wei and Yamada, Jun and Yuan, Wentao and Carlson, Mark and Ramos, Fabio and Birchfield, Stan and Fox, Dieter and Eppner, Clemens},
  journal={arXiv preprint arXiv:2507.13097},
  year={2025},
}
```
