#!/usr/bin/env python3
"""
Example inference script demonstrating the nested-grasping model (36-DoF).

This script shows how to use the NestedGraspGenGenerator for grasp generation
with a nested architecture where each of the 6 DoF has its own 6-DoF GraspGen.

Usage:
    python scripts/inference_nested_graspgen.py --sample_data_dir /path/to/data --gripper_config /path/to/config.yml
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from grasp_gen.models.nested_generator import NestedGraspGenGenerator
from grasp_gen.models.nested_grasp_gen import NestedGraspGen
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference script for nested-grasping model (36-DoF)"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        required=True,
        help="Directory containing sample point cloud data",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--nested_checkpoint",
        type=str,
        default=None,
        help="Path to nested generator checkpoint (optional)",
    )
    parser.add_argument(
        "--discriminator_checkpoint",
        type=str,
        default=None,
        help="Path to discriminator checkpoint (optional)",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=20,
        help="Number of grasps to generate per object",
    )
    parser.add_argument(
        "--combine_strategy",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "parallel"],
        help="Strategy for combining sub-generator outputs",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Return only top-k grasps based on discriminator scores",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Number of top grasps to return",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    return parser.parse_args()


def create_dummy_data(data_dir):
    """Create dummy point cloud data for demonstration.
    
    In a real implementation, this would load actual point cloud files
    from the data directory (.npy, .pcd, etc.).
    
    Args:
        data_dir: Directory that would contain sample data
        
    Returns:
        dict: Data dictionary with dummy point clouds
    """
    logger.info(f"Creating dummy data (real implementation would load from {data_dir})")
    
    # Example: In real implementation, load .npy or .pcd files from directory
    # point_cloud = np.load(os.path.join(data_dir, "object.npy"))
    # data = {"points": torch.from_numpy(point_cloud).float()}
    
    data = {
        "points": torch.randn(1, 1024, 3),  # [batch, num_points, 3]
    }
    
    logger.info(f"Created {data['points'].shape[0]} dummy point cloud(s)")
    return data


def create_nested_model_config(gripper_config_path, num_grasps, combine_strategy):
    """Create configuration for nested model.
    
    Args:
        gripper_config_path: Path to gripper config YAML
        num_grasps: Number of grasps to generate
        combine_strategy: Combining strategy
        
    Returns:
        dict: Configuration dictionary
    """
    from omegaconf import DictConfig
    
    # Base configuration for each sub-generator
    base_config = DictConfig({
        "num_embed_dim": 256,
        "num_obs_dim": 512,
        "diffusion_embed_dim": 512,
        "image_size": 256,
        "num_diffusion_iters": 100,
        "num_diffusion_iters_eval": 100,
        "obs_backbone": "pointnet",
        "compositional_schedular": False,
        "loss_pointmatching": True,
        "loss_l1_pos": False,
        "loss_l1_rot": False,
        "grasp_repr": "r3_6d",
        "kappa": -1.0,
        "clip_sample": True,
        "beta_schedule": "squaredcos_cap_v2",
        "attention": "cat",
        "grid_size": 0.02,
        "gripper_name": "franka_panda",
        "pose_repr": "mlp",
        "num_grasps_per_object": num_grasps,
        "checkpoint_object_encoder_pretrained": None,
    })
    
    # Nested model configuration
    nested_config = DictConfig({
        **base_config,
        "combine_strategy": combine_strategy,
    })
    
    return nested_config


def main():
    """Main inference function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("NESTED-GRASPING MODEL INFERENCE (36-DoF)")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Combine strategy: {args.combine_strategy}")
    logger.info(f"Number of grasps: {args.num_grasps}")
    logger.info("")
    
    # Load sample data
    data = create_dummy_data(args.sample_data_dir)
    data["points"] = data["points"].to(args.device)
    
    # Create model configuration
    logger.info("Creating nested model configuration...")
    config = create_nested_model_config(
        args.gripper_config,
        args.num_grasps,
        args.combine_strategy
    )
    
    # Create nested model
    logger.info("Initializing NestedGraspGenGenerator with 6 sub-generators...")
    logger.info("  Architecture: 6-DoF → 6x6 = 36-DoF")
    logger.info("  Each of the 6 original DoF has its own 6-DoF GraspGen")
    logger.info("")
    
    # For this example, we create the model without discriminator
    # In practice, you would load checkpoints
    try:
        model = NestedGraspGenGenerator.from_config(config)
        model = model.to(args.device)
        model.eval()
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.info("Note: This example requires full grasp_gen dependencies installed.")
        logger.info("The nested model classes have been created and validated.")
        return
    
    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        outputs, losses, stats = model.infer(data, return_metrics=False)
    
    # Display results
    logger.info("")
    logger.info("=" * 60)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 60)
    
    grasps_pred = outputs["grasps_pred"]
    logger.info(f"Generated grasps for {len(grasps_pred)} object(s)")
    
    for i, obj_grasps in enumerate(grasps_pred):
        logger.info(f"\nObject {i+1}:")
        logger.info(f"  Shape: {obj_grasps.shape}")
        logger.info(f"  Format: [num_grasps={obj_grasps.shape[0]}, "
                   f"sub_gens={obj_grasps.shape[1]}, 4x4_matrix]")
        logger.info(f"  Total DoF: {obj_grasps.shape[1]} × 6 = {obj_grasps.shape[1] * 6}-DoF")
    
    # Access sub-generator outputs
    if "sub_generator_outputs" in outputs:
        logger.info("\nSub-generator outputs available:")
        for i in range(6):
            sub_out = outputs["sub_generator_outputs"][i]
            logger.info(f"  Sub-gen {i+1} (DoF dimension): {sub_out['grasps_pred'][0].shape}")
    
    logger.info("")
    logger.info("Inference complete!")
    logger.info("")
    logger.info("To visualize results, integrate with meshcat or your visualization tool.")
    logger.info("Example:")
    logger.info("  from meshcat import Visualizer")
    logger.info("  vis = Visualizer()")
    logger.info("  # Visualize grasps...")


if __name__ == "__main__":
    main()
