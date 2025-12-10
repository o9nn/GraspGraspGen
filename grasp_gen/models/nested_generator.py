#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

from grasp_gen.models.generator import GraspGenGenerator
from grasp_gen.utils.math_utils import matrix_to_rt, rt_to_matrix
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


class NestedGraspGenGenerator(nn.Module):
    """Nested-Grasping adaptation of GraspGen with 36-DoF.
    
    This class implements a hierarchical grasping model where each of the 6 degrees of freedom
    has its own 6-DoF GraspGen implementation, resulting in a 6x6 = 36-DoF model.
    
    The model works by:
    1. Taking the input point cloud and processing it through 6 independent GraspGen sub-models
    2. Each sub-model generates grasps for one of the original 6-DoF dimensions
    3. The outputs are combined to produce a 36-dimensional grasp representation
    
    Args:
        sub_generator_configs (List): List of 6 configuration objects for each sub-generator
        combine_strategy (str): Strategy for combining sub-generator outputs ('concat', 'hierarchical'). Default: 'hierarchical'
        grasp_repr (str): Grasp representation type. Default: 'r3_6d'
    """

    def __init__(
        self,
        sub_generator_configs: Optional[List[Dict]] = None,
        combine_strategy: str = "hierarchical",
        grasp_repr: str = "r3_6d",
        **kwargs
    ):
        super().__init__()
        
        self.combine_strategy = combine_strategy
        self.grasp_repr = grasp_repr
        
        # Determine output dimension based on grasp representation
        if self.grasp_repr == "r3_6d":
            self.base_output_dim = 9  # 3 for position, 6 for rotation
        elif self.grasp_repr in ["r3_so3", "r3_euler"]:
            self.base_output_dim = 6  # 3 for position, 3 for rotation
        else:
            raise NotImplementedError(f"Rotation representation {grasp_repr} is not implemented!")
        
        # Total output dimension: 6 sub-generators, each producing base_output_dim values
        self.output_dim = 6 * self.base_output_dim
        
        # Create 6 sub-generators, one for each original DoF
        self.sub_generators = nn.ModuleList()
        
        if sub_generator_configs is None:
            # Use default configuration for all sub-generators if not provided
            logger.warning("No sub-generator configs provided, using default config for all 6 sub-generators")
            for i in range(6):
                self.sub_generators.append(GraspGenGenerator(**kwargs))
        else:
            if len(sub_generator_configs) != 6:
                raise ValueError(f"Expected 6 sub-generator configs, got {len(sub_generator_configs)}")
            for i, config in enumerate(sub_generator_configs):
                logger.info(f"Creating sub-generator {i+1}/6")
                if hasattr(config, '__dict__'):
                    # Config object - use from_config
                    self.sub_generators.append(GraspGenGenerator.from_config(config))
                else:
                    # Dict config - use direct initialization
                    self.sub_generators.append(GraspGenGenerator(**config))
        
        # Dimension labels for interpretation
        self.dof_names = ['x', 'y', 'z', 'rot_1', 'rot_2', 'rot_3']
        
        logger.info(f"NestedGraspGenGenerator initialized with {len(self.sub_generators)} sub-generators")
        logger.info(f"Base DoF: 6, Sub-generator DoF: {self.base_output_dim}, Total DoF: {self.output_dim}")

    @classmethod
    def from_config(cls, cfg):
        """Creates a NestedGraspGenGenerator instance from a configuration object.
        
        Args:
            cfg: Configuration object containing nested model parameters
            
        Returns:
            NestedGraspGenGenerator: Instantiated nested model
        """
        # Extract sub-generator configurations
        sub_configs = []
        if hasattr(cfg, 'sub_generators') and cfg.sub_generators is not None:
            for i in range(6):
                if hasattr(cfg.sub_generators, f'sub_gen_{i}'):
                    sub_configs.append(getattr(cfg.sub_generators, f'sub_gen_{i}'))
                else:
                    # Use base config for all sub-generators
                    sub_configs.append(cfg)
        else:
            # Use base config for all sub-generators
            sub_configs = [cfg] * 6
        
        args = {
            "sub_generator_configs": sub_configs,
            "combine_strategy": getattr(cfg, 'combine_strategy', 'hierarchical'),
            "grasp_repr": cfg.grasp_repr,
        }
        return cls(**args)

    def forward(self, data, cfg=None, eval=False):
        """Forward pass of the nested model.
        
        Args:
            data: Input data dictionary containing point clouds and optionally ground truth grasps
            cfg: Optional configuration object
            eval (bool): Whether to run in evaluation mode
            
        Returns:
            tuple: (outputs, losses, stats) containing model predictions, losses and metrics
        """
        if eval:
            return self.forward_inference(data, return_metrics=True)
        else:
            return self.forward_train(data)

    def forward_train(self, data):
        """Training forward pass implementing the nested diffusion process.
        
        Args:
            data: Input data dictionary containing point clouds and ground truth grasps
            
        Returns:
            tuple: (outputs, losses, stats) containing predictions, training losses and metrics
        """
        device = data["points"].device
        
        # Process through each sub-generator
        all_outputs = []
        all_losses = {}
        all_stats = {}
        
        if self.combine_strategy == "hierarchical":
            # Hierarchical: each sub-generator processes output from previous one
            current_data = data.copy()
            
            for i, sub_gen in enumerate(self.sub_generators):
                logger.debug(f"Processing sub-generator {i+1}/6 (DoF: {self.dof_names[i]})")
                
                # Forward pass through sub-generator
                outputs, losses, stats = sub_gen.forward_train(current_data)
                
                # Store outputs
                all_outputs.append(outputs)
                
                # Accumulate losses with DoF-specific prefix
                for loss_key, loss_val in losses.items():
                    all_losses[f"sub_{i}_{loss_key}"] = loss_val
                
                # Accumulate stats
                for stat_key, stat_val in stats.items():
                    all_stats[f"sub_{i}_{stat_key}"] = stat_val
                
                # For hierarchical processing, update data with current predictions
                # This allows the next sub-generator to refine based on previous output
                if i < 5:  # Don't need to update after last sub-generator
                    if "grasps_pred_mat" in outputs:
                        current_data["grasps"] = outputs["grasps_pred_mat"]
        else:
            # Parallel: all sub-generators process independently
            for i, sub_gen in enumerate(self.sub_generators):
                logger.debug(f"Processing sub-generator {i+1}/6 (DoF: {self.dof_names[i]})")
                
                outputs, losses, stats = sub_gen.forward_train(data)
                all_outputs.append(outputs)
                
                for loss_key, loss_val in losses.items():
                    all_losses[f"sub_{i}_{loss_key}"] = loss_val
                
                for stat_key, stat_val in stats.items():
                    all_stats[f"sub_{i}_{stat_key}"] = stat_val
        
        # Combine outputs from all sub-generators
        combined_outputs = self._combine_outputs(all_outputs, data)
        
        return combined_outputs, all_losses, all_stats

    def forward_inference(self, data, return_metrics=False):
        """Inference forward pass implementing the nested reverse diffusion process.
        
        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute evaluation metrics
            
        Returns:
            tuple: (outputs, losses, stats) containing generated grasps and optional metrics
        """
        device = data["points"].device
        
        # Process through each sub-generator
        all_outputs = []
        all_stats = {}
        
        if self.combine_strategy == "hierarchical":
            # Hierarchical: each sub-generator processes output from previous one
            current_data = data.copy()
            
            for i, sub_gen in enumerate(self.sub_generators):
                logger.debug(f"Inference sub-generator {i+1}/6 (DoF: {self.dof_names[i]})")
                
                # Inference through sub-generator
                outputs, _, stats = sub_gen.forward_inference(current_data, return_metrics=return_metrics)
                
                all_outputs.append(outputs)
                
                # Accumulate stats
                for stat_key, stat_val in stats.items():
                    all_stats[f"sub_{i}_{stat_key}"] = stat_val
                
                # Update data with current predictions for next sub-generator
                if i < 5:  # Don't need to update after last sub-generator
                    if "grasps_pred" in outputs:
                        current_data["grasps"] = outputs["grasps_pred"]
        else:
            # Parallel: all sub-generators process independently
            for i, sub_gen in enumerate(self.sub_generators):
                logger.debug(f"Inference sub-generator {i+1}/6 (DoF: {self.dof_names[i]})")
                
                outputs, _, stats = sub_gen.forward_inference(data, return_metrics=return_metrics)
                all_outputs.append(outputs)
                
                for stat_key, stat_val in stats.items():
                    all_stats[f"sub_{i}_{stat_key}"] = stat_val
        
        # Combine outputs from all sub-generators
        combined_outputs = self._combine_outputs(all_outputs, data)
        
        return combined_outputs, {}, all_stats

    def infer(self, data, return_metrics=False):
        """Inference method for generating nested grasps.
        
        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute and return evaluation metrics
            
        Returns:
            tuple: (outputs, losses, stats) containing generated grasps and optional metrics
        """
        return self.forward_inference(data, return_metrics=return_metrics)

    def _combine_outputs(self, all_outputs: List[Dict], data: Dict) -> Dict:
        """Combine outputs from all sub-generators into final 36-DoF output.
        
        Args:
            all_outputs: List of output dictionaries from each sub-generator
            data: Original input data
            
        Returns:
            Dict: Combined output dictionary with 36-DoF representations
        """
        device = data["points"].device
        
        # Extract predicted grasps from each sub-generator
        sub_grasps = [outputs["grasps_pred"] for outputs in all_outputs]
        
        # Combine grasps - for now, we concatenate the pose representations
        # This creates a 36-DoF representation (6 sub-generators × 6 base DoFs each)
        num_objects = len(sub_grasps[0])
        combined_grasps = []
        
        for obj_idx in range(num_objects):
            # Get grasps for this object from all sub-generators
            obj_sub_grasps = [sub_grasps[i][obj_idx] for i in range(6)]
            
            # Stack along a new dimension to preserve the nested structure
            # Shape: [num_grasps, 6_sub_gens, 4, 4]
            combined = torch.stack(obj_sub_grasps, dim=1)
            combined_grasps.append(combined)
        
        # Create output dictionary
        combined_outputs = {
            "grasps_pred": combined_grasps,  # List of [num_grasps, 6, 4, 4] tensors
            "sub_generator_outputs": all_outputs,  # Preserve individual outputs for analysis
            "grasp_confidence": torch.mean(torch.stack([outputs["grasp_confidence"] for outputs in all_outputs]), dim=0),
            "grasping_masks": all_outputs[0]["grasping_masks"],  # Use first sub-generator's masks
            "grasp_contacts": all_outputs[0]["grasp_contacts"],
            "instance_masks": all_outputs[0]["instance_masks"],
        }
        
        # Include per-iteration outputs if available
        if "grasps_per_iteration" in all_outputs[0]:
            # Stack iteration outputs from all sub-generators
            combined_outputs["grasps_per_iteration"] = [
                outputs["grasps_per_iteration"] for outputs in all_outputs
            ]
        
        # Include likelihood if available
        if "likelihood" in all_outputs[0]:
            # Average likelihoods across sub-generators
            combined_outputs["likelihood"] = torch.mean(
                torch.stack([outputs["likelihood"] for outputs in all_outputs]), dim=0
            )
        
        return combined_outputs
