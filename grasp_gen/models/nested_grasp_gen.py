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
from omegaconf import DictConfig

from grasp_gen.models.nested_generator import NestedGraspGenGenerator
from grasp_gen.models.discriminator import GraspGenDiscriminator
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


class NestedGraspGen(nn.Module):
    """Combined nested model that uses hierarchical 36-DoF generation and discriminative evaluation.
    
    This class combines a NestedGraspGenGenerator with a GraspGenDiscriminator to both
    generate and evaluate grasps in a single pipeline with 36 degrees of freedom.
    
    Args:
        nested_generator_cfg (DictConfig): Configuration for the nested grasp generator
        grasp_discriminator_cfg (DictConfig): Configuration for the grasp discriminator
    """

    def __init__(
        self, nested_generator_cfg: DictConfig, grasp_discriminator_cfg: DictConfig
    ):
        super(NestedGraspGen, self).__init__()
        self.nested_generator = NestedGraspGenGenerator.from_config(nested_generator_cfg)
        self.grasp_discriminator = GraspGenDiscriminator.from_config(
            grasp_discriminator_cfg
        )

    def forward(self, data):
        """Forward pass combining nested generation and discrimination.
        
        Args:
            data: Input data dictionary containing point clouds
            
        Returns:
            tuple: (outputs, losses, stats) containing generated and scored grasps
        """
        outputs, _, stats = self.nested_generator.infer(data, return_metrics=True)
        
        # For discrimination, we need to evaluate each sub-generator's output
        # Use the final combined grasps for scoring
        data_for_disc = data.copy()
        
        # Extract the last sub-generator's grasps (most refined)
        if "sub_generator_outputs" in outputs:
            last_sub_outputs = outputs["sub_generator_outputs"][-1]
            data_for_disc["grasps_pred"] = last_sub_outputs["grasps_pred"]
        else:
            # Fallback to combined output
            # Need to reshape nested structure for discriminator
            combined_grasps = outputs["grasps_pred"]
            # Take the last sub-generator dimension
            data_for_disc["grasps_pred"] = [cg[:, -1, :, :] for cg in combined_grasps]
        
        data_for_disc.update({"grasp_key": "grasps_pred"})
        disc_outputs, _, _ = self.grasp_discriminator.infer(data_for_disc)
        
        # Merge discriminator outputs with nested generator outputs
        outputs.update(disc_outputs)
        
        return outputs, {}, stats

    def infer(self, data, return_metrics=False):
        """Inference method for generating and evaluating nested grasps.
        
        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute evaluation metrics
            
        Returns:
            tuple: (outputs, losses, stats) containing generated and scored grasps with metrics
        """
        return self.forward(data)

    @classmethod
    def from_config(
        cls, nested_generator_cfg: DictConfig, grasp_discriminator_cfg: DictConfig
    ):
        """Creates a NestedGraspGen instance from configuration objects.
        
        Args:
            nested_generator_cfg (DictConfig): Configuration for the nested grasp generator
            grasp_discriminator_cfg (DictConfig): Configuration for the grasp discriminator
            
        Returns:
            NestedGraspGen: Instantiated nested model
        """
        return NestedGraspGen(nested_generator_cfg, grasp_discriminator_cfg)

    def load_state_dict(
        self, nested_generator_ckpt_filepath: str, grasp_discriminator_ckpt_filepath: str
    ):
        """Loads pretrained weights for both nested generator and discriminator.
        
        Args:
            nested_generator_ckpt_filepath (str): Path to nested generator checkpoint
            grasp_discriminator_ckpt_filepath (str): Path to discriminator checkpoint
        """
        logger.info(
            f"Loading nested generator checkpoint from {nested_generator_ckpt_filepath}"
        )
        ckpt = torch.load(nested_generator_ckpt_filepath, map_location="cpu")
        self.nested_generator.load_state_dict(ckpt["model"])

        logger.info(
            f"Loading discriminator checkpoint from {grasp_discriminator_ckpt_filepath}"
        )
        ckpt = torch.load(grasp_discriminator_ckpt_filepath, map_location="cpu")
        self.grasp_discriminator.load_state_dict(ckpt["model"])
