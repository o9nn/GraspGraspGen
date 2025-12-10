import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import grasp_gen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from grasp_gen.models.nested_generator import NestedGraspGenGenerator
from grasp_gen.models.nested_grasp_gen import NestedGraspGen


class TestNestedGraspGenGenerator:
    """Test suite for NestedGraspGenGenerator model."""
    
    def test_initialization(self):
        """Test that NestedGraspGenGenerator initializes correctly."""
        # Create a simple config
        config = {
            "num_embed_dim": 256,
            "num_obs_dim": 512,
            "diffusion_embed_dim": 512,
            "image_size": 256,
            "num_diffusion_iters": 10,  # Small for testing
            "num_diffusion_iters_eval": 10,
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
            "num_grasps_per_object": 5,
            "checkpoint_object_encoder_pretrained": None,
        }
        
        # Create nested model with 6 identical sub-generators
        nested_model = NestedGraspGenGenerator(
            sub_generator_configs=[config] * 6,
            combine_strategy="hierarchical",
            grasp_repr="r3_6d"
        )
        
        # Check that 6 sub-generators were created
        assert len(nested_model.sub_generators) == 6
        
        # Check output dimension: 6 sub-generators * 9 (base output for r3_6d)
        assert nested_model.output_dim == 54  # 6 * 9
        assert nested_model.base_output_dim == 9
        
        # Check DoF names
        assert len(nested_model.dof_names) == 6
        assert nested_model.dof_names == ['x', 'y', 'z', 'rot_1', 'rot_2', 'rot_3']
    
    def test_initialization_r3_so3(self):
        """Test initialization with r3_so3 representation."""
        config = {
            "num_embed_dim": 256,
            "num_obs_dim": 512,
            "diffusion_embed_dim": 512,
            "image_size": 256,
            "num_diffusion_iters": 10,
            "num_diffusion_iters_eval": 10,
            "obs_backbone": "pointnet",
            "compositional_schedular": False,
            "loss_pointmatching": True,
            "loss_l1_pos": False,
            "loss_l1_rot": False,
            "grasp_repr": "r3_so3",
            "kappa": -1.0,
            "clip_sample": True,
            "beta_schedule": "squaredcos_cap_v2",
            "attention": "cat",
            "grid_size": 0.02,
            "gripper_name": "franka_panda",
            "pose_repr": "mlp",
            "num_grasps_per_object": 5,
            "checkpoint_object_encoder_pretrained": None,
        }
        
        nested_model = NestedGraspGenGenerator(
            sub_generator_configs=[config] * 6,
            grasp_repr="r3_so3"
        )
        
        # Check output dimension: 6 sub-generators * 6 (base output for r3_so3)
        assert nested_model.output_dim == 36  # 6 * 6 = 36-DoF as specified
        assert nested_model.base_output_dim == 6
    
    @patch('grasp_gen.models.nested_generator.GraspGenGenerator')
    def test_forward_train_hierarchical(self, mock_generator_class):
        """Test forward training pass with hierarchical strategy."""
        # Create mock sub-generators
        mock_sub_gens = []
        for i in range(6):
            mock_gen = MagicMock()
            # Mock the forward_train output
            mock_gen.forward_train.return_value = (
                {
                    "grasps_pred_mat": torch.randn(2, 5, 4, 4),
                    "pred_noise_pts_mat": torch.randn(2, 5, 4, 4),
                },
                {"noise_pred": (2.0, torch.tensor(0.5))},
                {"error_trans_l2": torch.tensor(0.1)}
            )
            mock_sub_gens.append(mock_gen)
        
        # Create nested model with mocked sub-generators
        nested_model = NestedGraspGenGenerator(
            sub_generator_configs=None,
            combine_strategy="hierarchical",
            grasp_repr="r3_6d"
        )
        nested_model.sub_generators = nn.ModuleList(mock_sub_gens)
        
        # Create dummy input data
        data = {
            "points": torch.randn(2, 1024, 3),
            "grasps": [torch.randn(5, 4, 4) for _ in range(2)]
        }
        
        # Forward pass
        outputs, losses, stats = nested_model.forward_train(data)
        
        # Check that all sub-generators were called
        assert all(gen.forward_train.called for gen in mock_sub_gens)
        
        # Check that we have losses from all sub-generators
        assert len(losses) == 6  # One loss per sub-generator
        
        # Check that we have stats from all sub-generators
        assert len(stats) == 6
        
        # Check that outputs were combined
        assert "grasps_pred" in outputs
        assert "sub_generator_outputs" in outputs
        assert len(outputs["sub_generator_outputs"]) == 6
    
    @patch('grasp_gen.models.nested_generator.GraspGenGenerator')
    def test_forward_inference(self, mock_generator_class):
        """Test forward inference pass."""
        # Create mock sub-generators
        mock_sub_gens = []
        for i in range(6):
            mock_gen = MagicMock()
            # Mock the forward_inference output
            mock_gen.forward_inference.return_value = (
                {
                    "grasps_pred": [torch.randn(5, 4, 4) for _ in range(2)],
                    "grasp_confidence": torch.randn(2),
                    "grasping_masks": torch.randn(2),
                    "grasp_contacts": torch.randn(2),
                    "instance_masks": torch.randn(2),
                },
                {},
                {"recall": torch.tensor(0.8)}
            )
            mock_sub_gens.append(mock_gen)
        
        # Create nested model
        nested_model = NestedGraspGenGenerator(
            sub_generator_configs=None,
            combine_strategy="hierarchical",
            grasp_repr="r3_6d"
        )
        nested_model.sub_generators = nn.ModuleList(mock_sub_gens)
        
        # Create dummy input data
        data = {
            "points": torch.randn(2, 1024, 3),
        }
        
        # Inference pass
        outputs, losses, stats = nested_model.forward_inference(data, return_metrics=True)
        
        # Check that all sub-generators were called
        assert all(gen.forward_inference.called for gen in mock_sub_gens)
        
        # Check outputs
        assert "grasps_pred" in outputs
        assert len(outputs["grasps_pred"]) == 2  # 2 objects in batch
        
        # Check that nested structure is preserved
        # Each grasp should have shape [num_grasps, 6_sub_gens, 4, 4]
        assert outputs["grasps_pred"][0].shape[1] == 6
    
    def test_combine_outputs(self):
        """Test output combination from sub-generators."""
        nested_model = NestedGraspGenGenerator(
            sub_generator_configs=None,
            combine_strategy="hierarchical",
            grasp_repr="r3_6d"
        )
        
        # Create mock outputs from sub-generators
        all_outputs = []
        for i in range(6):
            # Create likelihood tensor based on index for variety in test data
            if i == 0:
                likelihood = torch.randn(2, 5, 1)
            elif i == 1:
                likelihood = torch.randn(2, 3, 1)
            else:
                likelihood = torch.randn(2, 5, 1)
            
            outputs = {
                "grasps_pred": [torch.randn(5, 4, 4), torch.randn(3, 4, 4)],
                "grasp_confidence": torch.randn(2),
                "grasping_masks": torch.randn(2),
                "grasp_contacts": torch.randn(2),
                "instance_masks": torch.randn(2),
                "likelihood": likelihood,
            }
            all_outputs.append(outputs)
        
        data = {"points": torch.randn(2, 1024, 3)}
        
        # Combine outputs
        combined = nested_model._combine_outputs(all_outputs, data)
        
        # Check combined structure
        assert "grasps_pred" in combined
        assert len(combined["grasps_pred"]) == 2  # 2 objects
        
        # Check nested structure: [num_grasps, 6_sub_gens, 4, 4]
        assert combined["grasps_pred"][0].shape[1] == 6
        assert combined["grasps_pred"][0].shape[2] == 4
        assert combined["grasps_pred"][0].shape[3] == 4
        
        # Check that sub-generator outputs are preserved
        assert "sub_generator_outputs" in combined
        assert len(combined["sub_generator_outputs"]) == 6


class TestNestedGraspGen:
    """Test suite for NestedGraspGen combined model."""
    
    @patch('grasp_gen.models.nested_grasp_gen.NestedGraspGenGenerator')
    @patch('grasp_gen.models.nested_grasp_gen.GraspGenDiscriminator')
    def test_initialization(self, mock_disc_class, mock_gen_class):
        """Test NestedGraspGen initialization."""
        from omegaconf import DictConfig
        
        gen_cfg = DictConfig({"grasp_repr": "r3_6d"})
        disc_cfg = DictConfig({"num_obs_dim": 512})
        
        model = NestedGraspGen(gen_cfg, disc_cfg)
        
        # Check that from_config was called
        mock_gen_class.from_config.assert_called_once()
        mock_disc_class.from_config.assert_called_once()
    
    @patch('grasp_gen.models.nested_grasp_gen.NestedGraspGenGenerator')
    @patch('grasp_gen.models.nested_grasp_gen.GraspGenDiscriminator')
    def test_forward(self, mock_disc_class, mock_gen_class):
        """Test forward pass of combined model."""
        from omegaconf import DictConfig
        
        # Create mock generator
        mock_generator = MagicMock()
        mock_generator.infer.return_value = (
            {
                "grasps_pred": [torch.randn(5, 6, 4, 4)],
                "sub_generator_outputs": [
                    {"grasps_pred": [torch.randn(5, 4, 4)]} for _ in range(6)
                ],
                "grasp_confidence": torch.randn(1),
            },
            {},
            {"recall": torch.tensor(0.8)}
        )
        mock_gen_class.from_config.return_value = mock_generator
        
        # Create mock discriminator
        mock_discriminator = MagicMock()
        mock_discriminator.infer.return_value = (
            {"scores": torch.randn(5)},
            {},
            {}
        )
        mock_disc_class.from_config.return_value = mock_discriminator
        
        gen_cfg = DictConfig({"grasp_repr": "r3_6d"})
        disc_cfg = DictConfig({"num_obs_dim": 512})
        
        model = NestedGraspGen(gen_cfg, disc_cfg)
        
        # Forward pass
        data = {"points": torch.randn(1, 1024, 3)}
        outputs, losses, stats = model.forward(data)
        
        # Check that both generator and discriminator were called
        mock_generator.infer.assert_called_once()
        mock_discriminator.infer.assert_called_once()
        
        # Check that discriminator scores are in output
        assert "scores" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
