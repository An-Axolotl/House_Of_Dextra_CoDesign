# --------------------------------------------------------
# Codesign Hand: Reorientation Environment
# Based on LEAP Hand reorientation environment
# --------------------------------------------------------

from __future__ import annotations

import torch
from collections.abc import Sequence
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab.sensors import ContactSensor, ContactSensorCfg

import os
import re

from .reorientation_env_cfg import ReorientationEnvCfg
from ....utils import adr_utils, obs_utils
from ....utils.adr import LeapHandADR
from ....utils.morph_vector import parse_morph_from_filename, _digits_to_links, build_hand_morph_obs
from ....utils.usd_refs import (
    get_current_stage,
    get_env_object_usd_path,
    get_env_robot_usd_path,
    normalize_asset_path,
)

from ....assets.objects import OBJECT_USDS, OBJECT_TYPE_MAP, OBJECT_TYPE_TENSOR

class ReorientationEnv(DirectRLEnv):
    cfg: ReorientationEnvCfg

    def __init__(self, cfg: ReorientationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()
        # self._build_action_mask_from_morph()
        self._build_action_mask_and_hand_tracking()

        # Build finger bodies list dynamically
        self._build_finger_bodies_from_morphology()
        
        # Determine which contact sensors are actually active
        self._determine_active_contact_sensors()
        
        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] += 0.01
        
        # continuous z-axis rotation parameters
        self.target_z_angle = torch.full((self.num_envs,), 2 * math.pi / self.cfg.z_rotation_steps, dtype=torch.float, device=self.device)
        
        # default goal positions and rotations
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0  # Identity quaternion
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # track initial object position for position stability penalty
        self.initial_object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # define default joint position
        self.override_default_joint_pos = torch.tensor(self.cfg.default_joint_pos, device=self.device).repeat(self.num_envs, 1)

        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_rot[:, 0] = 1.0 

        # initialize history tensor
        self.obs_hist_buf = torch.zeros((self.num_envs, self.cfg.obs_per_timestep, self.cfg.hist_len), device=self.device, dtype=torch.float)            
        # self.output_obs_hist_buf = torch.zeros(self.cfg.scene.num_envs, self.cfg.observation_space // self.cfg.hist_len, self.cfg.hist_len, device=self.cfg.sim.device, dtype=torch.float)
            
        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Check for GHS evaluation mode from environment variable
        if "CODESIGN_GHS_EVALUATION" in os.environ:
            self.cfg.ghs_evaluation_mode = bool(int(os.environ.get("CODESIGN_GHS_EVALUATION", "0")))
            print(f"[ghs-eval] GHS evaluation mode: {self.cfg.ghs_evaluation_mode}")
        
        # Set episode lengths based on mode
        if self.cfg.ghs_evaluation_mode:
            # Fixed episode lengths for fair evaluation
            fixed_steps = int(self.cfg.fixed_eval_time_s / (self.cfg.sim.dt * self.cfg.decimation))
            self.randomized_episode_lengths = torch.full(
                (self.num_envs,), fixed_steps, dtype=torch.int32, device=self.device
            )
            print(f"[ghs-eval] Using fixed episode length: {fixed_steps} steps ({self.cfg.fixed_eval_time_s}s)")
        else:
            # Randomized episode lengths for training
            self.randomized_episode_lengths = torch.randint(
                int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)), 
                self.max_episode_length + 1, 
                (self.num_envs,), 
                dtype=torch.int32, 
                device=self.device
            )
            print(f"[training] Using randomized episode lengths: {self.randomized_episode_lengths.min()}-{self.randomized_episode_lengths.max()} steps")

        if self.cfg.enable_adr:
            self.leap_adr = LeapHandADR(self.event_manager, 
                                        self.cfg.adr_cfg_dict, 
                                        self.cfg.adr_custom_cfg_dict)
            self.step_since_last_dr_change = 0
            self.leap_adr.set_num_increments(self.cfg.starting_adr_increments)
            adr_utils.init_adr_obs_act_noise(self)
            
            # Update observation buffer for latency
            self.obs_hist_buf = torch.zeros(
                self.num_envs, 
                self.cfg.obs_per_timestep, 
                self.cfg.hist_len + self.cfg.obs_max_latency, 
                device=cfg.sim.device, 
                dtype=torch.float
            )
            self.obs_latency = torch.empty((self.num_envs, self.cfg.obs_per_timestep), device=self.cfg.sim.device)
            self.act_latency = torch.empty((self.num_envs, self.cfg.action_space), device=self.cfg.sim.device)
            self.act_hist_buf = torch.zeros(
                self.num_envs, 
                self.cfg.action_space, 
                self.cfg.act_max_latency + 1, 
                device=self.cfg.sim.device, 
                dtype=torch.float
            )
            
            if self.cfg.debug:
                print("Starting ADR ranges: ")
                print(self.leap_adr.print_params())

        # Initialize extras if not already present
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        
        # Setup contact sensors before scene initialization
        self._setup_dynamic_contact_sensors_config()
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        self.get_object_indices()
        self._build_env_morph_vec()
        
    def force_aggregate_incomplete_episodes(self):
        """
        Aggregate metrics for ALL environments, even if episodes haven't completed.
        Call this at the end of evaluation (after timeout) to ensure metrics are captured.
        """
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        d_idx = self.env_to_design_idx[all_env_ids]
        valid = d_idx.ge(0)
        
        if valid.any():
            d = d_idx[valid]
            envs = all_env_ids[valid]
            
            # Only aggregate if there's actually accumulated data (ep_len > 0)
            has_data = self._ep_len[envs] > 0
            
            if has_data.any():
                d_with_data = d[has_data]
                envs_with_data = envs[has_data]
                
                # Aggregate into per-design totals
                self._design_total_episodes.index_add_(0, d_with_data, torch.ones_like(d_with_data, dtype=torch.long))
                self._design_total_reward.index_add_(0, d_with_data, self._ep_reward[envs_with_data])
                self._design_success_count.index_add_(0, d_with_data, self._ep_success[envs_with_data])
                self._design_contact_sum.index_add_(0, d_with_data, self._ep_contact[envs_with_data])
                self._design_grasp_sum.index_add_(0, d_with_data, self._ep_grasp[envs_with_data])
                self._design_pose_pen_sum.index_add_(0, d_with_data, self._ep_pose[envs_with_data])
                self._design_torque_pen_sum.index_add_(0, d_with_data, self._ep_torque[envs_with_data])
                self._design_stab_pen_sum.index_add_(0, d_with_data, self._ep_stab[envs_with_data])
                self._design_tipdist_sum.index_add_(0, d_with_data, self._ep_tipdist[envs_with_data])
                self._design_total_steps.index_add_(0, d_with_data, self._ep_len[envs_with_data])
                self._design_hit_count.index_add_(0, d_with_data, self._ep_hits[envs_with_data])
                self._design_omega_sum.index_add_(0, d_with_data, self._ep_omega_sum[envs_with_data])
                self._design_contact_steps.index_add_(0, d_with_data, self._ep_contact_steps[envs_with_data])
                
                print(f"[force_aggregate] Aggregated metrics from {has_data.sum().item()} environments with incomplete episodes")
        
    def _rollup_done_episodes(self, done_env_ids: torch.Tensor):
        if done_env_ids.numel() == 0:
            return
        # map env -> design indices
        d_idx = self.env_to_design_idx[done_env_ids]  # shape [K]
        # index_add into per-design accumulators
        self._design_total_steps.index_add_(0, d_idx, self._ep_len[done_env_ids])
        self._design_hit_count.index_add_(0, d_idx, self._ep_hits[done_env_ids])
        self._design_total_episodes.index_add_(0, d_idx, torch.ones_like(d_idx, dtype=torch.long))
        # reset per-env counters
        self._ep_len[done_env_ids]  = 0
        self._ep_hits[done_env_ids] = 0

    def print_hand_design_performance(self):
        """Print performance summary for each hand design with raw omega metrics"""
        summary = self.get_hand_design_metrics_summary_gpu()
        
        if not summary:
            print("[hand-metrics] No hand design metrics available")
            return
        
        print(f"\n[hand-metrics] Performance Summary for {len(summary)} Hand Designs:")
        print("=" * 80)
        
        # Sort by omega score (best first)
        sorted_designs = sorted(summary.items(), key=lambda x: x[1].get('omega_score', 0.0), reverse=True)
        
        for rank, (design_id, metrics) in enumerate(sorted_designs, 1):
            print(f"\n{rank}. Design: {design_id}")
            
            # Training metrics
            print(f"   🎯 Training Score (ω): {metrics.get('omega_score', 0.0):.4f}")
            
            # Raw rotation metrics for research
            print(f"   📊 Raw Rotation Metrics:")
            print(f"      |ω_z|: {metrics.get('raw_omega_z_rad_per_sec', 0.0):.3f} rad/s = {metrics.get('raw_omega_z_deg_per_sec', 0.0):.1f} deg/s")
            print(f"      RPS: {metrics.get('rps_from_omega', 0.0):.3f} rev/s = {metrics.get('raw_omega_z_rpm', 0.0):.1f} RPM")
            print(f"      Steps evaluated: {metrics.get('contact_steps', 0)} ({metrics.get('step_fraction_evaluated', 0.0):.1%} of total)")
            
            # Comparison metrics
            print(f"   📈 Goal-based RPS: {metrics.get('turns_per_second', 0.0):.4f} (discrete hits)")
            
            # Environment info
            print(f"   🔧 Environments: {metrics['env_count']} ({metrics['env_ids'][:3]}{'...' if len(metrics['env_ids']) > 3 else ''})")
            print(f"      Episodes: {metrics['total_episodes']}, Time: {metrics.get('evaluation_time_s', 0):.1f}s")
            print(f"      Reward: {metrics['average_reward']:.3f}")
        
        print("=" * 80)
        
    def get_hand_design_metrics_summary_gpu(self):
        if not hasattr(self, "design_ids"): return {}
        eps = self._design_total_episodes.clamp_min(1).float()

        # read a cap from cfg (fallback reasonable default)
        reward_cap = float(getattr(self.cfg, "reward_cap", 30000.0))  # tune to your reward scale

        out = {}
        for i, did in enumerate(self.design_ids):
            tepi = int(self._design_total_episodes[i].item())
            if tepi == 0:
                continue

            avg_reward = float((self._design_total_reward[i] / eps[i]).item())

            steps_i = float(self._design_total_steps[i].item())

            # Choose time calculation based on evaluation mode
            if self.cfg.ghs_evaluation_mode:
                # Use FIXED evaluation time for fair comparison
                time_s = self.cfg.fixed_eval_time_s
            else:
                # Use actual simulated time for training metrics
                sec_per_step = float(self.cfg.sim.dt * self.cfg.decimation)
                time_s = max(steps_i * sec_per_step, 1e-6)

            hits_i = float(self._design_hit_count[i].item())
            z_steps = float(max(self.cfg.z_rotation_steps, 1.0))
            turns = hits_i / z_steps
            rps = turns / time_s
            
            # Omega based continuous scoring
            omega_sum = float(self._design_omega_sum[i].item())
            contact_steps = float(max(self._design_contact_steps[i].item(), 1))
            avg_abs_omega_z = omega_sum / contact_steps  # mean |ω_z| while in contact
            rps_from_omega = avg_abs_omega_z / (2.0 * 3.14159)  # convert to rev/s

            rps_cap = 3.0  # sort of an achievable upper bound
            omega_score = max(0.0, min(rps_from_omega / rps_cap, 1.0 - 1e-6))  # clamp to [0, 1-ε)
        
            # normalized reward in [0,1)
            # negatives → 0; clamp upper by reward_cap
            norm_reward = max(0.0, min(avg_reward / reward_cap, 1.0 - 1e-6))

            out[did] = {
                "env_count": len(self.hand_design_to_envs.get(did, [])),
                "total_episodes": tepi,
                "total_reward": float(self._design_total_reward[i].item()),
                "success_count": int(self._design_success_count[i].item()),
                "average_reward": avg_reward,
                "norm_reward": float(norm_reward),
                "reward_cap": float(reward_cap),
                "recent_avg_reward": avg_reward,
                "success_rate": float((self._design_success_count[i].float() / eps[i]).item()),
                "avg_contact_reward": float((self._design_contact_sum[i] / eps[i]).item()),
                "avg_grasp_reward": float((self._design_grasp_sum[i] / eps[i]).item()),
                "avg_pose_diff_penalty": float((self._design_pose_pen_sum[i] / eps[i]).item()),
                "avg_torque_penalty": float((self._design_torque_pen_sum[i] / eps[i]).item()),
                "avg_position_stability_penalty": float((self._design_stab_pen_sum[i] / eps[i]).item()),
                "avg_fingertip_distance_penalty": float((self._design_tipdist_sum[i] / eps[i]).item()),
                "env_ids": self.hand_design_to_envs.get(did, []),
                "total_steps": int(steps_i),
                "total_hits": int(hits_i),
                "turns_per_second": float(rps),
                "turns_per_minute": float(rps * 60.0),
                "raw_omega_z_rad_per_sec": float(avg_abs_omega_z),           # Raw |ω_z| in rad/s
                "raw_omega_z_deg_per_sec": float(avg_abs_omega_z * 57.2958), # Raw |ω_z| in deg/s  
                "raw_omega_z_rpm": float(rps_from_omega * 60.0),            # Raw omega in RPM
                "rps_from_omega": float(rps_from_omega),                     # Raw omega in RPS
                "omega_score": float(omega_score),                          # Processed [0,1) training signal
                "omega_cap_rps": float(rps_cap),                            # Cap used for normalization
                "contact_steps": int(contact_steps),                        # Total steps (since no contact gating)
                "step_fraction_evaluated": float(contact_steps / max(steps_i, 1)), # Should be ~1.0 now
                "evaluation_mode": self.cfg.ghs_evaluation_mode,
                "evaluation_time_s": float(time_s),
            }

        return out
    
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        pose_diff_penalty = ((self.cur_targets[:, self.actuated_dof_indices] - self.override_default_joint_pos) ** 2).sum(-1)
        torque_penalty = (self.hand.data.computed_torque ** 2).sum(-1)
        
        masked_actions = self.actions if not hasattr(self, "env_action_mask") else self.actions * self.env_action_mask

        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
            position_stability_penalty,
            contact_reward,
            grasp_reward,
            fingertip_distance_penalty,
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.fingertip_pos,
            self.fingertip_net_contact_value,  # finger-to-finger contact (penalty)
            self.fingertip_object_contact_value,  # finger-to-object contact (reward)
            self.individual_finger_object_contacts,  # individual finger contacts for grasp reward
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.object_linvel,
            self.object_angvel,
            self.initial_object_pos,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            masked_actions,
            self.cfg.action_penalty_scale,
            pose_diff_penalty, 
            self.cfg.pose_diff_penalty_scale,
            torque_penalty,
            self.cfg.torque_penalty_scale,
            self.cfg.position_stability_penalty_scale,
            self.cfg.position_stability_threshold,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            self.cfg.contact_reward_scale,
            self.cfg.min_contact_reward,
            self.cfg.grasp_reward_scale,
            self.cfg.fingertip_distance_penalty_scale,
            self.cfg.approach_reward_scale,
        )

        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean() / self.cfg.z_rotation_steps
        self.extras["log"]["pose_diff_penalty"] = pose_diff_penalty.mean() 
        self.extras["log"]["torque_info"] = torque_penalty.mean()
        self.extras["log"]["position_stability_penalty"] = position_stability_penalty.mean()
        self.extras["log"]["contact_reward"] = contact_reward.mean()
        self.extras["log"]["grasp_reward"] = grasp_reward.mean()
        self.extras["log"]["fingertip_distance_penalty"] = fingertip_distance_penalty.mean()
        self.extras["log"]['object_linvel'] = torch.norm(self.object_linvel, p=1, dim=-1).mean()
        self.extras["log"]['roll'] = self.object_angvel[:, 0].mean()
        self.extras["log"]['pitch'] = self.object_angvel[:, 1].mean()
        self.extras["log"]['yaw'] = self.object_angvel[:, 2].mean()

        abs_omega_z = torch.abs(self.object_angvel[:, 2])       # [N] float

        # Track omega regardless of contact.
        self._ep_omega_sum += abs_omega_z.detach()  # No contact gating
        self._ep_contact_steps += torch.ones_like(abs_omega_z, dtype=torch.long)  # Count all steps

        # Log episode length statistics
        self.extras["log"]["avg_episode_length_s"] = (self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation).mean()
        self.extras["log"]["min_episode_length_s"] = (self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation).min()
        self.extras["log"]["max_episode_length_s"] = (self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation).max()

        # vectorized per-step accumulation (stay on GPU; detach to avoid autograd)
        self._ep_reward  += total_reward.detach()
        self._ep_success  = torch.maximum(self._ep_success, self.successes.long())
        self._ep_contact += contact_reward.detach()
        self._ep_grasp   += grasp_reward.detach()
        self._ep_pose    += pose_diff_penalty.detach()
        self._ep_torque  += torque_penalty.detach()
        self._ep_stab    += position_stability_penalty.detach()
        self._ep_tipdist += fingertip_distance_penalty.detach()
        self._ep_len += 1

        if self.cfg.enable_adr:
            adr_criteria = ((self.consecutive_successes / self.cfg.z_rotation_steps) / 
                        (self.randomized_episode_lengths.float().mean() * 
                            self.cfg.sim.dt * self.cfg.decimation)).float().mean()
            self.extras["log"]["adr_criteria"] = adr_criteria

        # update goal if the goal has been reached            
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            # one "hit" per env that reached the next z-rotation goal
            self._ep_hits[goal_env_ids] += 1
            self._update_continuous_z_rotation(goal_env_ids)
            self.reset_goal_buf[goal_env_ids] = 0

        return total_reward

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        
        if self.cfg.enable_adr:
            # Add action noise
            hand_noise = self.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise")
            if hand_noise > 0:
                noise = torch.randn_like(actions) * hand_noise
                self.actions = actions + noise
            # Apply action latency
            self.actions = obs_utils.create_action_latency(self, self.actions)
            
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        
        # apply morphology-based mask (zeros out dead DOFs per env)
        if hasattr(self, "env_action_mask"):
            self.actions = self.actions * self.env_action_mask

    def _apply_action(self) -> None:

        if self.cfg.action_type=="relative":
            if self.cfg.debug:
                self.actions *= 0
                
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                targets,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        elif self.cfg.action_type=="absolute":
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                self.cur_targets[:, self.actuated_dof_indices],
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        else:
            raise ValueError(f"Unsupported action type: {self.cfg.action_type}. Must be relative or absolute.")

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        if self.cfg.enable_adr:
            adr_utils.apply_object_wrench(self, self.object, "object")

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _update_continuous_z_rotation(self, goal_env_ids):        
        # create quaternion for z-axis rotation
        add_rot = quat_from_angle_axis(self.target_z_angle, self.z_unit_tensor)
        self.goal_rot[goal_env_ids] = quat_mul(add_rot[goal_env_ids], self.goal_rot[goal_env_ids])
        
        # update goal markers
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _get_observations(self) -> dict:
        frame = unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        if self.cfg.store_cur_actions:
            frame = torch.cat((frame, self.cur_targets[:]), dim=-1)   # [N, 40]

        # Add object position to the frame
        if self.cfg.enable_privileged_obs:
            frame = torch.cat((frame, self.object_pos), dim=-1)  # [N, 40 + 3] or [N, 43]

        # write into history
        self.obs_hist_buf[:, :, :-1] = self.obs_hist_buf[:, :, 1:]
        self.obs_hist_buf[:, :, -1]  = frame

        # stack history -> [N, <obs-per-frame>*hist_len]
        hist = self.obs_hist_buf.transpose(1, 2).reshape(self.num_envs, -1)
        
        # existing: object type OH
        obj_oh = self.env_obj_type_one_hot

        # Morphology vector (binary presence + normalized grammar scales).
        if hasattr(self, 'env_morph_vec') and self.env_morph_vec is not None:
            morph_vec = self.env_morph_vec
        else:
            morph_vec = torch.zeros((self.num_envs, self.cfg.MORPH_DIM), device=self.device)

        obs = torch.cat((hist, obj_oh, morph_vec), dim=-1)

        if self.cfg.debug and not hasattr(self, "_obs_shape_printed"):
            print("obs dim =", obs.shape[-1])
            self._obs_shape_printed = True
        
        return {"policy": obs.float()}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Normal timeout logic
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        if self.cfg.ghs_evaluation_mode:
            # During GHS evaluation: disable early termination for fair comparison
            out_of_reach = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        else:
            # Normal training: allow early termination
            goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
            out_of_reach = (goal_dist >= self.cfg.fall_dist) | (self.object_pos[:, 2] < self.cfg.fall_height)
            
            # Check if object is flipped
            obj_z = matrix_from_quat(self.object_rot)[:, :, 2]
            goal_z = matrix_from_quat(self.goal_rot)[:, :, 2]
            diff = torch.sum(obj_z * goal_z, dim=1)
            flipped = (torch.abs(diff) < 0.5)
            
            out_of_reach = out_of_reach | flipped

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        d_idx   = self.env_to_design_idx[env_ids]
        valid   = d_idx.ge(0)
        if valid.any():
            d = d_idx[valid]
            self._design_total_episodes.index_add_(0, d, torch.ones_like(d, dtype=torch.long))
            self._design_total_reward.index_add_(0, d, self._ep_reward[env_ids][valid])
            self._design_success_count.index_add_(0, d, self._ep_success[env_ids][valid])
            self._design_contact_sum.index_add_(0, d, self._ep_contact[env_ids][valid])
            self._design_grasp_sum.index_add_(0, d, self._ep_grasp[env_ids][valid])
            self._design_pose_pen_sum.index_add_(0, d, self._ep_pose[env_ids][valid])
            self._design_torque_pen_sum.index_add_(0, d, self._ep_torque[env_ids][valid])
            self._design_stab_pen_sum.index_add_(0, d, self._ep_stab[env_ids][valid])
            self._design_tipdist_sum.index_add_(0, d, self._ep_tipdist[env_ids][valid])
            self._design_total_steps.index_add_(0, d, self._ep_len[env_ids][valid])
            self._design_hit_count.index_add_(0, d, self._ep_hits[env_ids][valid])
            self._design_omega_sum.index_add_(0, d, self._ep_omega_sum[env_ids][valid])
            self._design_contact_steps.index_add_(0, d, self._ep_contact_steps[env_ids][valid])
            
        # clear per-env accumulators for the reset envs
        self._ep_reward[env_ids]  = 0
        self._ep_success[env_ids] = 0
        self._ep_contact[env_ids] = 0
        self._ep_grasp[env_ids]   = 0
        self._ep_pose[env_ids]    = 0
        self._ep_torque[env_ids]  = 0
        self._ep_stab[env_ids]    = 0
        self._ep_tipdist[env_ids] = 0
        self._ep_len[env_ids]     = 0
        self._ep_omega_sum[env_ids] = 0
        self._ep_contact_steps[env_ids] = 0
        
        # Calculate ADR criteria if enabled
        if self.cfg.enable_adr:
            adr_criteria = ((self.consecutive_successes.float().mean() / self.cfg.z_rotation_steps) / 
                        (self.randomized_episode_lengths.float().mean() * 
                            self.cfg.sim.dt * self.cfg.decimation)).float().mean()

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # Set episode lengths based on evaluation mode
        if self.cfg.ghs_evaluation_mode:
            # Fixed episode lengths for fair evaluation
            fixed_steps = int(self.cfg.fixed_eval_time_s / (self.cfg.sim.dt * self.cfg.decimation))
            self.randomized_episode_lengths[env_ids] = fixed_steps
        else:
            # Randomized episode lengths for training (existing behavior)
            self.randomized_episode_lengths[env_ids] = torch.randint(
                int(self.cfg.min_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)), 
                self.max_episode_length + 1, 
                (len(env_ids),), 
                dtype=torch.int32, 
                device=self.device
            )

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        dof_pos = self.override_default_joint_pos[env_ids] 
        dof_vel = self.hand.data.default_joint_vel[env_ids] 
        
        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        
        # After setting object_default_state, add ADR randomization
        if self.cfg.enable_adr:
            # Object spawn randomization
            x_width = self.leap_adr.get_custom_param_value("object_spawn", "x_width_spawn")
            y_width = self.leap_adr.get_custom_param_value("object_spawn", "y_width_spawn")
            x_rot = self.leap_adr.get_custom_param_value("object_spawn", "x_rotation")
            y_rot = self.leap_adr.get_custom_param_value("object_spawn", "y_rotation")
            z_rot = self.leap_adr.get_custom_param_value("object_spawn", "z_rotation")
            
            # Apply position randomization
            if x_width > 0 or y_width > 0:
                pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
                object_default_state[:, 0] += pos_noise[:, 0] * x_width
                object_default_state[:, 1] += pos_noise[:, 1] * y_width
            
            # Apply rotation randomization
            if x_rot > 0:
                x_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                x_rot_quat = quat_from_angle_axis(x_rot_noise * x_rot, self.x_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(x_rot_quat, object_default_state[:, 3:7])
                
            if y_rot > 0:
                y_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                y_rot_quat = quat_from_angle_axis(y_rot_noise * y_rot, self.y_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(y_rot_quat, object_default_state[:, 3:7])
                
            if z_rot > 0:
                z_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
                z_rot_quat = quat_from_angle_axis(z_rot_noise * z_rot, self.z_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(z_rot_quat, object_default_state[:, 3:7])
            
            # Joint spawn randomization
            joint_pos_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_pos_noise")
            joint_vel_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_vel_noise")
            
            if joint_pos_noise_width > 0:
                joint_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
                dof_pos += joint_pos_noise * joint_pos_noise_width
                
            if joint_vel_noise_width > 0:
                joint_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
                dof_vel += joint_vel_noise * joint_vel_noise_width

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # reset hand
        if hasattr(self, "env_action_mask"):
            dof_pos = dof_pos * self.env_action_mask[env_ids]

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.successes[env_ids] = 0

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        
        if self.cfg.enable_adr and len(env_ids) > 0:
            adr_utils.update_adr_obs_act_noise(self, env_ids)

            obs_latency_resets =  self.leap_adr.get_custom_param_value("obs_latency","latency") - torch.randint(0, self.cfg.obs_latency_rand + 1, (len(env_ids),1), device=self.cfg.sim.device)
            obs_latency_resets = torch.maximum(obs_latency_resets, torch.tensor(0))
            self.obs_latency[env_ids, :] = obs_latency_resets.expand(-1, self.cfg.obs_per_timestep)
            
            act_latency_resets = self.leap_adr.get_custom_param_value("action_latency","hand_latency") - torch.randint(0, self.cfg.act_latency_rand + 1, (len(env_ids), 1), device=self.cfg.sim.device)
            act_latency_resets = torch.maximum(act_latency_resets, torch.tensor(0))
            self.act_latency[env_ids, :] = act_latency_resets.expand(-1, self.cfg.action_space)
            
            self.extras["log"]["num_adr_increases"] = self.leap_adr.num_increments()
            
            if self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change and\
                (adr_criteria  >= self.cfg.min_rot_adr_coeff):
                self.step_since_last_dr_change = 0
                self.leap_adr.increase_ranges()
                self.leap_adr.print_params()
                self.consecutive_successes.fill_(0.0)
            else:
                self.step_since_last_dr_change += 1

            # update whether to apply wrench for the episode
            self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device) 
            self.apply_wrench = torch.where(
                torch.rand(self.num_envs, device=self.device) <= self.cfg.wrench_prob_per_rollout,
                True,
                False)

        # initialize goal rotation
        self._compute_intermediate_values()
        
        # Store initial object position for position stability penalty
        self.initial_object_pos[env_ids] = self.object_pos[env_ids].clone()
        
        r,p,y = euler_xyz_from_quat(self.object_rot[env_ids])
        r[:].fill_(0.0)
        p[:].fill_(0.0)
        self.goal_rot[env_ids] = quat_from_euler_xyz(r,p,y)

        self._update_continuous_z_rotation(env_ids)

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel 
        
        # ---- fingertip-to-fingertip contact (normal forces; filtered) for PENALTY ----
        deadband = 0.1  # N, tune as needed
        finger_to_finger_mags = []
        
        # Only process sensors that actually exist
        for finger_idx in self.active_contact_sensors:
            finger_num = finger_idx + 1
            sensor_name = f"fingertip_{finger_num}_contact"
            
            if sensor_name in self.scene.sensors:
                mat = self.scene.sensors[sensor_name].data.force_matrix_w
                if mat is not None:
                    # normal force magnitude per filtered partner
                    m = torch.linalg.norm(mat, dim=-1)             # [N, 1, 4]
                    m = torch.clamp(m - deadband, min=0.0)         # deadband
                    m = m.sum(dim=(-1, -2))                        # -> [N]
                    finger_to_finger_mags.append(m)
                else:
                    # If sensor exists but no contact data, add zeros
                    finger_to_finger_mags.append(torch.zeros(self.num_envs, device=self.device))
        
        if finger_to_finger_mags:
            tip_to_tip = torch.stack(finger_to_finger_mags, dim=1)              # [N, num_active_sensors]
            num_active = len(self.active_contact_sensors)
            self.fingertip_net_contact_value = 0.5 * tip_to_tip.sum(dim=1) / max(num_active, 1)
        else:
            # No contact sensors active
            self.fingertip_net_contact_value = torch.zeros(self.num_envs, device=self.device)
        
        self.extras["log"]["tip_tip_contact"] = self.fingertip_net_contact_value.mean()

        # ---- fingertip-to-object contact (normal forces) for REWARD ----
        # Store individual finger contact forces for more precise grasp reward calculation
        self.individual_finger_object_contacts = torch.zeros((self.num_envs, 5), device=self.device)
        
        for finger_idx in self.active_contact_sensors:
            finger_num = finger_idx + 1
            sensor_name = f"fingertip_{finger_num}_object_contact"
            
            if sensor_name in self.scene.sensors:
                mat = self.scene.sensors[sensor_name].data.force_matrix_w
                if mat is not None:
                    # normal force magnitude per filtered partner (object)
                    m = torch.linalg.norm(mat, dim=-1)             # [N, 1, 1] for single object
                    m = torch.clamp(m - deadband, min=0.0)         # deadband
                    m = m.sum(dim=(-1, -2))                        # -> [N]
                    self.individual_finger_object_contacts[:, finger_idx] = m
        
        # Sum contact forces from all fingertips touching the object (existing behavior)
        self.fingertip_object_contact_value = self.individual_finger_object_contacts.sum(dim=1)
        
        self.extras["log"]["finger_object_contact"] = self.fingertip_object_contact_value.mean()

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w #w,x,y,z
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w 

    def _build_action_mask_and_hand_tracking(self) -> None:
        """
        Build action mask AND hand design tracking in a single pass.
        This replaces both _build_action_mask_from_morph() and _init_hand_design_tracking().
        """
        F = self.cfg.F_MAX   # 5
        L = self.cfg.L_MAX   # 3  -> [base, middle, end]
        G_MAX = self.cfg.G_SCALE_MAX
        N = self.num_envs

        # Initialize tracking data structures
        self.env_to_hand_design = {}  # Maps env_id to hand design identifier
        self.hand_design_to_envs = {}  # Maps hand design to list of env_ids
        
        print(f"[combined] Building action masks and hand tracking for {self.num_envs} environments...")

        # Get USD stage to inspect what hand design is used in each environment.
        stage = get_current_stage()

        mask = torch.ones((N, 20), device=self.device, dtype=torch.float32)
        
        for env_i in range(N):
            usd_path = get_env_robot_usd_path(stage, env_i)
            
            if usd_path:
                # Extract design identifier from USD filename for hand tracking
                design_id = os.path.basename(usd_path).replace('.usd', '')
                self.env_to_hand_design[env_i] = design_id
                
                # Add to reverse mapping for hand tracking
                if design_id not in self.hand_design_to_envs:
                    self.hand_design_to_envs[design_id] = []
                self.hand_design_to_envs[design_id].append(env_i)
                
                # Parse servo presence per finger for action masking
                basename = os.path.basename(usd_path)
                for f in range(F):
                    # Default to no servos present
                    servo_present = [False, False, False]  # [base, middle, end]
                    
                    # Look for f{f+1}_<digits> pattern
                    pattern = f"f{f+1}_([0-9]*)"
                    match = re.search(pattern, basename.lower())
                    if match:
                        digits = match.group(1)
                        _, _, servo_present = _digits_to_links(digits, L, G_MAX)
                    
                    # Palm: enabled if ANY servo is present
                    if not any(servo_present):
                        mask[env_i, f] = 0  # palm
                    
                    # Base, middle, end: enabled if that specific servo is present
                    if not servo_present[0]:  # base
                        mask[env_i, 5 + f] = 0
                    if not servo_present[1]:  # middle  
                        mask[env_i, 10 + f] = 0
                    if not servo_present[2]:  # end
                        mask[env_i, 15 + f] = 0
                
                # Show details for first 10 envs
                if self.cfg.debug:
                    if env_i < 10:
                        print(f"  Env {env_i}: Design '{design_id}' from {basename}")
            else:
                print(f"  Env {env_i}: No hand USD found")
                self.env_to_hand_design[env_i] = "unknown"
                
        # after building self.hand_design_to_envs in _build_action_mask_and_hand_tracking()
        self.design_ids = sorted(self.hand_design_to_envs.keys())
        self.design_id_to_idx = {d:i for i,d in enumerate(self.design_ids)}
        self.env_to_design_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for d, envs in self.hand_design_to_envs.items():
            idx = self.design_id_to_idx[d]
            self.env_to_design_idx[torch.tensor(envs, device=self.device, dtype=torch.long)] = idx

        D = len(self.design_ids)
        # Per-design (GPU) accumulators
        self._design_total_episodes = torch.zeros(D, dtype=torch.long, device=self.device)
        self._design_total_reward   = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_success_count  = torch.zeros(D, dtype=torch.long, device=self.device)
        self._design_contact_sum    = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_grasp_sum      = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_pose_pen_sum   = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_torque_pen_sum = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_stab_pen_sum   = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_tipdist_sum    = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_total_steps    = torch.zeros(D, dtype=torch.long, device=self.device)
        self._design_hit_count      = torch.zeros(D, dtype=torch.long, device=self.device)
        self._design_omega_sum      = torch.zeros(D, dtype=torch.float32, device=self.device)
        self._design_contact_steps  = torch.zeros(D, dtype=torch.long, device=self.device)
    
        # Per-env episode accumulators (GPU)
        self._ep_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_contact = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_grasp   = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_pose    = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_torque  = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_stab    = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_tipdist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_len    = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_hits = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # add this
        self._ep_omega_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._ep_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.env_action_mask = mask

        if self.cfg.debug:
            print(f"[combined] Action mask for first environment:")
            print(f"  Palm mask:   {self.env_action_mask[0, 0:5].tolist()}")
            print(f"  Base mask:   {self.env_action_mask[0, 5:10].tolist()}")
            print(f"  Middle mask: {self.env_action_mask[0, 10:15].tolist()}")
            print(f"  End mask:    {self.env_action_mask[0, 15:20].tolist()}")

            print(f"[combined] Mapped {len(self.hand_design_to_envs)} unique hand designs:")
            for design_id, env_list in self.hand_design_to_envs.items():
                print(f"  '{design_id}': {len(env_list)} environments ({env_list[:3]}{'...' if len(env_list) > 3 else ''})")
        
        print(f"[combined] Action masking and hand tracking complete")


    def _build_env_morph_vec(self) -> None:
        """Create [num_envs, MORPH_DIM] tensor describing each env's hand morphology."""
        stage = get_current_stage()

        # build one list then tensorize (fewer GPU syncs)
        vecs = []
        for i in range(self.num_envs):
            usd_path = get_env_robot_usd_path(stage, i)
            base = os.path.basename(usd_path) if usd_path else ""
            spec = parse_morph_from_filename(base, F_MAX=self.cfg.F_MAX, L_MAX=self.cfg.L_MAX, G_MAX=self.cfg.G_SCALE_MAX)
            vecs.append(build_hand_morph_obs(spec))

        morph = torch.tensor(vecs, dtype=torch.float32, device=self.device)
        expected = self.cfg.MORPH_DIM  # set to F + F*(L + L) = F*(2L) + F
        if morph.shape[1] != expected:
            raise RuntimeError(f"MORPH_DIM mismatch: got {morph.shape[1]}, expected {expected}")
        self.env_morph_vec = morph
        
        if self.cfg.debug:
            print("\n\n")
            print(self.env_morph_vec)
            print("\n\n")

    def get_object_indices(self):
        stage = get_current_stage()

        # map asset path -> index in your OBJECT_USDS
        object_usds_real = [normalize_asset_path(p) for p in OBJECT_USDS]
        path_to_idx = {p: i for i, p in enumerate(object_usds_real)}
        allowed_paths = set(path_to_idx)
        chosen = [
            get_env_object_usd_path(
                stage,
                i,
                allowed_paths=allowed_paths,
                normalize_paths=True,
            )
            for i in range(self.num_envs)
        ]

        env_obj_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        for i, p in enumerate(chosen):
            if p is not None and p in path_to_idx:
                env_obj_idx[i] = path_to_idx[p]

        valid = env_obj_idx.ge(0)
        env_obj_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        obj_type_tensor = OBJECT_TYPE_TENSOR.to(self.device) 
        env_obj_type[valid] = obj_type_tensor[env_obj_idx[valid]]

        self.env_obj_file_idx = env_obj_idx
        self.env_obj_type_idx = env_obj_type
        self.env_obj_type_one_hot = torch.nn.functional.one_hot(
            env_obj_type,
            num_classes=self.cfg.NUM_OBJ_TYPES,
        ).float()
        
        if self.cfg.debug:
            print("\n\nObject indices for first 10 environments:")
            for i in range(min(10, self.num_envs)):
                idx = self.env_obj_file_idx[i].item()
                type_idx = self.env_obj_type_idx[i].item()
                type_name = OBJECT_TYPE_MAP.get(type_idx, "Unknown")
                print(f"Env {i}: Obj idx {idx}, Type idx {type_idx} ({type_name})")
            print("\n")
            
    def _build_finger_bodies_from_morphology(self):
        """Build finger bodies list based on actual morphology."""
        # Get unique fingers across all environments
        valid_fingers = self.cfg.get_valid_fingers_from_morphology(self.env_morph_vec)
        all_possible_fingers = set()
        for env_fingers in valid_fingers.values():
            all_possible_fingers.update(env_fingers)
        
        # Build finger bodies list for existing fingers only
        self.finger_bodies = []
        self.existing_fingertip_names = []
        
        for finger_idx in sorted(all_possible_fingers):
            finger_num = finger_idx + 1
            body_name = f'f{finger_num}_finger_end_lever'
            
            # Check if this body exists in the articulation
            if body_name in self.hand.body_names:
                self.finger_bodies.append(self.hand.body_names.index(body_name))
                self.existing_fingertip_names.append(body_name)
        
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)
        
        if self.cfg.debug:
            print(f"Active fingertips: {self.existing_fingertip_names}")
            print(f"Finger body indices: {self.finger_bodies}")
        
    def _setup_dynamic_contact_sensors_config(self):
        """Setup contact sensor configurations in scene before initialization."""
        # Set up two types of contact sensors:
        # 1. Finger-to-finger contact (keep existing penalty)
        # 2. Finger-to-object contact (add reward)
        
        for finger_idx in range(5):
            finger_num = finger_idx + 1  # Convert 0-based to 1-based
            
            # 1. Finger-to-finger contact sensors (existing - for penalty)
            other_fingers = [f"f{i+1}_finger_end_lever" for i in range(5) if i != finger_idx]
            finger_contact_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/f{finger_num}_finger_end_lever",
                filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{name}" for name in other_fingers],
                debug_vis=False,
                update_period=0.0,
                history_length=0,
            )
            self.scene.sensors[f"fingertip_{finger_num}_contact"] = ContactSensor(finger_contact_cfg)
            
            # 2. Finger-to-object contact sensors (new - for reward)
            object_contact_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/f{finger_num}_finger_end_lever",
                filter_prim_paths_expr=["/World/envs/env_.*/object"],
                debug_vis=False,
                update_period=0.0,
                history_length=0,
            )
            self.scene.sensors[f"fingertip_{finger_num}_object_contact"] = ContactSensor(object_contact_cfg)
        
        print("Added finger-to-finger and finger-to-object contact sensor configurations for all 5 fingers")
        
    def _determine_active_contact_sensors(self):
        """Determine which contact sensors are actually working after scene initialization."""
        self.active_contact_sensors = []
        
        for finger_idx in range(5):
            finger_num = finger_idx + 1
            sensor_name = f"fingertip_{finger_num}_contact"
            body_name = f'f{finger_num}_finger_end_lever'
            
            # Check if the sensor exists and the corresponding body exists
            if (sensor_name in self.scene.sensors and 
                body_name in self.hand.body_names):
                self.active_contact_sensors.append(finger_idx)
                
        print(f"Active contact sensors for fingers: {[i+1 for i in self.active_contact_sensors]}")
        
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    fingertip_pos: torch.Tensor,
    fingertip_finger_contact_value: torch.Tensor,  # finger-to-finger contact (penalty)
    fingertip_object_contact_value: torch.Tensor,  # finger-to-object contact (reward)
    individual_finger_object_contacts: torch.Tensor,  # [N, 5] individual finger contacts for grasp reward
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    initial_object_pos: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    pose_diff_penalty: torch.Tensor,
    pose_diff_penalty_scale: float,
    torque_penalty: torch.Tensor,
    torque_penalty_scale: float,
    position_stability_penalty_scale: float,
    position_stability_threshold: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
    contact_reward_scale: float,
    min_contact_reward: float,
    grasp_reward_scale: float,
    fingertip_distance_penalty_scale: float,
    approach_reward_scale: float,
):

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)
    pose_diff_penalty = pose_diff_penalty * pose_diff_penalty_scale
    
    # Contact reward: encourage touching the object (using finger-to-object contact)
    contact_reward = fingertip_object_contact_value * contact_reward_scale
    # Add minimum reward for any contact to encourage initial contact
    contact_reward = torch.where(fingertip_object_contact_value > 0.01, 
                                 contact_reward + min_contact_reward, 
                                 contact_reward)
    
    # Finger-to-finger contact penalty (penalize self-collision)
    finger_collision_penalty = fingertip_finger_contact_value * (-0.05)  # negative penalty
    
    # Grasp reward: count fingers with significant contact per environment
    fingers_in_contact = (individual_finger_object_contacts > 0.1).sum(dim=1).float()  # [N]
    grasp_reward = torch.where(fingers_in_contact >= 2.0, 
                               grasp_reward_scale * (fingers_in_contact - 1.0), 
                               torch.zeros_like(goal_dist))
    
    # Fingertip distance penalty: encourage fingertips to be close to object
    fingertip_dist_penalty = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1)
    avg_fingertip_distance = torch.mean(fingertip_dist_penalty, dim=-1)
    fingertip_dist_penalty = avg_fingertip_distance * fingertip_distance_penalty_scale
    
    # Approach reward: when no object contact, reward for getting closer to object
    approach_reward = torch.where(
        fingertip_object_contact_value < 0.01,  # No significant object contact
        approach_reward_scale / (avg_fingertip_distance + 0.1),  # Inverse distance reward
        torch.zeros_like(avg_fingertip_distance)
    )

    # Position stability penalty - only penalize if beyond threshold
    position_displacement = torch.norm(object_pos - initial_object_pos, p=2, dim=-1)
    position_stability_penalty = torch.where(
        position_displacement > position_stability_threshold,
        (position_displacement - position_stability_threshold) * position_stability_penalty_scale,
        torch.zeros_like(position_displacement)
    )

    # Total reward includes contact rewards and finger collision penalty
    reward = (dist_rew + rot_rew + action_penalty * action_penalty_scale + 
              pose_diff_penalty + torque_penalty * torque_penalty_scale + 
              contact_reward + grasp_reward + fingertip_dist_penalty + 
              position_stability_penalty + approach_reward + finger_collision_penalty)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where((torch.abs(rot_dist) <= success_tolerance) & (goal_dist <= 0.025), torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Stability reward: cube is not spinning too fast to the point where joints are messed up
    reward = torch.where((object_angvel[:, 2] > 0.25) & (object_angvel[:, 2] < 1.5), reward + 1, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (reward, goal_resets, successes, cons_successes, position_stability_penalty, 
            contact_reward, grasp_reward, fingertip_dist_penalty)
