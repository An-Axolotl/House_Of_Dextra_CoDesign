# --------------------------------------------------------
# Codesign Hand: Reorientation Environment Configuration
# Based on LEAP Hand reorientation environment
# --------------------------------------------------------

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import torch
import os

from ....assets.lego_hand import LEGO_HAND_CFG
from ....assets.objects import OBJECT_USDS

#########################################
######### Domain Randomization ##########
#########################################

_object_group = os.getenv("CODESIGN_HAND_GROUP", "").strip().lower()
_scale_high = 1.3 if _object_group == "sym4" else 1.2


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,  
        min_step_count_between_reset=720,  
        mode="reset", 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  
            "operation": "abs",  # or "scale"/"add"
            "stiffness_distribution_params":(3.0, 3.0), 
            "damping_distribution_params": (0.1, 0.1),  
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_scale_size = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (1.1, _scale_high),
        },
    )
    
def _read_hand_group_from_env():
    g = os.getenv("CODESIGN_HAND_GROUP", "").strip().lower()
    # canonicalize a bit
    if g in {"sym3", "sym4", "sym5", "anth21", "anth27", "anth33"}:
        return g
    return ""  # unknown/missing => fallback

def _is_anthro(g: str) -> bool:
    return g.startswith("anth")


##############################################
######### Environment Configuration ##########
##############################################

@configclass
class ReorientationEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4  # 
    min_episode_length_s = 20.0
    episode_length_s = 120.0
    action_space = 20  # 20 joints for the custom hand
    debug = False
    
    fixed_eval_time_s: float = 60.0
    """Fixed evaluation time in seconds for fair comparison during GHS evaluation"""
    
    ghs_evaluation_mode: bool = False
    """Whether to use fixed episode lengths and disable early termination for fair GHS evaluation"""
    
    # observation related
    enable_privileged_obs = True
    # encoded information includes the number of fingers, number of servos, number of modifiable links, and number of grammars per link
    F_MAX = 5 # maximum number of fingers
    L_MAX = 3 # max number of modifiable links per body (currently 2 due to latest solidworks assembly)
    G_SCALE_MAX = 10 # max number of grammar scaling per modifiable link, determines possible values between 0 and 1 for this observation
    MORPH_DIM = F_MAX * (1 + 2 * L_MAX) # 1 for finger presence, and each modifiable link has L_MAX G_SCALE values 
    NUM_OBJ_TYPES = 4
    hist_len = 3
    if enable_privileged_obs:
        obs_per_timestep = 43                    # 20 joints + 20 current targets
    else:
        obs_per_timestep = 40                    # 20 joints + 20 current targets
    # If you are keeping the old hand-coded vector too, include both in obs dim:
    observation_space = (
        obs_per_timestep * hist_len
        + NUM_OBJ_TYPES
        + MORPH_DIM               # existing, from filename grammar features
    )
    
    store_cur_actions = True
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.2,
            dynamic_friction=0.4,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**25
        ),
    )

    # robot - using the custom hand configuration
    robot_cfg: ArticulationCfg = LEGO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Actuated joint names for the 20-joint hand (5 fingers, 4 joints each)
    actuated_joint_names = [
        'f1_finger_palm_joint', 'f1_finger_base_joint', 'f1_finger_middle_joint', 'f1_finger_end_joint',
        'f2_finger_palm_joint', 'f2_finger_base_joint', 'f2_finger_middle_joint', 'f2_finger_end_joint',
        'f3_finger_palm_joint', 'f3_finger_base_joint', 'f3_finger_middle_joint', 'f3_finger_end_joint',
        'f4_finger_palm_joint', 'f4_finger_base_joint', 'f4_finger_middle_joint', 'f4_finger_end_joint',
        'f5_finger_palm_joint', 'f5_finger_base_joint', 'f5_finger_middle_joint', 'f5_finger_end_joint'
    ]
    
    # Fingertip body names for the 5-finger hand
    fingertip_body_names = [
        'f1_finger_end_lever', 
        'f2_finger_end_lever', 
        'f3_finger_end_lever', 
        'f4_finger_end_lever', 
        'f5_finger_end_lever'
    ]
    
    # === Default joint pose presets (controlled by env variable) ===
    _group = _read_hand_group_from_env()
    _preset = os.getenv("CODESIGN_JOINT_PRESET", "symmetric").strip().lower()
    _use_anthro = _preset == "anthro" or _is_anthro(_group)

    # Symmetric groups preset (fallback default)
    _DEFAULT_JOINT_POS_SYMM = [[
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.487, 0.487, 0.487, 0.487, 0.487,
        1.46, 1.46, 1.46, 1.46, 1.46,
        1.16, 1.16, 1.16, 1.16, 1.16
    ]]

    # Anthro groups preset
    _DEFAULT_JOINT_POS_ANTHRO = [[
        0.0, 0.0, 0.0, 0.0, 0.0,
        1.57, 0.69, 0.69, 0.69, 0.69,
        1.18, 0.75, 0.75, 0.75, 0.75,
        1.18, 1.39, 0.77, 0.77, 1.18
    ]]

    # Pick the preset — default to symmetric if no env is set
    default_joint_pos = _DEFAULT_JOINT_POS_ANTHRO if _use_anthro else _DEFAULT_JOINT_POS_SYMM


    # use this for lego_hand_v2
    # default_joint_pos = [[
    #     0.0, 0.698, -0.0977, 0.23, 0.0,
    #     1.04, 0.848, 0.0, 0.0, 0.778,
    #     1.21, 0.809, 1.53, 1.57, 1.15,
    #     1.04, 1.35, 0.0, 0.0, 1.07,
    # ]]
    
        # Decide object spawn pose based on anthro vs symmetric
    _spawn_pos_sym = (0.0,  0.0, 0.49)
    _spawn_pos_anth = (0.0, -0.05, 0.49)
    _spawn_pos = _spawn_pos_anth if _use_anthro else _spawn_pos_sym
    
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=OBJECT_USDS,
            random_choice=True,  # or False + your own mapping
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # Let PhysX compute mass/inertia from collisions using a fixed density
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            scale=(1.0, 1.0, 1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_spawn_pos, rot=(1.0, 0.0, 0.0, 0.0)), # use this for non original lego hands
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.05, 0.49), rot=(1.0, 0.0, 0.0, 0.0)), # use this for anthro groups
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, -0.0, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # in-hand object
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         # usd_path="/workspace/codesign_sim2real/source/codesign/codesign/assets/objects/set_obj1_regular_block/set_obj1_regular_block.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=400.0),
    #         scale=(1.2, 1.2, 1.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, -0.0, 0.54), rot=(1.0, 0.0, 0.0, 0.0)),
    # )
    
    # goal object - dexcube
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.2, 1.2, 1.2),
            )
        },
    )

    # Contact sensor configurations will be created dynamically based on morphology
    # Remove static contact sensor definitions
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=False)
    
    # reward scales
    z_rotation_steps = 16
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    torque_penalty_scale = -0.0
    pose_diff_penalty_scale = -0.1 # originally -0.3
    position_stability_penalty_scale = -0.1  # smaller penalty to avoid policy learning not to touch object
    position_stability_threshold = 0.03  # threshold distance in meters before penalty applies (5cm)
    reach_goal_bonus = 250
    fall_penalty = -10
    fall_dist = 0.07
    fall_height = 0.2
    success_tolerance = 0.2 
    av_factor = 0.1
    
    # Contact reward parameters
    contact_reward_scale = 2.0  # Positive reward for making contact with object
    min_contact_reward = 0.5   # Minimum reward for any contact
    fingertip_distance_penalty_scale = -1.0  # Penalty for fingertips being far from object
    grasp_reward_scale = 5.0   # Bonus for multiple fingertips in contact
    approach_reward_scale = 1.0  # Reward for approaching object when no contact
    action_type="relative" # absolute
    act_moving_average = 1./24

    # domain randomization config
    events: EventCfg = EventCfg() 
    
    # ADR configuration
    enable_adr = True  # Start with False for testing, then enable
    starting_adr_increments = 0  # 0 for no DR up to num_adr_increments for max DR 
    min_rot_adr_coeff = 0.15  # min 1 full rotation every 6.67 seconds needed to increase ADR
    min_steps_for_dr_change = 240 * 4  # number of steps
    obs_timesteps = hist_len  # same as hist_len
    wrench_trigger_every = 90  # resample every this many policy steps 
    torsional_radius = 0.0  # m
    wrench_prob_per_rollout = 0.5
    
    # ADR configuration dictionaries
    adr_cfg_dict = {
        "num_increments": 25,
        "robot_physics_material": {
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.5)
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (2.5, 3.1),
            "damping_distribution_params": (0.05, 0.15)
        },
        "object_physics_material": {
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.5)
        },
        "object_scale_mass": {
            "mass_distribution_params": (0.9, 1.3)
        }
    }
    
    adr_custom_cfg_dict = {
        "object_wrench": {
            "max_linear_accel": (0.5, 5.)
        },
        "object_spawn": {
            "x_width_spawn": (0.0, 0.01),
            "y_width_spawn": (0.0, 0.01),
            "x_rotation": (0.0, 0.1),       
            "y_rotation": (0.0, 0.1),  
            "z_rotation": (0.0, 0.0),     
        },
        "object_state_noise": {
            "object_pos_noise": (0.0, 0.00),  # m
            "object_pos_bias": (0.0, 0.0),
            "object_rot_noise": (0.0, 0.0),  # rad
            "object_rot_bias": (0.0, 0.0), 
        },
        "robot_spawn": {
            "joint_pos_noise": (0.0, 0.05),
            "joint_vel_noise": (0.0, 0.01)
        },
        "robot_state_noise": {
            "robot_noise": (0.0, 0.05),
            "robot_bias": (0.0, 0.03)
        },
        "robot_action_noise": {
            "hand_noise": (0.1, 0.2)
        },
        "action_latency": {
            "hand_latency": (0.0, 3.0),
        },
        "obs_latency": { 
            "latency": (0.0, 0.0),
        },
    }
    
    act_max_latency = int(adr_custom_cfg_dict["action_latency"]["hand_latency"][1])
    act_latency_rand = 1
    obs_max_latency = int(adr_custom_cfg_dict["obs_latency"]["latency"][1])
    obs_latency_rand = 1

    def get_valid_fingers_from_morphology(self, env_morph_vec: torch.Tensor) -> dict:
        """Extract which fingers are present for each environment from morphology vector."""
        import torch
        valid_fingers = {}
        for env_idx in range(env_morph_vec.shape[0]):
            # First F_MAX elements indicate finger presence
            finger_presence = env_morph_vec[env_idx, :self.F_MAX]
            valid_fingers[env_idx] = torch.nonzero(finger_presence, as_tuple=True)[0].tolist()
        return valid_fingers
