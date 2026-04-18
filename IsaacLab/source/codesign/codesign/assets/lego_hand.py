"""
Codesign project hand configs file for IsaacLab.

Modified template from https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/allegro.py
"""

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Base folder where your generated assets live.
# This assumes this config file sits next to the "assets" folder.
# If you ever move it, you can override via env var CODESIGN_HAND_ASSETS_DIR.
ASSETS_DIR = Path(os.environ.get(
    "CODESIGN_HAND_ASSETS_DIR",
    f"{Path(__file__).parent}/group4"  # == /workspace/.../codesign/assets/
)).resolve()

ignore_dir = ["configuration", "robot_meshes"]

# collects usd files from a directory ignoring usd files from 'ignore_dir'
def collect_main_usds(root: Path) -> list[str]:
    """
    Return USDs that follow the <dir>/<dir>.usd convention, skipping any in 'configuration/' or 'robot_meshes/'.
    Examples included:
      /.../assets/cd_hand/cd_hand.usd
      /.../assets/test/rounded_hand/rounded_hand.usd
    """
    usds: list[str] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if d.name in ignore_dir:
            continue
        main_usd = d / f"{d.name}.usd"
        if main_usd.exists():
            usds.append(main_usd.as_posix())
            
    return sorted(usds)

def _resolve_main_usd(candidate: Path) -> Path | None:
    """Resolve either a direct USD path or a hand directory into the main USD."""
    if candidate.is_file() and candidate.suffix == ".usd":
        return candidate.resolve()

    if candidate.is_dir():
        main_usd = candidate / f"{candidate.name}.usd"
        if main_usd.exists():
            return main_usd.resolve()

    return None


def resolve_hand_selector(selector: str, root: Path) -> str:
    """
    Resolve a hand selector into a concrete USD path.

    Supported selector forms:
      - absolute USD path
      - absolute hand directory
      - relative USD path under root
      - relative hand directory under root
      - hand directory name / USD stem anywhere under root
    """
    raw_path = Path(os.path.expanduser(os.path.expandvars(selector)))
    direct_candidates = [raw_path] if raw_path.is_absolute() else [raw_path, root / raw_path]

    for candidate in direct_candidates:
        resolved = _resolve_main_usd(candidate)
        if resolved is not None:
            return resolved.as_posix()

    matches: list[Path] = []
    for usd_path in map(Path, collect_main_usds(root)):
        if usd_path.stem == selector or usd_path.parent.name == selector:
            matches.append(usd_path)

    if len(matches) == 1:
        return matches[0].resolve().as_posix()

    if len(matches) > 1:
        match_list = ", ".join(path.parent.name for path in matches)
        raise ValueError(f"Hand selector '{selector}' matched multiple hands under {root}: {match_list}")

    raise FileNotFoundError(f"Could not resolve hand selector '{selector}' under {root}")


def resolve_parallel_usds_from_env(root: Path) -> list[str]:
    """Resolve hand overrides from env vars, falling back to directory scan."""
    custom_usd = os.environ.get("CODESIGN_HAND_USD_PATH")
    if custom_usd:
        return [resolve_hand_selector(custom_usd, root)]

    custom_hand = os.environ.get("CODESIGN_HAND_NAME")
    if custom_hand:
        return [resolve_hand_selector(custom_hand, root)]

    parallel_usds = []
    for i in range(100):  # Support up to 100 parallel designs
        usd_path = os.environ.get(f"CODESIGN_HAND_USD_PATH_{i}")
        if usd_path:
            parallel_usds.append(resolve_hand_selector(usd_path, root))
        elif parallel_usds:
            break

    if parallel_usds:
        return parallel_usds

    hand_names = [name.strip() for name in os.environ.get("CODESIGN_HAND_NAMES", "").split(",") if name.strip()]
    if hand_names:
        return [resolve_hand_selector(name, root) for name in hand_names]

    return collect_main_usds(root)


HAND_USDS = resolve_parallel_usds_from_env(ASSETS_DIR)

# uncomment to hardcode selection
# HAND_USDS = [
    # f"{Path(__file__).parent}/few_hands_test/hand_f1_140_f2_10_f3_420_f4_130_f5/hand_f1_140_f2_10_f3_420_f4_130_f5.usd",
    # f"{Path(__file__).parent}/g1_samepalm_5fing_3servo/hand_f1_240_f2_320_f3_430_f4_240_f5_130/hand_f1_240_f2_320_f3_430_f4_240_f5_130.usd",
    # f"{Path(__file__).parent}/few_hands_test/hand_f1_10_f2_420_f3_130_f4_10_f5_320/hand_f1_10_f2_420_f3_130_f4_10_f5_320.usd",
    # f"{Path(__file__).parent}/cd_hand/rounded_hand/rounded_hand.usd",
    # f"{Path(__file__).parent}/cd_hand/wedged_hand/wedged_hand.usd",
    # f"{Path(__file__).parent}/cd_hand/cd_hand_mod/cd_hand_mod.usd",
    # f"{Path(__file__).parent}/test/rounded_hand/rounded_hand.usd",
# ]

if not HAND_USDS:
    raise RuntimeError(f"No hand USDs found under {ASSETS_DIR}")

if len(HAND_USDS) > 1:
    print(f"Loaded {len(HAND_USDS)} hand designs for parallel evaluation")
    
LEGO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=HAND_USDS,          # list of variants
        random_choice=True,          # choose one per environment
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 1.0, 0.0, 0.0),
        joint_pos={
            # keep these names consistent across variants
            "f1_finger_palm_joint": 0.0, "f1_finger_base_joint": 0.0, "f1_finger_middle_joint": 0.0, "f1_finger_end_joint": 0.0,
            "f2_finger_palm_joint": 0.0, "f2_finger_base_joint": 0.0, "f2_finger_middle_joint": 0.0, "f2_finger_end_joint": 0.0,
            "f3_finger_palm_joint": 0.0, "f3_finger_base_joint": 0.0, "f3_finger_middle_joint": 0.0, "f3_finger_end_joint": 0.0,
            "f4_finger_palm_joint": 0.0, "f4_finger_base_joint": 0.0, "f4_finger_middle_joint": 0.0, "f4_finger_end_joint": 0.0,
            "f5_finger_palm_joint": 0.0, "f5_finger_base_joint": 0.0, "f5_finger_middle_joint": 0.0, "f5_finger_end_joint": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["f.*_finger_.*_joint"],
            # effort_limit=0.5,
            # velocity_limit=100.0,
            effort_limit_sim=0.35,
            velocity_limit_sim=7.2,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
