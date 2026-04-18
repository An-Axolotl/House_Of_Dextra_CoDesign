# codesign/assets/objects.py

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from pathlib import Path

ASSETS_ROOT = Path(__file__).parent / "objects"

OBJECT_NAMES = [
    # "ball",
    # "cross4_0", "cross4_1", "cross4_2", "cross4_3", "cross4_4",
    "set_obj1_regular_block", "set_obj2_block", "set_obj3_block", "set_obj4_block",
    "set_obj5_block", "set_obj6_block_corner", "set_obj7_block", "set_obj8_short_block",
    "set_obj9_thin_block", "set_obj10_thin_block_corner", "set_obj13_irregular_block", 
    "set_obj14_irregular_block_cross", "set_obj15_irregular_block_time",
    "set_obj11_cylinder", "set_obj12_cylinder_corner", "set_obj16_cylinder_axis",
    "set_obj11_cylinder", "set_obj12_cylinder_corner", "set_obj16_cylinder_axis",
]

# Map each object name to its type encoding
OBJECT_TYPE_MAP = {
    # "ball": 0,
    # "cross4_0": 1, "cross4_1": 1, "cross4_2": 1, "cross4_3": 1, "cross4_4": 1,
    "set_obj1_regular_block": 2, "set_obj2_block": 2, "set_obj3_block": 2, "set_obj4_block": 2,
    "set_obj5_block": 2, "set_obj6_block_corner": 2, "set_obj7_block": 2, "set_obj8_short_block": 2,
    "set_obj9_thin_block": 2, "set_obj10_thin_block_corner": 2, "set_obj13_irregular_block": 2,
    "set_obj14_irregular_block_cross": 2, "set_obj15_irregular_block_time": 2,
    "set_obj11_cylinder": 3, "set_obj12_cylinder_corner": 3, "set_obj16_cylinder_axis": 3,
    "set_obj11_cylinder": 3, "set_obj12_cylinder_corner": 3, "set_obj16_cylinder_axis": 3,
}

# Create a tensor of type encodings for each object in OBJECT_NAMES
OBJECT_TYPE_TENSOR = torch.tensor([OBJECT_TYPE_MAP[name] for name in OBJECT_NAMES], dtype=torch.long, device="cuda")

def top_level_usd_paths(root: Path, names):
    paths = []
    missing = []
    for name in names:
        # expected path: <root>/<name>/<name>.usd
        candidate = root / name / f"{name}.usd"
        if candidate.is_file():
            paths.append(str(candidate))
            continue
        # fallback: if the importer saved with a different top name, pick the first USD right under the folder
        alts = sorted((root / name).glob("*.usd"))
        if alts:
            paths.append(str(alts[0]))
        else:
            missing.append(name)
    if missing:
        print("[WARN] No USD found for:", ", ".join(missing))
    return paths

OBJECT_USDS = top_level_usd_paths(ASSETS_ROOT, OBJECT_NAMES)
# OBJECT_USDS.append(f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd")