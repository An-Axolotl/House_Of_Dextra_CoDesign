#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finger MJCF generator with:
  • Grammar stacking (1..10) at base (grammar1) and middle (grammar2)
  • Random modifiable servos among {base, middle, end} with removal order base -> middle (end always present)
  • Ghost joints for removed servos at ORIGINAL joint positions (P1 for base, P2 for middle)
  • Ghost joint limits either tiny (default) or exactly 0..0 (if allowed by your MuJoCo)

Usage examples:
  python converters/generate_rand_gram_joint_finger.py --out meshes/lego_hand/finger_random.xml
  python converters/generate_rand_gram_joint_finger.py --seed 7 --servos 2 --g1 3 --g2 1
  python converters/generate_rand_gram_joint_finger.py --ghost-mode zero     # try zero-length limits
  python converters/generate_rand_gram_joint_finger.py --ghost-mode tiny --ghost-eps 1e-9
"""
import argparse
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ----------------- Kinematic constants updated from finger1.xml -----------------
P0 = (0.01863, 0.06172, -0.028)      # finger_palm_joint pos
P1 = (0.03572, 0.06215, -0.03264)    # finger_base_joint pos  
P2 = (0.03457, 0.09362, -0.03378)    # finger_middle_joint pos
P3 = (0.03353, 0.1251, -0.03478)     # finger_end_joint pos

AX0 = (0.0, 0.0, 1.0)
AX1 = (-0.99941918, -0.03407779, 0.0)
AX2 = (-0.99941918, -0.03407779, 0.0)  # middle
AX3 = (-0.99941884, -0.03408779, 0.0)  # end (slightly different)

RNG0 = (-0.698, 0.698)   # palm
RNG1 = (0.0,  1.5708)    # base & middle
RNG2 = (0.0,  1.5708)    # end

# Grammar stack step Δg (meters) - 2.54mm distance
# delta x: 0.09mm, delta y: 2.54mm, delta z: 0.0mm
DG_DEFAULT = (-0.00009, 0.00254, -0.0)

# Compensation per removed servo (meters) - 31.51mm distance  
# delta x: 1.15mm, delta y: 31.47mm, delta z: 1.03mm
COMP_DEFAULT = (0.00115, -0.03147, 0.00103)

# Fingertip configurations - each type has collision meshes, visual mesh, and file paths
FINGERTIP_CONFIGS = {
    "standard": {
        "collision_meshes": ["finger_tip_part0", "finger_tip_part1"],
        "visual_file": "robot_meshes/visual/finger_tip_vis.obj",
        "collision_files": ["robot_meshes/finger_tip_part0.obj", "robot_meshes/finger_tip_part1.obj"]
    },
    "rounded": {
        "collision_meshes": ["finger_tip_part0", "finger_tip_part1", "finger_tip_part2", "finger_tip_part3"],
        "visual_file": "robot_meshes/visual/finger_tip_rounded_vis.obj",
        "collision_files": ["robot_meshes/finger_tip_rounded_part0.obj", "robot_meshes/finger_tip_rounded_part1.obj", 
                           "robot_meshes/finger_tip_rounded_part2.obj", "robot_meshes/finger_tip_rounded_part3.obj"]
    },
    "wedged": {
        "collision_meshes": ["finger_tip_part0"],
        "visual_file": "robot_meshes/visual/finger_tip_wedged_vis.obj",
        "collision_files": ["robot_meshes/finger_tip_wedged_part0.obj"]
    },
    "thinner": {
        "collision_meshes": ["finger_tip_part0", "finger_tip_part1"],
        "visual_file": "robot_meshes/visual/finger_tip_thinner_vis.obj",
        "collision_files": ["robot_meshes/finger_tip_thinner_part0.obj", "robot_meshes/finger_tip_thinner_part1.obj"]
    }
}

# Mesh names
VIS = {
    "palm": "finger_palm_servo_vis",
    "base": "finger_base_vis",
    "base_lever": "finger_base_lever_vis",
    "g1": "finger_grammar1_vis",
    "mid": "finger_middle_vis",
    "mid_lever": "finger_middle_lever_vis",
    "g2": "finger_grammar2_vis",
    "end": "finger_end_vis",
    "end_lever": "finger_end_lever_vis",
    "tip": "finger_tip_vis",
}
COLL = {
    "palm": "finger_palm_servo",
    "base": "finger_base",
    "base_lever_part0": "finger_base_lever_part0",
    "base_lever_part1": "finger_base_lever_part1",
    "base_lever_part2": "finger_base_lever_part2",
    "g1": "finger_grammar1",
    "mid": "finger_middle",
    "mid_lever_part0": "finger_middle_lever_part0",
    "mid_lever_part1": "finger_middle_lever_part1",
    "mid_lever_part2": "finger_middle_lever_part2",
    "g2": "finger_grammar2",
    "end": "finger_end",
    "end_lever_part0": "finger_end_lever_part0",
    "end_lever_part1": "finger_end_lever_part1",
    "end_lever_part2": "finger_end_lever_part2",
    "end_lever_part3": "finger_end_lever_part3",
    # tip meshes will be dynamically determined based on selected fingertip
}

def f3(v):
    return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"

# ----------------- XML helpers -----------------
def add_asset_section(root, fingertip_type):
    asset = ET.SubElement(root, "asset")
    # base
    ET.SubElement(asset, "mesh", name=COLL["palm"], file="robot_meshes/finger_palm_servo_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["base"], file="robot_meshes/finger_base_servo_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["base_lever_part0"], file="robot_meshes/finger_base_lever_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["base_lever_part1"], file="robot_meshes/finger_base_lever_part1.obj")
    ET.SubElement(asset, "mesh", name=COLL["base_lever_part2"], file="robot_meshes/finger_base_lever_part2.obj")
    ET.SubElement(asset, "mesh", name=COLL["g1"], file="robot_meshes/finger_grammar1_part0.obj")
    # middle
    ET.SubElement(asset, "mesh", name=COLL["mid"], file="robot_meshes/finger_middle_servo_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["mid_lever_part0"], file="robot_meshes/finger_middle_lever_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["mid_lever_part1"], file="robot_meshes/finger_middle_lever_part1.obj")
    ET.SubElement(asset, "mesh", name=COLL["mid_lever_part2"], file="robot_meshes/finger_middle_lever_part2.obj")
    ET.SubElement(asset, "mesh", name=COLL["g2"], file="robot_meshes/finger_grammar2_part0.obj")
    # end
    ET.SubElement(asset, "mesh", name=COLL["end"], file="robot_meshes/finger_end_servo_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["end_lever_part0"], file="robot_meshes/finger_end_lever_part0.obj")
    ET.SubElement(asset, "mesh", name=COLL["end_lever_part1"], file="robot_meshes/finger_end_lever_part1.obj")
    ET.SubElement(asset, "mesh", name=COLL["end_lever_part2"], file="robot_meshes/finger_end_lever_part2.obj")
    ET.SubElement(asset, "mesh", name=COLL["end_lever_part3"], file="robot_meshes/finger_end_lever_part3.obj")
    
    # Add fingertip meshes based on selected type
    tip_config = FINGERTIP_CONFIGS[fingertip_type]
    for i, (mesh_name, file_path) in enumerate(zip(tip_config["collision_meshes"], tip_config["collision_files"])):
        ET.SubElement(asset, "mesh", name=mesh_name, file=file_path)
    
    # visuals
    ET.SubElement(asset, "mesh", name=VIS["palm"], file="robot_meshes/visual/finger_palm_servo_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["base"], file="robot_meshes/visual/finger_base_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["base_lever"], file="robot_meshes/visual/finger_base_lever_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["g1"], file="robot_meshes/visual/finger_grammar1_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["mid"], file="robot_meshes/visual/finger_middle_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["mid_lever"], file="robot_meshes/visual/finger_middle_lever_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["g2"], file="robot_meshes/visual/finger_grammar2_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["end"], file="robot_meshes/visual/finger_end_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["end_lever"], file="robot_meshes/visual/finger_end_lever_vis.obj")
    ET.SubElement(asset, "mesh", name=VIS["tip"], file=tip_config["visual_file"])

def add_default_section(root):
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", solimp="0.999 0.999 0.001 0.0001 1", solref="0.0001 1",
                  friction=".2", condim="6", margin="0.002")
    ET.SubElement(default, "joint", damping="0.01", frictionloss="0.001", armature="0.0015")
    dservo = ET.SubElement(default, "default", **{"class": "servo"})
    ET.SubElement(dservo, "position", kp="2.0", kv="0.1")
    dtip = ET.SubElement(dservo, "default", **{"class": "tip"})
    ET.SubElement(dtip, "geom", friction="0.2 0.002 0.0001")
    dhand = ET.SubElement(dservo, "default", **{"class": "hand"})
    ET.SubElement(dhand, "geom", friction="0.8 0.01 0.001")

def add_contact_section(root):
    contact = ET.SubElement(root, "contact")
    pairs = [
        ("finger_base", "palm"), ("palm", "finger_base"),
        ("finger_palm_servo", "palm"), ("palm", "finger_palm_servo"),
        ("finger_base_lever", "palm"), ("palm", "finger_base_lever"),
        ("finger_middle", "palm"), ("palm", "finger_middle"),
        ("finger_middle_lever", "palm"), ("palm", "finger_middle_lever"),
        ("finger_end", "palm"), ("palm", "finger_end"),
        ("finger_end_lever", "palm"), ("palm", "finger_end_lever"),
        ("finger_root", "finger_palm_servo"),
        ("finger_root", "finger_base"),
        ("finger_root", "finger_base_lever"),
        ("finger_root", "finger_middle"),
        ("finger_root", "finger_end"),
        ("finger_palm_servo", "finger_base"),
        ("finger_palm_servo", "finger_base_lever"),
        ("finger_palm_servo", "finger_middle"),
        ("finger_palm_servo", "finger_end"),
        ("finger_base", "finger_base_lever"),
        ("finger_base", "finger_middle"),
        ("finger_base", "finger_end"),
        ("finger_base_lever", "finger_middle"),
        ("finger_base_lever", "finger_end"),
        ("finger_middle", "finger_end"),
    ]
    for a, b in pairs:
        ET.SubElement(contact, "exclude", body1=a, body2=b)

def add_geom(parent, name, mesh, group="1", rgba="1 1 1 1", cls=None):
    if cls is None:
        attrs = {"name": name, "type": "mesh", "mesh": mesh, "group": group, "rgba": rgba}
        if group == "0":  # visual geoms
            attrs.update({"contype": "0", "conaffinity": "0"})
        return ET.SubElement(parent, "geom", **attrs)
    else:
        attrs = {"class": cls, "name": name, "type": "mesh", "mesh": mesh, "group": group, "rgba": rgba}
        if group == "0":  # visual geoms
            attrs.update({"contype": "0", "conaffinity": "0"})
        return ET.SubElement(parent, "geom", **attrs)

def add_joint(parent, name, jtype, pos, axis=None, rng=None, limited=None):
    attrs = {"name": name, "type": jtype, "pos": f3(pos)}
    if axis is not None:
        attrs["axis"] = f3(axis)
    if rng is not None:
        attrs["range"] = f"{rng[0]} {rng[1]}"
    if limited is not None:
        attrs["limited"] = "true" if limited else "false"
    return ET.SubElement(parent, "joint", **attrs)

def add_ghost_joint(parent, name, pos, axis, mode="tiny", eps=1e-8):
    """
    Ghost hinge at ORIGINAL joint position.
      mode="tiny": range="0 eps"
      mode="zero": range="0 0"  (may be rejected by some MuJoCo builds; if so, use tiny)
    """
    rng = (0.0, 0.0) if mode == "zero" else (0.0, float(eps))
    return add_joint(parent, name, "hinge", pos, axis=axis, rng=rng, limited=True)

def add_grammar_stack(parent_body, root_name, vis_name, coll_name, count, delta):
    """
    Always create a root grammar body named finger_{root_name}. If count==0, it's a ghost container.
    If count>=1, put stack0 geoms under root; stacks 1..N-1 are nested bodies at +delta each.
    """
    grammar_root = ET.SubElement(parent_body, "body", name=f"finger_{root_name}", pos="0 0 0")
    if count <= 0:
        return grammar_root, False

    add_geom(grammar_root, f"finger_{root_name}_0_geom", coll_name)
    add_geom(grammar_root, f"finger_{root_name}_0_vis_geom", vis_name, group="0")

    deepest = grammar_root
    for i in range(1, count):
        b = ET.SubElement(deepest, "body", name=f"finger_{root_name}_{i}", pos=f3(delta))
        add_geom(b, f"finger_{root_name}_{i}_geom", coll_name)
        add_geom(b, f"finger_{root_name}_{i}_vis_geom", vis_name, group="0")
        deepest = b
    return deepest, True

# ----------------- Worldbody with servo removal -----------------
def build_worldbody(root, g1_count, g2_count, delta, comp, mod_servos_present, ghost_mode, ghost_eps, fingertip_type, is_ghost_finger=False):
    # For ghost fingers, make all joints ghost joints with minimal structure
    if is_ghost_finger:
        base_present = False
        middle_present = False
        end_present = True  # Keep minimal structure
        g1_active = 0
        g2_active = 0
    else:
        # Presence rule: end always present; removal order base -> middle
        base_present   = (mod_servos_present >= 3)
        middle_present = (mod_servos_present >= 2)
        end_present    = True

        # Force grammars off if their servo site is removed
        g1_active = g1_count if base_present else 0
        g2_active = g2_count if middle_present else 0

    world = ET.SubElement(root, "worldbody")
    broot = ET.SubElement(world, "body", name="finger_root", pos="0 0 0")

    # For ghost fingers, create minimal massless structure with no geometry
    if is_ghost_finger:
        # Minimal palm structure for ghost fingers - NO GEOMETRY
        palm = ET.SubElement(broot, "body", name="finger_palm_servo", pos="0 0 0")
        add_inertial_if_massless(palm)
        
        # Minimal base structure with ghost joints only - NO GEOMETRY
        base = ET.SubElement(palm, "body", name="finger_base", pos="0 0 0")
        add_ghost_joint(base, "finger_palm_joint", P0, AX0, mode=ghost_mode, eps=ghost_eps)
        add_inertial_if_massless(base)

        base_lev = ET.SubElement(base, "body", name="finger_base_lever", pos="0 0 0")
        add_ghost_joint(base_lev, "finger_base_joint", P1, AX1, mode=ghost_mode, eps=ghost_eps)
        add_inertial_if_massless(base_lev)

        # Minimal grammar containers - NO GEOMETRY
        g1_root = ET.SubElement(base_lev, "body", name="finger_grammar1", pos="0 0 0")
        add_inertial_if_massless(g1_root)

        mid = ET.SubElement(g1_root, "body", name="finger_middle", pos="0 0 0")
        add_inertial_if_massless(mid)

        mid_lev = ET.SubElement(mid, "body", name="finger_middle_lever", pos="0 0 0")
        add_ghost_joint(mid_lev, "finger_middle_joint", P2, AX2, mode=ghost_mode, eps=ghost_eps)
        add_inertial_if_massless(mid_lev)

        g2_root = ET.SubElement(mid_lev, "body", name="finger_grammar2", pos="0 0 0")
        add_inertial_if_massless(g2_root)

        end = ET.SubElement(g2_root, "body", name="finger_end", pos="0 0 0")
        add_inertial_if_massless(end)

        end_lev = ET.SubElement(end, "body", name="finger_end_lever", pos="0 0 0")
        add_ghost_joint(end_lev, "finger_end_joint", P3, AX3, mode=ghost_mode, eps=ghost_eps)
        add_inertial_if_massless(end_lev)

        # Minimal tip - NO GEOMETRY
        tip = ET.SubElement(end_lev, "body", name="finger_tip", pos="0 0 0")
        add_inertial_if_massless(tip)
        
        return

    # Normal finger generation for non-ghost fingers
    # palm (always) - ONLY FOR NON-GHOST FINGERS
    palm = ET.SubElement(broot, "body", name="finger_palm_servo", pos="0 0 0")
    add_geom(palm, "finger_palm_servo_geom", COLL["palm"])
    add_geom(palm, "finger_palm_servo_vis_geom", VIS["palm"], group="0")

    # base
    base = ET.SubElement(palm, "body", name="finger_base", pos="0 0 0")
    add_joint(base, "finger_palm_joint", "hinge", P0, AX0, RNG0, limited=True)
    if base_present:
        add_geom(base, "finger_base_geom", COLL["base"])
        add_geom(base, "finger_base_vis_geom", VIS["base"], group="0")
    add_inertial_if_massless(base)

    base_lev = ET.SubElement(base, "body", name="finger_base_lever", pos="0 0 0")
    if base_present:
        add_joint(base_lev, "finger_base_joint", "hinge", P1, AX1, RNG1, limited=True)
        add_geom(base_lev, "finger_base_lever_part0_geom", COLL["base_lever_part0"])
        add_geom(base_lev, "finger_base_lever_part1_geom", COLL["base_lever_part1"])
        add_geom(base_lev, "finger_base_lever_part2_geom", COLL["base_lever_part2"])
        add_geom(base_lev, "finger_base_lever_vis_geom", VIS["base_lever"], group="0")
    else:
        # Ghost base joint at ORIGINAL pos
        add_ghost_joint(base_lev, "finger_base_joint", P1, AX1, mode=ghost_mode, eps=ghost_eps)
    add_inertial_if_massless(base_lev)

    # grammar1 (ghost container if base removed)
    g1_deep, _ = add_grammar_stack(base_lev, "grammar1", VIS["g1"], COLL["g1"], g1_active, delta)

    # middle
    middle_pos = (0.0, 0.0, 0.0)
    if not base_present:
        middle_pos = (middle_pos[0] + comp[0], middle_pos[1] + comp[1], middle_pos[2] + comp[2])
    mid = ET.SubElement(g1_deep, "body", name="finger_middle", pos=f3(middle_pos))
    if middle_present:
        add_geom(mid, "finger_middle_geom", COLL["mid"])
        add_geom(mid, "finger_middle_vis_geom", VIS["mid"], group="0")
    add_inertial_if_massless(mid)

    mid_lev = ET.SubElement(mid, "body", name="finger_middle_lever", pos="0 0 0")
    if middle_present:
        add_joint(mid_lev, "finger_middle_joint", "hinge", P2, AX2, RNG1, limited=True)
        add_geom(mid_lev, "finger_middle_lever_part0_geom", COLL["mid_lever_part0"])
        add_geom(mid_lev, "finger_middle_lever_part1_geom", COLL["mid_lever_part1"])
        add_geom(mid_lev, "finger_middle_lever_part2_geom", COLL["mid_lever_part2"])
        add_geom(mid_lev, "finger_middle_lever_vis_geom", VIS["mid_lever"], group="0")
    else:
        # Ghost middle joint at ORIGINAL pos
        add_ghost_joint(mid_lev, "finger_middle_joint", P2, AX2, mode=ghost_mode, eps=ghost_eps)
    add_inertial_if_massless(mid_lev)

    # grammar2 (ghost container if middle removed)
    g2_deep, _ = add_grammar_stack(mid_lev, "grammar2", VIS["g2"], COLL["g2"], g2_active, delta)

    # end (always)
    end_pos = (0.0, 0.0, 0.0)
    if not middle_present:
        end_pos = (end_pos[0] + comp[0], end_pos[1] + comp[1], end_pos[2] + comp[2])
    end = ET.SubElement(g2_deep, "body", name="finger_end", pos=f3(end_pos))
    if end_present:
        add_geom(end, "finger_end_geom", COLL["end"])
        add_geom(end, "finger_end_vis_geom", VIS["end"], group="0")

    end_lev = ET.SubElement(end, "body", name="finger_end_lever", pos="0 0 0")
    add_joint(end_lev, "finger_end_joint", "hinge", P3, AX3, RNG2, limited=True)
    add_geom(end_lev, "finger_end_lever_part0_geom", COLL["end_lever_part0"])
    add_geom(end_lev, "finger_end_lever_part1_geom", COLL["end_lever_part1"])
    add_geom(end_lev, "finger_end_lever_part2_geom", COLL["end_lever_part2"])
    add_geom(end_lev, "finger_end_lever_part3_geom", COLL["end_lever_part3"])
    add_geom(end_lev, "finger_end_lever_vis_geom", VIS["end_lever"], group="0")

    tip = ET.SubElement(end_lev, "body", name="finger_tip", pos="0 0 0")
    
    # Add all collision geoms for the selected fingertip type
    tip_config = FINGERTIP_CONFIGS[fingertip_type]
    for i, mesh_name in enumerate(tip_config["collision_meshes"]):
        add_geom(tip, f"finger_tip_part{i}_geom", mesh_name, cls="tip")
    
    add_geom(tip, "finger_tip_vis_geom", VIS["tip"], group="0")

def add_actuators(root, base_present, middle_present, is_ghost_finger=False):
    actuator = ET.SubElement(root, "actuator")
    
    if is_ghost_finger:
        # For ghost fingers, create actuators with very limited range for all joints
        ET.SubElement(actuator, "position", name="finger_palm_joint", **{"class": "servo"},
                      joint="finger_palm_joint", forcerange="-0.001 0.001", inheritrange="1")
        ET.SubElement(actuator, "position", name="finger_base_joint", **{"class": "servo"},
                      joint="finger_base_joint", forcerange="-0.001 0.001", inheritrange="1")
        ET.SubElement(actuator, "position", name="finger_middle_joint", **{"class": "servo"},
                      joint="finger_middle_joint", forcerange="-0.001 0.001", inheritrange="1")
        ET.SubElement(actuator, "position", name="finger_end_joint", **{"class": "servo"},
                      joint="finger_end_joint", forcerange="-0.001 0.001", inheritrange="1")
    else:
        # Normal actuators
        ET.SubElement(actuator, "position", name="finger_palm_joint", **{"class": "servo"},
                      joint="finger_palm_joint", forcerange="-0.8 0.8", inheritrange="1")
        if base_present:
            ET.SubElement(actuator, "position", name="finger_base_joint", **{"class": "servo"},
                          joint="finger_base_joint", forcerange="-0.8 0.8", inheritrange="1")
        else:
            # Ghost actuator for non-present base joint  
            ET.SubElement(actuator, "position", name="finger_base_joint", **{"class": "servo"},
                          joint="finger_base_joint", forcerange="-0.001 0.001", inheritrange="1")
        if middle_present:
            ET.SubElement(actuator, "position", name="finger_middle_joint", **{"class": "servo"},
                          joint="finger_middle_joint", forcerange="-0.80 0.80", inheritrange="1")
        else:
            # Ghost actuator for non-present middle joint
            ET.SubElement(actuator, "position", name="finger_middle_joint", **{"class": "servo"},
                          joint="finger_middle_joint", forcerange="-0.001 0.001", inheritrange="1")
        ET.SubElement(actuator, "position", name="finger_end_joint", **{"class": "servo"},
                      joint="finger_end_joint", forcerange="-0.3 0.3", inheritrange="1")
    
def add_inertial_if_massless(body, mass=1e-6, diag=1e-6):
    """
    If `body` has a joint but no geoms AND no inertial yet, inject a tiny inertial.
    MuJoCo requires pos=... on <inertial>, so we set pos="0 0 0".
    """
    has_joint = any(child.tag == "joint" for child in body)
    has_geom  = any(child.tag == "geom"  for child in body)
    has_inert = any(child.tag == "inertial" for child in body)
    if has_joint and not has_geom and not has_inert:
        ET.SubElement(
            body,
            "inertial",
            mass=f"{mass}",
            diaginertia=f"{diag} {diag} {diag}",
            pos="0 0 0",
        )


def prettify(elem) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

def build_mjcf(g1_count, g2_count, delta, comp, mod_servos_present, ghost_mode, ghost_eps, fingertip_type, is_ghost_finger=False):
    root = ET.Element("mujoco", model="finger")
    ET.SubElement(root, "compiler", angle="radian")
    add_asset_section(root, fingertip_type)
    add_default_section(root)
    add_contact_section(root)
    build_worldbody(root, g1_count, g2_count, delta, comp, mod_servos_present, ghost_mode, ghost_eps, fingertip_type, is_ghost_finger)
    base_present   = (mod_servos_present >= 3) and not is_ghost_finger
    middle_present = (mod_servos_present >= 2) and not is_ghost_finger
    add_actuators(root, base_present, middle_present, is_ghost_finger)
    return root

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Finger MJCF generator with servo removal + grammar stacking + ghost joints.")
    ap.add_argument("--out", type=str, default="finger_random.xml", help="Output MJCF path")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--ghost-finger", action="store_true", help="Generate a ghost finger with minimal dead joints")
    # Grammar counts - updated range to 1-10
    ap.add_argument("--g1", type=int, default=None, help="Grammar count at base site (1..10)")
    ap.add_argument("--g2", type=int, default=None, help="Grammar count at middle site (1..10)")
    ap.add_argument("--min-g", type=int, default=1, help="Minimum random grammar count per site")
    ap.add_argument("--max-g", type=int, default=10, help="Maximum random grammar count per site")
    # Grammar stack step Δg
    ap.add_argument("--delta", type=float, nargs=3, default=DG_DEFAULT, metavar=("DX", "DY", "DZ"))
    # Compensation vector
    ap.add_argument("--comp", type=float, nargs=3, default=COMP_DEFAULT, metavar=("CX", "CY", "CZ"))
    # Modifiable servos present (among base, middle, end). End always present; removal order base->middle.
    ap.add_argument("--servos", type=int, default=None, choices=[1, 2, 3],
                    help="Number of PRESENT modifiable servos among {base, middle, end}. If omitted, random in [1,3].")
    ap.add_argument("--min-servos", type=int, default=1)
    ap.add_argument("--max-servos", type=int, default=3)
    # Fingertip selection
    ap.add_argument("--tip", type=str, default=None, choices=list(FINGERTIP_CONFIGS.keys()),
                    help="Fingertip type. If omitted, random selection.")
    # Ghost joint limits
    ap.add_argument("--ghost-mode", choices=["tiny", "zero"], default="tiny",
                    help='Ghost joint limit mode: "tiny" -> [0, eps], "zero" -> [0, 0] (may be invalid on some MuJoCo builds)')
    ap.add_argument("--ghost-eps", type=float, default=1e-8, help="Upper limit when --ghost-mode=tiny")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.ghost_finger:
        # Generate ghost finger with minimal structure
        g1, g2 = 0, 0
        mod_servos_present = 1
        fingertip_type = "standard"  # Use standard for ghost
        
        root = build_mjcf(
            g1, g2, tuple(args.delta), tuple(args.comp),
            mod_servos_present, args.ghost_mode, args.ghost_eps, fingertip_type, is_ghost_finger=True
        )
        xml_str = prettify(root)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(xml_str)
        
        print(f"Wrote ghost finger {args.out}")
        return

    if not (1 <= args.min_g <= args.max_g <= 10):
        raise SystemExit("min/max grammar must satisfy 1 <= min_g <= max_g <= 10")
    if not (1 <= args.min_servos <= args.max_servos <= 3):
        raise SystemExit("min/max servos must satisfy 1 <= min_servos <= 3 and <= 3")

    g1 = args.g1 if args.g1 is not None else random.randint(args.min_g, args.max_g)
    g2 = args.g2 if args.g2 is not None else random.randint(args.min_g, args.max_g)
    mod_servos_present = args.servos if args.servos is not None else random.randint(args.min_servos, args.max_servos)
    fingertip_type = args.tip if args.tip is not None else random.choice(list(FINGERTIP_CONFIGS.keys()))

    root = build_mjcf(
        g1, g2, tuple(args.delta), tuple(args.comp),
        mod_servos_present, args.ghost_mode, args.ghost_eps, fingertip_type
    )
    xml_str = prettify(root)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(xml_str)

    base_present   = (mod_servos_present >= 3)
    middle_present = (mod_servos_present >= 2)
    tip_parts_count = len(FINGERTIP_CONFIGS[fingertip_type]["collision_meshes"])
    print(
        f"Wrote {args.out} | servos_present={mod_servos_present} "
        f"(base={'on' if base_present else 'off'}, middle={'on' if middle_present else 'off'}, end=on) "
        f"| grammar1={g1 if base_present else 0} grammar2={g2 if middle_present else 0} "
        f"| fingertip={fingertip_type} ({tip_parts_count} parts) "
        f"| Δg={tuple(args.delta)} comp={tuple(args.comp)} ghost=({args.ghost_mode}, eps={args.ghost_eps})"
    )

if __name__ == "__main__":
    main()
