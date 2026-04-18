#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a convex-hull palm *mesh* (OBJ) from randomized finger placements.

What it does
------------
- Samples finger base positions (x, y) around a palm-sized circle and yaw angles.
- Builds a set of grip points from those placements.
- Computes the palm center height from a horn stack-up model and an offset.
- Creates a *3D convex hull* of two layers (top/bottom at fixed thickness) to get a closed, watertight palm mesh.
- Writes the mesh to OBJ (triangles), plus a small JSON metadata file.

Notes
-----
- This script generates **only the palm mesh**; no fingers/MJCF are created.
- The mesh is convex by construction (convex hull). For visual-only nonconvex palms,
  you would need a different meshing method.

Usage examples
--------------
  python generate_palm_mesh.py --out palm_hull.obj --fingers 5 --palm-radius 0.12 --seed 7
  python generate_palm_mesh.py --fingers 4 --thickness 0.02 --grip-sep-in 1.25
  python generate_palm_mesh.py --fingers 3 --z-policy top --z-top 0.075

Outputs
-------
- OBJ mesh at --out (default: palm_hull.obj)
- Metadata JSON next to the OBJ (same basename + .json)
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull as CH
from collections import Counter

import trimesh
import coacd


# ----------------------- Helpers & data structures -----------------------

@dataclass
class PalmParams:
    fingers: int = 5
    palm_radius: float = 0.12           # meters
    min_radius_factor: float = 0.7      # inner ring factor for placement
    grip_sep_in: float = 1.5            # inches, left/right horn separation
    palm_offset_in: float = 0.25        # inches, horns-to-palm-center offset
    min_center_in: float = 3.0          # inches, ensure center coverage
    thickness: float = 0.0381           # meters, ~1 inch slab
    horn_stack: float = 0.036 + 0.002   # meters, servo + horn stack-up
    motor_z_seed: float = 0.030         # meters, height seed for horns base
    seed: int = None

    # Z plane policy: "auto" (from horns), or "top" (explicit z_top), or "center" (explicit palm_center_z)
    z_policy: str = "auto"
    z_top: float = None                 # used when z_policy == "top"
    palm_center_z: float = None         # used when z_policy == "center"
    
    mount_offset: float = 0.004         # meters above z_top to place the submodel root
    qfix_name: str = "identity"         # one-time axis correction preset
    flip_normals: bool = False          # flip triangle winding if mesh looks inside-out
    
    # SOLIDWORKS reference (the export you trust)
    ref_base: Tuple[float, float, float] = (0.019395, 0.03924, 0.0)
    ref_tip:  Tuple[float, float, float] = (0.01838, 0.06902, 0.0)
    
    # ---- placement controls ----
    placement_mode: str = "asymmetric"   # asymmetric | symmetric | anthro-top-heavy
    min_angle_deg: float = 12.0          # hard minimum pairwise separation
    symmetric_start_deg: float = 0.0     # where to put the first finger for symmetric mode
    symmetric_jitter_deg: float = 0.0    # per-finger jitter (+/-) around symmetric angles
    y_mirrored: bool = False     # if your world has a 180° X flip, set True
    
    # discrete slot placement
    use_discrete_slots: bool = True
    slot_count: int = 36              # S
    min_sep_slots: int = 4            # δ (e.g., 4 slots ≈ 40° at S=36)
    mount_slots: List[int] = None     # optional override: exact slot ids (length=fingers)
    jitter_deg: float = 0.0           # keep 0.0 for mesh; add jitter at runtime in PPO
    
    # anthropomorphic bands (closed-open on circle)
    thumb_bottom_deg: Tuple[float, float] = (210.0, 330.0)  # bottom arc for thumb
    top_band_deg: Tuple[float, float] = (300.0, 60.0)       # wrapped arc for non-thumbs
    # freeze top-finger positions (discrete mode) and/or the thumb slot
    anthro_top_fixed_slots: List[int] = None  # length must be fingers-1
    thumb_fixed_slot: Optional[int] = None    # single slot index for the thumb

@dataclass
class PalmProducts:
    positions: List[Tuple[float, float]]
    angles_deg: List[float]
    palm_center_z: float
    z_top: float
    z_bottom: float
    vertices: np.ndarray   # (N,3)
    faces: np.ndarray      # (M,3) indices into vertices
    finger_positions: List[Tuple[float, float, float]]  # 3D positions
    # finger_quaternions: List[Tuple[float, float, float, float]]  # (x,y,z,w) quaternions
    finger_quaternions: List[Tuple[float, float, float, float]]  # (w,x,y,z)

    
# --- Quaternion helpers (MuJoCo expects w,x,y,z) ---
def rotz_quat_wxyz(yaw_rad: float):
    import math
    c = math.cos(yaw_rad * 0.5)
    s = math.sin(yaw_rad * 0.5)
    return (c, 0.0, 0.0, s)  # rotation about +Z, (w,x,y,z)

def qmul(q2, q1):  # compose q = q2 ⊗ q1 , both wxyz
    w2,x2,y2,z2 = q2; w1,x1,y1,z1 = q1
    return (
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
    )

QFIX_PRESETS = {
    "identity": (1.0, 0.0, 0.0, 0.0),     # already +Z up, +X forward
    "x180":     (0.0, 1.0, 0.0, 0.0),     # flip Z (down->up)
    "x90":      (0.70710678, 0.70710678, 0.0, 0.0),
    "y90":      (0.70710678, 0.0, 0.70710678, 0.0),
}

def make_mjcf_frames(products, model_base="finger_model", prefix="f", attach_body="finger_root"):
    lines = []
    for i, ((x,y,z), q) in enumerate(zip(products.finger_positions, products.finger_quaternions), start=1):
        w,xq,yq,zq = q
        lines.append(
            f'  <frame pos="{x:.6f} {y:.6f} {z:.6f}" quat="{w:.6f} {xq:.6f} {yq:.6f} {zq:.6f}">'
            f'\n    <attach model="{model_base}_{i}" body="{attach_body}" prefix="{prefix}{i}_"/>'
            f'\n  </frame>'
        )
    return lines


# ----------------------- Core geometry routines -----------------------

def generate_finger_positions(num_fingers: int, palm_radius: float, min_radius_factor: float, params: PalmParams):
    Rmin = palm_radius * min_radius_factor
    Rmax = palm_radius * 0.9

    # 1) choose angles (possibly via slots)
    if params.use_discrete_slots:
        S = params.slot_count
        δ = params.min_sep_slots

        if params.mount_slots is not None:
            assert len(params.mount_slots) == num_fingers, "--mount-slots must match --fingers"
            slots = [s % S for s in params.mount_slots]
        else:
            if params.placement_mode == "symmetric":
                base = [int(round((i * S) / max(1, num_fingers))) % S for i in range(num_fingers)]
                slots = sample_slots_with_min_sep(num_fingers, S, δ, allowed=None)
                # prefer near symmetric anchors by small local bias
                # (optional: keep simple for now)
                slots = base if len(set(base)) == num_fingers else slots

            elif params.placement_mode == "anthro-top-heavy":
                thumb_mask = arc_to_slot_mask(*params.thumb_bottom_deg, S)
                top_mask   = arc_to_slot_mask(*params.top_band_deg,   S)

                # 1) Pick thumb (fixed or sampled within bottom arc)
                if params.thumb_fixed_slot is not None:
                    thumb = params.thumb_fixed_slot % S
                    assert thumb_mask[thumb], "--thumb-fixed-slot must lie within thumb-bottom-deg arc"
                else:
                    thumb = sample_slots_with_min_sep(1, S, δ, allowed=thumb_mask)[0]

                # 2) Top fingers (fixed list or sampled within top band), keep min-sep from thumb and each other
                allowed = top_mask.copy()
                for d in range(-max(δ,2)+1, max(δ,2)):   # forbid a neighborhood around thumb
                    allowed[(thumb + d) % S] = False

                if params.anthro_top_fixed_slots:
                    assert len(params.anthro_top_fixed_slots) == num_fingers-1, \
                        "--anthro-top-fixed-slots must have (fingers-1) entries in anthro mode"
                    rest = []
                    for s in params.anthro_top_fixed_slots:
                        s = s % S
                        assert allowed[s], "top fixed slot not allowed (top-band) or too close to thumb"
                        # simple pairwise min-sep check among top fixed choices
                        if any((abs((s-r) % S) <= (δ-1) or abs((r-s) % S) <= (δ-1)) for r in rest):
                            raise AssertionError("top fixed slots violate min-sep among themselves")
                        rest.append(s)
                else:
                    rest = sample_slots_with_min_sep(num_fingers-1, S, δ, allowed=allowed)

                slots = [thumb] + rest

            else:  # "asymmetric"
                slots = sample_slots_with_min_sep(num_fingers, S, δ, allowed=None)

        # map slots to angles and optional *mesh-time* jitter (keep 0 for reproducibility)
        angles_deg = []
        for s in slots:
            a = slot_to_angle_deg(s, S)
            a = (a + random.uniform(-params.jitter_deg, params.jitter_deg)) % 360.0
            angles_deg.append(a)

    else:
        # fallback: existing continuous samplers (unchanged)
        if params.placement_mode == "symmetric":
            base = _symmetric_angles(num_fingers, params.symmetric_start_deg)
            angles_deg = [(a + random.uniform(-params.symmetric_jitter_deg, params.symmetric_jitter_deg)) % 360.0
                          for a in base]
        elif params.placement_mode == "anthro-top-heavy":
            # existing anthro sampling (unchanged)
            ...
        else:
            angles_deg = _sample_with_min_sep(num_fingers, 0.0, 360.0, params.min_angle_deg, max_tries=4000)

    if params.y_mirrored and params.placement_mode != "anthro-top-heavy":
        angles_deg = [(-a) % 360.0 for a in angles_deg]

    # 2) radii & XY (unchanged)
    positions = []
    for a in angles_deg:
        ang = math.radians(a)
        r = random.uniform(Rmin, Rmax)
        positions.append((math.cos(ang) * r, math.sin(ang) * r))

    return positions, angles_deg


def grip_points_from_motors(motor_positions: List[Tuple[float, float, float]],
                            motor_angles_deg: List[float],
                            grip_sep_m: float,
                            palm_offset_m: float,
                            horn_stack: float):
    """Compute grip points (two per finger) plus a central grid, and palm_center_z.
    The returned grip point z is not used for final palm thickness; we recompute top/bottom around palm_center_z.
    """
    grip_points = []
    horn_positions = []

    for (mx, my, mz), angle_deg in zip(motor_positions, motor_angles_deg):
        horn_z = mz + horn_stack
        horn_positions.append((mx, my, horn_z))

        ang = math.radians(angle_deg)
        radial = (math.cos(ang), math.sin(ang))
        perp = (-radial[1], radial[0])
        half_sep = grip_sep_m / 2.0
        grip_z = horn_z - palm_offset_m

        grip_points.append((mx + perp[0] * half_sep, my + perp[1] * half_sep, grip_z))
        grip_points.append((mx - perp[0] * half_sep, my - perp[1] * half_sep, grip_z))

    if horn_positions:
        avg_horn_z = sum(p[2] for p in horn_positions) / len(horn_positions)
        palm_center_z = avg_horn_z - palm_offset_m
    else:
        palm_center_z = 0.0

    return grip_points, palm_center_z


def add_center_grid(grip_points: List[Tuple[float, float, float]], min_center_m: float, center_z: float):
    """Ensure central coverage by adding a 5x5 grid in the middle."""
    half = min_center_m / 2.0
    for i in range(5):
        for j in range(5):
            x = -half + (i / 4.0) * min_center_m
            y = -half + (j / 4.0) * min_center_m
            grip_points.append((x, y, center_z))


def build_palm_hull(grip_points, palm_center_z, thickness, flip=False):
    """Closed prism from the 2D convex hull of (x,y) points.
       Returns (V, F, z_top, z_bot) with outward-facing normals.
    """
    half = thickness * 0.5
    pts_xy = np.array([(gx, gy) for gx,gy,_ in grip_points], dtype=np.float64)

    hull2d = CH(pts_xy)
    order = hull2d.vertices            # CCW when seen from +Z
    n = len(order)

    # vertices: top ring then bottom ring (same CCW order)
    top_z = palm_center_z + half
    bot_z = palm_center_z - half
    top = [(pts_xy[i,0], pts_xy[i,1], top_z) for i in order]
    bot = [(pts_xy[i,0], pts_xy[i,1], bot_z) for i in order]
    V = np.array(top + bot, dtype=np.float64)

    F = []

    # --- top cap (CCW => normal +Z)
    for i in range(1, n-1):
        F.append([0, i, i+1])

    # --- bottom cap (CW => normal -Z)
    base = n
    for i in range(1, n-1):
        F.append([base, base+i+1, base+i])

    # --- sides (outward)
    for i in range(n):
        a  = i
        b  = (i+1) % n
        at = a            # top ring
        bt = b
        ab = base + a     # bottom ring
        bb = base + b
        # two outward-facing tris per quad
        F.append([at, bt, bb])
        F.append([at, bb, ab])

    F = np.array(F, dtype=np.int32)

    if flip:
        F = F[:, ::-1]    # flip all triangle windings

    return V, F, top_z, bot_z


# ----------------------- IO utilities -----------------------

def write_obj(path: str, V: np.ndarray, F: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # compute per-face normals
    def tri_normal(a,b,c):
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            return n / norm
        else:
            return np.array([0.0, 0.0, 1.0])  # fallback

    N = np.zeros((F.shape[0], 3), dtype=np.float64)
    for i,(i0,i1,i2) in enumerate(F):
        N[i] = tri_normal(V[i0], V[i1], V[i2])

    with open(path, "w", encoding="utf-8") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in N:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        # reference the same face-normal index for all three vertices of that face
        for i,(i0,i1,i2) in enumerate(F, start=1):
            f.write(f"f {i0+1}//{i} {i1+1}//{i} {i2+1}//{i}\n")


def analyze_mesh_normals(V, F):
    """Analyze mesh normal directions to detect inside-out geometry."""
    def tri_normal(a,b,c):
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    
    # Compute center of mesh
    center = np.mean(V, axis=0)
    
    # Check a few face normals
    outward_count = 0
    inward_count = 0
    
    for i, (i0, i1, i2) in enumerate(F[:min(10, len(F))]):  # Check first 10 faces
        face_center = (V[i0] + V[i1] + V[i2]) / 3.0
        normal = tri_normal(V[i0], V[i1], V[i2])
        to_center = center - face_center
        
        # If normal points away from center, it's outward
        if np.dot(normal, to_center) < 0:
            outward_count += 1
        else:
            inward_count += 1
    
    return outward_count, inward_count


def boundary_edges_count(F):
    edges = []
    for a,b,c in F:
        edges.extend([tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((c,a)))])
    cnt = Counter(edges)
    return sum(1 for _e,n in cnt.items() if n == 1)


def write_metadata(json_path: str, products: PalmProducts, params: PalmParams):
    meta = {
        "params": {
            "fingers": params.fingers,
            "palm_radius": params.palm_radius,
            "min_radius_factor": params.min_radius_factor,
            "grip_sep_in": params.grip_sep_in,
            "palm_offset_in": params.palm_offset_in,
            "min_center_in": params.min_center_in,
            "thickness": params.thickness,
            "horn_stack": params.horn_stack,
            "motor_z_seed": params.motor_z_seed,
            "z_policy": params.z_policy,
            "z_top": params.z_top,
            "palm_center_z": params.palm_center_z,
            "seed": params.seed,
        },
        "positions": products.positions,
        "angles_deg": products.angles_deg,
        "palm_center_z": products.palm_center_z,
        "z_top": products.z_top,
        "z_bottom": products.z_bottom,
        "vertex_count": int(products.vertices.shape[0]),
        "face_count": int(products.faces.shape[0]),
        "finger_positions": products.finger_positions,
        "finger_quaternions": products.finger_quaternions,
    }
    
    if params.use_discrete_slots:
        S = params.slot_count
        # recover slots from saved angles (nearest slot)
        slots = [int(round((a % 360.0) / (360.0 / S))) % S for a in products.angles_deg]
        meta["mount_slots"] = {"slot_count": S, "slots": slots, "min_sep_slots": params.min_sep_slots}
        
    meta["frames_mjcf"] = make_mjcf_frames(products)
    
    meta["placement"] = {
        "mode": params.placement_mode,
        "min_angle_deg": params.min_angle_deg,
        "symmetric_start_deg": params.symmetric_start_deg,
        "symmetric_jitter_deg": params.symmetric_jitter_deg,
        "thumb_bottom_deg": list(params.thumb_bottom_deg),
        "top_band_deg": list(params.top_band_deg),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_urdf_xacro_palm(xacro_path: str, products: PalmProducts, params: PalmParams):
    """Generate a palm-only URDF/Xacro file with the generated mesh."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    from xml.dom import minidom
    
    # Create robot element
    robot = ET.Element('robot', {
        'xmlns:xacro': 'http://www.ros.org/wiki/xacro',
        'name': 'generated_palm'
    })
    
    # Add comment with generation info
    comment = ET.Comment(f"""
Generated palm mesh with {params.fingers} finger attachment points
Palm parameters: radius={params.palm_radius}, thickness={params.thickness}
Mesh: {Path(xacro_path).with_suffix('.obj').name}
""")
    robot.append(comment)
    
    # Base link
    base_link = ET.SubElement(robot, 'link', {'name': 'base_link'})
    base_inertial = ET.SubElement(base_link, 'inertial')
    ET.SubElement(base_inertial, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    ET.SubElement(base_inertial, 'mass', {'value': '0.5'})
    ET.SubElement(base_inertial, 'inertia', {
        'ixx': '0.01', 'ixy': '0', 'ixz': '0',
        'iyy': '0.01', 'iyz': '0', 'izz': '0.01'
    })
    
    # Palm link with generated mesh
    palm_link = ET.SubElement(robot, 'link', {'name': 'palm'})
    
    # Visual - use relative path that will be updated by batch script
    visual = ET.SubElement(palm_link, 'visual')
    ET.SubElement(visual, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    visual_geom = ET.SubElement(visual, 'geometry')
    mesh_filename = Path(xacro_path).with_suffix('.obj').name
    ET.SubElement(visual_geom, 'mesh', {'filename': f'robot_meshes/visual/{mesh_filename}'})
    
    # Collision (use same mesh for simplicity) - will be updated by batch script
    collision = ET.SubElement(palm_link, 'collision', {'name': 'palm_collision'})
    ET.SubElement(collision, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    collision_geom = ET.SubElement(collision, 'geometry')
    ET.SubElement(collision_geom, 'mesh', {'filename': f'robot_meshes/{mesh_filename}'})
    
    # Inertial
    palm_inertial = ET.SubElement(palm_link, 'inertial')
    ET.SubElement(palm_inertial, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    ET.SubElement(palm_inertial, 'mass', {'value': '0.2'})
    ET.SubElement(palm_inertial, 'inertia', {
        'ixx': '0.001', 'ixy': '0', 'ixz': '0',
        'iyy': '0.001', 'iyz': '0', 'izz': '0.001'
    })
    
    # Fixed joint from base to palm
    joint = ET.SubElement(robot, 'joint', {'name': 'base_to_palm', 'type': 'fixed'})
    ET.SubElement(joint, 'parent', {'link': 'base_link'})
    ET.SubElement(joint, 'child', {'link': 'palm'})
    ET.SubElement(joint, 'origin', {'xyz': '0 0 0', 'rpy': '0 0 0'})
    
    # Write to file
    rough = ET.tostring(robot, encoding='utf-8')
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = pretty.split('\n')
    if lines and lines[0].startswith('<?xml'):
        lines = lines[1:]
    
    with open(xacro_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('\n'.join(lines))


def write_frame_data(frame_path: str, products: PalmProducts):
    """Write frame positioning data in a format suitable for batch scripts."""
    frame_data = {
        'finger_count': len(products.finger_positions),
        'frames': []
    }
    
    for i, (pos, quat) in enumerate(zip(products.finger_positions, products.finger_quaternions), 1):
        w, x, y, z = quat  # MuJoCo wxyz format
        frame_data['frames'].append({
            'finger': i,
            'pos': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
            'quat': {'w': w, 'x': x, 'y': y, 'z': z},
            'pos_str': f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
            'quat_str': f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"
        })
    
    import json
    with open(frame_path, 'w', encoding='utf-8') as f:
        json.dump(frame_data, f, indent=2)


# ----------------------- Orchestration -----------------------

def generate_palm_mesh(params: PalmParams) -> PalmProducts:
    if params.seed is not None:
        random.seed(params.seed)
        np.random.seed(params.seed)

    # 1) Sample finger base positions and yaws
    positions, angles_deg = generate_finger_positions(
        params.fingers, params.palm_radius, params.min_radius_factor, params
    )

    # 2) Motor positions (x,y) at a seed z; horns define a reference height
    motor_positions = [(x, y, params.motor_z_seed) for (x, y) in positions]

    # unit conversions (inches→meters)
    grip_sep_m = params.grip_sep_in * 0.0254
    palm_offset_m = params.palm_offset_in * 0.0254
    min_center_m = params.min_center_in * 0.0254

    # 3) Build grip points and nominal palm_center_z (unless overridden)
    grip_points, palm_center_auto = grip_points_from_motors(
        motor_positions, angles_deg, grip_sep_m, palm_offset_m, params.horn_stack
    )
    add_center_grid(grip_points, min_center_m, center_z=palm_center_auto)

    # 4) Decide palm_center_z based on policy
    if params.z_policy == "auto":
        palm_center_z = palm_center_auto
    elif params.z_policy == "top":
        if params.z_top is None:
            raise ValueError("z_policy 'top' requires --z-top")
        palm_center_z = params.z_top - params.thickness / 2.0
    elif params.z_policy == "center":
        if params.palm_center_z is None:
            raise ValueError("z_policy 'center' requires --palm-center-z")
        palm_center_z = params.palm_center_z
    else:
        raise ValueError("z_policy must be one of: auto, top, center")

    # 5) Hull mesh
    vertices, faces, z_top, z_bottom = build_palm_hull(grip_points, palm_center_z, params.thickness, flip=params.flip_normals)

    # 6) Finger poses at a consistent mount plane (z_top + offset), with q_fix
    q_fix = QFIX_PRESETS[params.qfix_name]
    finger_positions, finger_quaternions = compute_finger_poses(
        positions, angles_deg, z_top, params.mount_offset, q_fix, params
    )

    return PalmProducts(
        positions=positions,
        angles_deg=angles_deg,
        palm_center_z=palm_center_z,
        z_top=z_top,
        z_bottom=z_bottom,
        vertices=vertices,
        faces=faces,
        finger_positions=finger_positions,
        finger_quaternions=finger_quaternions,
    )


def z_constrained_transform(ref_base, ref_tip, new_base, new_yaw_rad):
    """Rotate ONLY about +Z to align old XY direction to new yaw, then translate base→base."""
    p1a = np.array(ref_base, float); p1b = np.array(ref_tip, float)
    p2a = np.array(new_base, float)

    v1 = p1b - p1a
    L  = np.linalg.norm(v1[:2]) if np.linalg.norm(v1[:2]) > 1e-9 else 0.05  # arbitrary if old XY zero
    dir_xy = np.array([math.cos(new_yaw_rad), math.sin(new_yaw_rad), 0.0])

    # yaw delta between old XY and new XY
    v1_xy = np.array([v1[0], v1[1], 0.0]); v1_xy /= (np.linalg.norm(v1_xy) + 1e-12)
    v2_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-12)
    cos_a = float(np.dot(v1_xy, v2_xy))
    sin_a = float(np.cross(v1_xy, v2_xy)[2])
    angle = math.atan2(sin_a, cos_a)

    # quaternion: pure Z rotation (MuJoCo wxyz)
    qz = (math.cos(angle*0.5), 0.0, 0.0, math.sin(angle*0.5))

    # translation (make ref_base land on new_base under Rz)
    c, s = math.cos(angle), math.sin(angle)
    Rz = np.array([[c,-s,0],[s, c,0],[0,0,1]])
    t  = p2a - Rz @ p1a
    return t.tolist(), qz


def compute_finger_poses(positions_xy, angles_deg, z_top, mount_offset, q_fix_wxyz, params):
    """Return finger (pos, quat_wxyz) using Z-constrained transform from SOLIDWORKS reference."""
    finger_positions, finger_quats = [], []
    for (x, y), a_deg in zip(positions_xy, angles_deg):
        new_base = (x, y, z_top + mount_offset)
        yaw = math.radians(a_deg)

        # Z-only alignment from (ref_base→ref_tip) to 'yaw' at 'new_base'
        pos, qz = z_constrained_transform(params.ref_base, params.ref_tip, new_base, yaw)

        # Apply one-time export-axis fix, then yaw (palm frame): q = yaw ⊗ qfix
        q = qmul(qz, q_fix_wxyz)  # both are (w,x,y,z)

        finger_positions.append(tuple(pos))
        finger_quats.append(q)
    return finger_positions, finger_quats

def _ang_diff_deg(a, b):
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d

def _in_arc(angle_deg, lo, hi):
    # supports wrapped arcs: e.g., lo=300, hi=60 means [300..360) U [0..60)
    if lo <= hi:
        return lo <= angle_deg <= hi
    return angle_deg >= lo or angle_deg <= hi

def _sample_with_min_sep(n, lo, hi, min_sep, max_tries=2000):
    """Sample n angles in [lo,hi] (possibly wrapped if lo>hi) with pairwise min_sep (deg)."""
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        cand = random.uniform(0.0, 360.0)
        if lo != 0.0 or hi != 360.0:
            if not _in_arc(cand, lo, hi):
                continue
        if all(_ang_diff_deg(cand, a) >= min_sep for a in out):
            out.append(cand)
    if len(out) < n:
        # relax slightly if needed
        needed = n - len(out)
        for _ in range(needed):
            out.append((lo + hi) * 0.5 if lo <= hi else lo)
    return sorted(out, key=lambda a: (a+360.0)%360.0)

def _symmetric_angles(n, start_deg):
    step = 360.0 / max(1, n)
    return [(start_deg + i*step) % 360.0 for i in range(n)]

def _mirror_angle(a_deg: float) -> float:
    """Reflect across +X: (x, y) -> (x, -y)."""
    return (-a_deg) % 360.0

def _mirror_arc(lo: float, hi: float) -> Tuple[float, float]:
    """Mirror a closed arc [lo..hi] (supports wrap) across +X."""
    # Under reflection, arc endpoints swap order
    return (_mirror_angle(hi), _mirror_angle(lo))

def slot_to_angle_deg(slot: int, S: int) -> float:
    return (360.0 * (slot % S)) / S

def arc_to_slot_mask(lo_deg: float, hi_deg: float, S: int) -> np.ndarray:
    mask = np.zeros(S, dtype=bool)
    for s in range(S):
        a = slot_to_angle_deg(s, S)
        # use existing _in_arc for wrap-aware check
        if _in_arc(a, lo_deg, hi_deg):
            mask[s] = True
    return mask

def sample_slots_with_min_sep(n: int, S: int, min_sep_slots: int, allowed: np.ndarray=None) -> List[int]:
    allowed = np.ones(S, dtype=bool) if allowed is None else allowed.copy()
    chosen = []
    tries = 0
    while len(chosen) < n and tries < 5000:
        tries += 1
        candidates = np.where(allowed)[0]
        if len(candidates) == 0:
            # simple backtrack: reset last choice and relax slightly
            if not chosen: break
            last = chosen.pop()
            # re-enable around last to try different branch
            for d in range(-min_sep_slots+1, min_sep_slots):
                allowed[(last + d) % S] = True
            continue
        s = int(random.choice(candidates))
        chosen.append(s)
        # forbid neighborhood around s
        for d in range(-min_sep_slots+1, min_sep_slots):
            allowed[(s + d) % S] = False
    return chosen


def parse_args() -> Tuple[PalmParams, str]:
    p = argparse.ArgumentParser(description="Generate a convex-hull palm mesh (OBJ)")
    p.add_argument("--out", type=str, default="palm_hull.obj", help="Output OBJ path")
    p.add_argument("--fingers", type=int, default=5, help="Number of fingers to place")
    p.add_argument("--palm-radius", type=float, default=0.12, dest="palm_radius", help="Palm radius (m)")
    p.add_argument("--min-radius-factor", type=float, default=0.9, dest="min_radius_factor",
                   help="Minimum fraction of radius for inner ring placement")

    p.add_argument("--grip-sep-in", type=float, default=1.5, dest="grip_sep_in",
                   help="Grip separation (inches), sets left/right span per finger")
    p.add_argument("--palm-offset-in", type=float, default=0.25, dest="palm_offset_in",
                   help="Offset (inches) from horn plane down to palm center")
    p.add_argument("--min-center-in", type=float, default=3.0, dest="min_center_in",
                   help="Ensure center coverage with a square grid of this size (inches)")

    p.add_argument("--thickness", type=float, default=0.0254, help="Palm slab thickness (m)")
    p.add_argument("--horn-stack", type=float, default=0.038, dest="horn_stack",
                   help="Servo+horn stack-up height (m)")
    p.add_argument("--motor-z-seed", type=float, default=0.030, dest="motor_z_seed",
                   help="Seed motor Z (m) for horn reference")

    p.add_argument("--z-policy", type=str, default="auto", choices=["auto", "top", "center"],
                   help="How to set the palm height: 'auto' from horns, or force 'top'/'center'")
    p.add_argument("--z-top", type=float, default=None, help="Absolute top Z (m) if z-policy=top")
    p.add_argument("--palm-center-z", type=float, default=None, dest="palm_center_z",
                   help="Absolute palm center Z (m) if z-policy=center")
    
    p.add_argument("--mount-offset", type=float, default=0.0015,
                   help="Meters above palm z_top to place finger root")
    p.add_argument("--qfix", type=str, default="identity",
                   choices=list(QFIX_PRESETS.keys()),
                   help="One-time axis correction for finger export frame")
    p.add_argument("--flip-normals", action="store_true",
                   help="Reverse triangle winding if mesh looks inside-out")

    p.add_argument("--ref-base", type=float, nargs=3, metavar=("BX","BY","BZ"),
                   default=[0.019395, 0.03924, 0.0], help="SOLIDWORKS reference base point")
    p.add_argument("--ref-tip", type=float, nargs=3, metavar=("TX","TY","TZ"),
                   default=[0.01838, 0.06902, 0.0], help="SOLIDWORKS reference tip point")

    p.add_argument("--seed", type=int, default=None, help="Random seed")
    
    p.add_argument("--placement-mode", type=str, default="asymmetric",
               choices=["asymmetric", "symmetric", "anthro-top-heavy"])
    p.add_argument("--min-angle-deg", type=float, default=12.0)
    p.add_argument("--symmetric-start-deg", type=float, default=0.0)
    p.add_argument("--symmetric-jitter-deg", type=float, default=0.0)
    p.add_argument("--thumb-bottom-deg", type=float, nargs=2, metavar=("LO","HI"),
                default=[210.0, 330.0])
    p.add_argument("--top-band-deg", type=float, nargs=2, metavar=("LO","HI"),
                default=[300.0, 60.0])  # allow wrap-around (LO>HI)
    p.add_argument("--y-mirrored", action="store_true",
               help="Mirror angles across +X (use if viewer has 180° X-roll)")
    
    p.add_argument("--discrete-slots",     dest="use_discrete_slots", action="store_true")
    p.add_argument("--no-discrete-slots",  dest="use_discrete_slots", action="store_false")
    p.set_defaults(use_discrete_slots=True)  # or False, pick your default

    p.add_argument("--slot-count", type=int, default=36)
    p.add_argument("--min-sep-slots", type=int, default=4)
    p.add_argument("--mount-slots", type=str, default=None,
                help='Comma list of slot indices, e.g. "0,6,12,18,24" (overrides sampling)')
    p.add_argument("--jitter-deg", type=float, default=0.0)
    
    p.add_argument("--anthro-top-fixed-slots", type=str, default=None,
               help='Comma list of slot indices for non-thumb fingers in anthro mode (len=fingers-1)')
    p.add_argument("--thumb-fixed-slot", type=int, default=None,
               help="Fix the thumb to this slot index in anthro mode (discrete slots)")


    args = p.parse_args()
    
    mount_slots = None
    if args.mount_slots:
        mount_slots = [int(s) for s in args.mount_slots.split(",") if s.strip()!=""]
    
    anthro_top_fixed_slots = None
    if args.anthro_top_fixed_slots:
        anthro_top_fixed_slots = [int(s) for s in args.anthro_top_fixed_slots.split(",") if s.strip()!=""]

    
    params = PalmParams(
        fingers=args.fingers,  # Actual number of fingers (not always 5)
        palm_radius=args.palm_radius,
        min_radius_factor=args.min_radius_factor,
        grip_sep_in=args.grip_sep_in,
        palm_offset_in=args.palm_offset_in,
        min_center_in=args.min_center_in,
        thickness=args.thickness,
        horn_stack=args.horn_stack,
        motor_z_seed=args.motor_z_seed,
        seed=args.seed,
        z_policy=args.z_policy,
        z_top=args.z_top,
        palm_center_z=args.palm_center_z,
        mount_offset=args.mount_offset,
        qfix_name=args.qfix,
        flip_normals=args.flip_normals,
        ref_base=tuple(args.ref_base),
        ref_tip=tuple(args.ref_tip),
        placement_mode=args.placement_mode,
        min_angle_deg=args.min_angle_deg,
        symmetric_start_deg=args.symmetric_start_deg,
        symmetric_jitter_deg=args.symmetric_jitter_deg,
        thumb_bottom_deg=tuple(args.thumb_bottom_deg),
        top_band_deg=tuple(args.top_band_deg),
        y_mirrored=args.y_mirrored,
        use_discrete_slots=args.use_discrete_slots,
        slot_count=args.slot_count,
        min_sep_slots=args.min_sep_slots,
        mount_slots=mount_slots,
        jitter_deg=args.jitter_deg,
        anthro_top_fixed_slots=anthro_top_fixed_slots,
        thumb_fixed_slot=args.thumb_fixed_slot,
    )
    return params, args.out


def main():
    params, out_path = parse_args()
    products = generate_palm_mesh(params)

    # Analyze mesh before writing
    outward, inward = analyze_mesh_normals(products.vertices, products.faces)
    
    # Write OBJ
    write_obj(out_path, products.vertices, products.faces)
    
    # Run convex decomposition on the generated OBJ
    print(f"\nRunning convex decomposition on {out_path}...")
    part_files = run_convex_decomposition(out_path, threshold=0.1)
    
    # Write metadata JSON
    base, _ = os.path.splitext(out_path)
    json_path = base + ".json"
    write_metadata(json_path, products, params)
    
    # Write URDF/Xacro palm
    xacro_path = base + ".urdf.xacro"
    write_urdf_xacro_palm(xacro_path, products, params)
    
    # Write frame positioning data
    frame_path = base + "_frames.json"
    write_frame_data(frame_path, products)

    print("\nPalm mesh generated:")
    print(f"  Original OBJ: {out_path}")
    print(f"  Convex parts: {len(part_files)} files")
    for part in part_files:
        print(f"    - {part}")
    print(f"  JSON: {json_path}")
    print(f"  URDF/Xacro: {xacro_path}")
    print(f"  Frame data: {frame_path}")
    print("  Stats:")
    print(f"    fingers        : {params.fingers}")
    print(f"    verts / faces  : {products.vertices.shape[0]} / {products.faces.shape[0]}")
    print(f"    open-edge count: {boundary_edges_count(products.faces)}")
    print(f"    normal analysis: {outward} outward, {inward} inward-facing")
    if inward > outward:
        print("    ⚠️  MESH APPEARS INSIDE-OUT! Try running with --flip-normals")
    print(f"    z_top / z_bottom: {products.z_top:.5f} / {products.z_bottom:.5f}")
    print("  Finger poses:")
    for i, (pos, quat) in enumerate(zip(products.finger_positions, products.finger_quaternions)):
        print(f"    finger {i}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")

    print("\n  Finger poses (MuJoCo wxyz quats):")
    for i, (pos, q) in enumerate(zip(products.finger_positions, products.finger_quaternions), start=1):
        print(f"    finger {i}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
              f"quat=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")

    print("\nPaste these inside <body name=\"palm\"> ... </body> in your hand.xml:")
    for line in make_mjcf_frames(products):
        print(line)


def run_convex_decomposition(obj_path: str, threshold: float = 0.1) -> List[str]:
    """
    Run CoACD convex decomposition on the given OBJ file.
    Replaces the original file with decomposed parts.
    Returns list of generated part file paths.
    """
    try:
        # Load the mesh
        raw = trimesh.load(obj_path, force="mesh")
        if raw is None:
            raise RuntimeError(f"Could not load mesh from '{obj_path}'")

        # Run CoACD decomposition
        parts = coacd.run_coacd(
            coacd.Mesh(raw.vertices, raw.faces),
            threshold=threshold,
            max_convex_hull=-1,
            preprocess_mode='auto',
            preprocess_resolution=50,
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=100,
            mcts_max_depth=3,
            pca=False,
            merge=True,  # not no_merge
            decimate=False,
            max_ch_vertex=256,
            extrude=False,
            extrude_margin=0.01,
            apx_mode='ch',
            seed=36
        )
        if not parts:
            raise RuntimeError(f"CoACD returned no convex parts for '{obj_path}'")

        # Get base name without extension
        base_path = os.path.splitext(obj_path)[0]
        part_paths = []

        # Save each convex hull as a separate part
        for idx, (vs, fs) in enumerate(parts):
            hull = trimesh.Trimesh(vs, fs, process=False)
            part_path = f"{base_path}_part{idx}.obj"
            hull.export(part_path)
            part_paths.append(part_path)
            print(f"   wrote convex part: {part_path}")

        # Remove the original non-decomposed file
        if len(part_paths) > 1:  # Only remove if we got multiple parts
            try:
                os.remove(obj_path)
                print(f"   removed original: {obj_path}")
            except OSError:
                pass

        return part_paths

    except Exception as e:
        msg = (
            f"ERROR: CoACD decomposition failed for '{obj_path}'. "
            f"Hard failing to avoid silently using non-decomposed geometry. Root cause: {e}"
        )
        print(msg)
        raise RuntimeError(msg) from e


if __name__ == "__main__":
    main()
