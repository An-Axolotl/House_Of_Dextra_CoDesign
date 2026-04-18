# CAD to MJCF to URDF/USD Pipeline

This page documents the baseline hand construction flow that starts from a constrained CAD model and ends in simulation assets.

## Pipeline Summary

`SolidWorks CAD` -> `per-link meshes + joint measurements` -> `MuJoCo MJCF` -> `URDF/xacro` -> `USD`

Key idea: all early-stage geometry and joint measurements are kept in one common CAD frame, then transformed in a controlled way only when a target format requires it.

## 1) Initial Export From CAD

Start from a fully constrained SolidWorks model of the palm and a canonical finger.

- Export each logical rigid link as one mesh.
- If one link is made from multiple small CAD parts (for example, housing + brackets), merge those parts at export and emit one mesh for that logical link.
- Export all links in the same local CAD coordinate frame.

Why this matters:

- The palm and finger meshes share the same origin and axes.
- In the first MJCF, meshes can be spawned at identity transform.
- No manual re-derivation of relative mesh transforms is needed.

## 2) Joint Origins and Axes From CAD

For each revolute hinge:

- Read the hinge center in SolidWorks and record its position in millimeters.
- Read two points on the hinge line and compute axis direction.
- Normalize the direction vector to get a unit axis.

For each joint (palm/base/middle/end), this yields:

- absolute joint origin in the shared CAD frame
- unit joint axis in the shared CAD frame

These values are written directly into MJCF hinge joints, preserving CAD kinematics exactly.

## 3) Collision Geometry Pass

Visual CAD meshes are often too detailed/non-convex for robust real-time contact.

Collision processing:

1. Convert exported meshes to OBJ if needed.
2. Run convex decomposition (CoACD).
3. Keep decomposed convex pieces as collision geometry.
4. Keep detailed original mesh as visual geometry.

Result:

- stable and efficient contact simulation
- preserved visual fidelity

## 4) Canonical Finger MJCF and Multi-Finger Assembly

Define one canonical finger model (example in this repo: `Generation/meshes/cd_hand/finger1.xml`) with:

- finger root body
- servo/segment chain
- hinge joints with CAD-derived origin + axis
- contact exclusions for intra-finger self collision

Then assemble the full hand (example: `Generation/meshes/cd_hand/hand.xml`) by reusing that finger with `<attach>`:

- register finger submodel in assets
- define one `<frame>` per mount location on palm
- attach finger instances with unique prefixes

Mount transforms are precomputed from canonical finger base to target palm mount points. In this project, these are constrained to z-axis rotations where possible to reduce small numerical misalignment.

## 5) MJCF to URDF/xacro Conversion

Conversion is split conceptually into two stages:

1. **Hand-level stage**
   - Build palm/base links and global transform.
   - Read finger attach frames from hand MJCF.
   - Emit one macro call per finger instance with `xyz/rpy` origin.
2. **Finger-internal stage**
   - Convert the finger body/joint chain itself into a reusable xacro macro.

### Critical Convention Mismatch

MuJoCo and URDF represent joint origins differently:

- MuJoCo joint positions are often expressed in absolute coordinates in the finger root frame.
- URDF joint origins must be relative to each parent link frame.

So for each revolute joint:

`joint_origin_rel = child_joint_pos_abs - nearest_parent_revolute_joint_pos_abs`

### Mesh Compensation Transform

After converting joints to relative coordinates, link frames shift. Mesh origins must be compensated so world-space geometry stays unchanged.

Typical compensation used:

`mesh_origin_offset = - joint_pos_abs_for_that_link`  
(or inherited parent reference for fixed grammar bodies)

This keeps visual and collision geometry aligned with the original MJCF placement.

## 6) Finger Macro Structure and Ghost-Compatible Links

The generated finger xacro macro generally defines:

- `finger_root` and a fixed mount joint (origin supplied by caller)
- revolute joints with converted relative origins
- fixed grammar-body joints where needed
- small nonzero inertial properties

This structure makes the finger reusable across multiple palm mount frames while preserving kinematics and geometry.

## 7) Finalization and Validation

1. Expand xacro into URDF (example output: `Generation/meshes/urdf/hand.urdf`).
2. Verify geometry alignment and joint behavior in a URDF viewer.
3. Batch convert URDF to USD for IsaacLab/IsaacGym.

The final USDs are used for training/evaluation, while consistency is maintained across all intermediate representations.

## Related Files in This Repo

- `Generation/meshes/cd_hand/hand.xml`
- `Generation/meshes/cd_hand/finger1.xml`
- `Generation/meshes/urdf/hand.urdf.xacro`
- `Generation/converters/mjcf_to_xacro_gram_joints.py`
- `Generation/converters/write_complete_hand_xacro.py`
