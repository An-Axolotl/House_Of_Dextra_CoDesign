# Parametric Hand Generator Design

This page explains the generator design space beyond the fixed CAD-derived hand.

Primary scripts:

- `Generation/converters/generate_palm_mesh.py`
- `Generation/converters/generate_rand_gram_joint_finger.py`
- `Generation/converters/mjcf_to_xacro_gram_joints.py`
- orchestrated by `Generation/converters/batch_build_rand_hands2.sh`

## 1) What Is Being Randomized

The generator spans two coupled design dimensions:

1. **Palm layout**
   - number of fingers
   - angular/radial base placement around a palm radius
   - placement mode (symmetric, asymmetric, anthropomorphic)
2. **Per-finger morphology**
   - number of actuated joints beyond base
   - grammar segment lengths
   - fingertip geometry type

## 2) Palm Layout Sampling

Finger bases are sampled on a circular layout with a slot-based constraint model.

- Circle is quantized into `S` discrete slots.
- Active finger slots are sampled with rejection.
- Minimum separation `D` (in slots) is enforced.

This prevents physically bad clustering while still allowing stochastic variation.

Supported layout modes include:

- `symmetric`
- `asymmetric`
- `anthro-top-heavy` (thumb in a bottom arc, remaining fingers in a top arc)

Optional fixed-slot controls let you hold specific mount locations constant (for reproducible anthropomorphic families).

## 3) Palm Mesh Construction

`generate_palm_mesh.py` builds a watertight convex palm mesh from sampled finger mounts.

High-level method:

1. Sample per-finger base positions in an annulus.
2. Lift points to 3D using motor/horn stack assumptions.
3. Create lateral grip points plus central support points.
4. Compute a 2D convex hull in `(x, y)`.
5. Extrude hull across palm thickness to form a convex slab.

Because the result is convex and simple, it is robust for contacts and downstream decomposition.

## 4) Generated Metadata and Attach Frames

The palm generator writes more than geometry:

- JSON metadata with sampled angles, slots, z extents, and mount poses
- per-finger mount transforms `(position, quaternion)`
- ready-to-paste MuJoCo `<frame>` and `<attach>` snippets
- palm-only URDF/xacro wrapper for conversion/inspection

This metadata enables direct pairing of generated palms with reusable finger models.

## 5) Per-Finger Grammar and Tip Variants

For each finger, the generator can vary:

- servo count (0 to 3 extra actuated segments, depending on mode/config)
- grammar lengths for proximal/distal sections
- fingertip type (`standard`, `wedged`, `rounded`, `thinner`)

These discrete choices create a large combinatorial hand family, from short rigid fingers to longer dexterous ones.

## 6) Ghost Joints and Uniform Kinematic Trees

IsaacLab training typically expects consistent joint ordering/dimensionality across a batch. The generator enforces this with a maximal template:

- base + three additional hinge levels per finger

If a sampled finger uses fewer active servos:

- unused joints remain in the model as ghost/locked joints
- ghost links are lightweight, often collision-disabled

Effect:

- all generated hands keep a uniform action/state interface
- actual morphology still varies through which joints are movable and through geometry parameters

## 7) Why This Structure Is Useful

The design supports morphology-conditioned policy training at scale:

- common code path for simulation/control
- broad mechanical variation in one asset family
- reproducible regeneration from seeds + metadata

This is how fixed hand groups used in experiments are treated: each group is a selected slice of a broader configurable design space, not a hard-coded one-off template.
