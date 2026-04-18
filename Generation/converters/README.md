# MuJoCo Converter/Generation Pipeline

This folder contains the hand-asset generation pipeline used by the current optimization/evaluation flow.

## Entrypoint

The main entrypoint is:

`Generation/converters/batch_build_rand_hands2.sh`

It is used during the optimization process by:

- `Main/ghs_runner.py` (which is used by `Main/quick_start_rotation.sh`)

It is also used for pretraining the generalist cross-embodied policy. More details are below.

### GHS eval group modes

`Main/ghs_runner.py` supports both single-group and multi-group evaluation with:
- `--eval-groups all` (default; all 6 groups)
- `--eval-groups <group>` (single group), e.g. `sym5`
- `--eval-groups <g1,g2,...>` (subset), e.g. `sym3,sym4,sym5`

Valid group IDs: `sym3`, `sym4`, `sym5`, `anth21`, `anth27`, `anth33`.


## Script Status

Active (used by current build flow):

- `batch_build_rand_hands2.sh` (top-level orchestrator)
- `generate_palm_mesh.py` (palm mesh + frame generation)
- `generate_rand_gram_joint_finger.py` (finger MJCF generation)
- `mjcf_to_xacro_gram_joints.py` (finger MJCF -> URDF/xacro conversion)
- `write_complete_hand_xml.py` (scene XML assembly for `--save-xml`)
- `write_complete_hand_xacro.py` (full hand xacro assembly)
- `write_hand_metadata.py` (metadata JSON writer)

Generation flow per hand:

1. Sample/parse finger codes and palm settings in `batch_build_rand_hands2.sh`.
2. Generate palm geometry and attachment frames with `generate_palm_mesh.py`.
3. Generate each finger MJCF with `generate_rand_gram_joint_finger.py`.
4. Convert each finger MJCF to URDF xacro with `mjcf_to_xacro_gram_joints.py`.
5. Write complete scene XML with `write_complete_hand_xml.py` (optional, `--save-xml`).
6. Compose full hand xacro with `write_complete_hand_xacro.py`.
7. Render final URDF with `xacrodoc`.
8. Write metadata JSON with `write_hand_metadata.py`.
9. Emit final metadata/paths per generated hand.

## Dependencies

Required:

- `python3`
- `xacrodoc`
- Python packages: `numpy`, `scipy`, `trimesh`, `coacd`
Install required Python deps:

```bash
pip install numpy scipy trimesh coacd
```

Verify dependency installation:

```bash
python Generation/converters/verify_generation_deps.py
```

Expected result: script exits with `[PASS] MuJoCo generator dependency verification succeeded.`
## Output Layout

For `-o <OUT_DIR>`, the generator writes:

- `<OUT_DIR>/*.urdf` final hand URDFs
- `<OUT_DIR>/_build/<hand_name>/...` intermediate per-hand artifacts
- `<OUT_DIR>/meshes/` generated palm mesh parts
- `<OUT_DIR>/robot_meshes/` copied shared finger mesh assets
- `<OUT_DIR>/metadata/*.meta.json` hand metadata
- `<OUT_DIR>/xml/` optional intermediate XMLs when `--save-xml` is enabled

## Examples

Run from repository root:

```bash
cd Generation
```

Basic modes:

```bash
# 1) All uniform combos (g1,g2 in 1..10, servos in 1..3)
./converters/batch_build_rand_hands2.sh -o sets/uniform --mode uniform-all

# 2) Thumb-random over g1,g2 and servos
./converters/batch_build_rand_hands2.sh -o sets/thumb_rand --mode thumb-random --seed 123

# 3) 100 random hands
./converters/batch_build_rand_hands2.sh -o sets/all_rand --mode all-random -n 100

# 4) Manual exact hands (-C can be repeated)
./converters/batch_build_rand_hands2.sh -o sets/manual --mode manual \
  -C "220,220,,," \
  -C "530,200,900,670,440"
```

Frequently used presets:

```bash
# Uniformly random
./converters/batch_build_rand_hands2.sh -o sets/test_hands --mode all-random -n 300 \
  --min-palm-fingers 3 --max-palm-fingers 5 --min-servos 2 --save-xml

# Fixed palm seed, random fingers
./converters/batch_build_rand_hands2.sh -o sets/same_palm_hands --mode all-random -n 100 \
  --palm-seed-mode fixed --palm-base-seed 42 --min-palm-fingers 5 --max-palm-fingers 5

# Same palm, 5 fingers, 3 servo segments
./converters/batch_build_rand_hands2.sh -o sets/g1_palm005_5fing_3servo --mode all-random -n 1 \
  --palm-seed-mode fixed --palm-base-seed 42 --min-palm-fingers 5 --max-palm-fingers 5 \
  --min-servos 3 --save-xml
```

Manual examples used for Sim2Real assets:

```bash
# hand_f1_220_f2_220_f3_f4_f5.urdf (three-finger, all standard tips)
./converters/batch_build_rand_hands2.sh --mode manual -C "220,220,,," \
  --fingertip-type standard --save-xml -o sets/manual

# hand_f1_330_f2_330_f3_330_f4_f5.urdf (five-finger, all standard tips)
./converters/batch_build_rand_hands2.sh --mode manual -C "330,330,330,," \
  --fingertip-type standard --save-xml -o sets/manual

# hand_f1_530_f2_200_f3_900_f4_670_f5_440.urdf (random tips)
./converters/batch_build_rand_hands2.sh --mode manual -C "530,200,900,670,440" \
  --fingertip-random --save-xml -o sets/manual

# hand_f1_330_f2_210_f3_080_f4_750_f5.urdf (random tips)
./converters/batch_build_rand_hands2.sh --mode manual -C "330,210,080,750," \
  --fingertip-random --save-xml -o sets/manual

# hand_f1_640_f2_10_f3_010_f4_80_f5.urdf (random tips)
./converters/batch_build_rand_hands2.sh --mode manual -C "640,10,010,80," \
  --fingertip-random --save-xml -o sets/manual
```

Group generation commands used in past runs:

```bash
# group 2
./converters/batch_build_rand_hands2.sh -o sets/group2 --save-xml --palm-radius 0.07 \
  --mode all-random -n "$COUNT" --min-palm-fingers 4 --max-palm-fingers 4 \
  --min-servos 2 --max-servos 3 --fingertip-random --placement-mode symmetric \
  --symmetric-start-deg 0 --symmetric-jitter-deg 3 --slot-count 36 --min-sep-slots 4 --seed 3

# group 3
./converters/batch_build_rand_hands2.sh -o sets/group3 --save-xml --palm-radius 0.07 \
  --mode all-random -n "$COUNT" --min-palm-fingers 5 --max-palm-fingers 5 \
  --min-servos 2 --max-servos 3 --fingertip-random --placement-mode symmetric \
  --symmetric-start-deg 0 --symmetric-jitter-deg 3 --slot-count 36 --min-sep-slots 4 --seed 3

# group 4
./converters/batch_build_rand_hands2.sh -o sets/group4 --save-xml --palm-radius 0.06 \
  --mode all-random -n "$COUNT" --min-palm-fingers 3 --max-palm-fingers 3 \
  --min-servos 2 --max-servos 3 --fingertip-random --placement-mode symmetric \
  --symmetric-start-deg 0 --symmetric-jitter-deg 3 --slot-count 36 --min-sep-slots 4 --seed 3

# group 7
./converters/batch_build_rand_hands2.sh -o sets/group7 --save-xml --palm-radius 0.07 \
  --mode all-random -n 1500 --min-palm-fingers 5 --max-palm-fingers 5 \
  --min-servos 2 --max-servos 3 --fingertip-random --placement-mode anthro-top-heavy \
  --thumb-bottom-deg 180 360 --top-band-deg 0 180 --slot-count 36 --min-sep-slots 4 \
  --anthro-top-fixed-slots "15,11,7,3" --thumb-fixed-slot 21 --thumb-fixed-servos 3 --seed 3

# group 8
./converters/batch_build_rand_hands2.sh -o sets/group8 --save-xml --palm-radius 0.07 \
  --mode all-random -n 1500 --min-palm-fingers 5 --max-palm-fingers 5 \
  --min-servos 2 --max-servos 3 --fingertip-random --placement-mode anthro-top-heavy \
  --thumb-bottom-deg 180 360 --top-band-deg 0 180 --slot-count 36 --min-sep-slots 4 \
  --anthro-top-fixed-slots "15,11,7,3" --thumb-fixed-slot 27 --thumb-fixed-servos 3 --seed 3

# group 9
./converters/batch_build_rand_hands2.sh -o sets/group9 --save-xml --palm-radius 0.07 \
  --mode all-random -n 1500 --min-palm-fingers 5 --max-palm-fingers 5 \
  --min-servos 2 --max-servos 3 --fingertip-random --placement-mode anthro-top-heavy \
  --thumb-bottom-deg 180 360 --top-band-deg 0 180 --slot-count 36 --min-sep-slots 4 \
  --anthro-top-fixed-slots "15,11,7,3" --thumb-fixed-slot 33 --thumb-fixed-servos 3 --seed 3
```
