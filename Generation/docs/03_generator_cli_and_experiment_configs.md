# Generator CLI and Experiment Configurations

This page describes practical usage of the generator driver:

`Generation/converters/batch_build_rand_hands2.sh`

## 1) Core Invocation Pattern

```bash
./converters/batch_build_rand_hands2.sh \
  -o <out_dir> \
  --mode <uniform-all|thumb-random|all-random|manual> \
  [options...]
```

Typical output in `<out_dir>`:

- `*.urdf` final hand URDFs
- `meshes/` generated palm meshes and convex parts
- `metadata/*.json` sampled per-hand/per-finger configuration
- `xml/` intermediate MJCF/XML emitted when `--save-xml` is enabled

## 2) Sampling Modes

### `--mode all-random`

Sample each finger independently.

- `-n N` number of hands
- `--min M --max M` grammar-code bounds
- `--min-servos K --max-servos K` actuated joint-count bounds

### `--mode uniform-all`

Enumerate `(g1, g2, servos)` combinations and apply the same code to all fingers.

Use this for controlled ablation sweeps with one canonical finger form per hand.

### `--mode thumb-random`

Use shared code for fingers 2-5, randomize thumb code separately.

- thumb index can be changed via `--thumb-index`

### `--mode manual`

Provide explicit per-finger codes through repeated `-C`.

Examples:

```bash
-C "330,330,330,,"        # three active fingers, two ghost slots
-C "530,200,900,670,440"  # five active fingers
```

Empty token positions are treated as ghost fingers. Internally, the build still pads to the maximal kinematic template.

## 3) Palm/Layout Controls

Finger count and palm geometry:

- `--min-palm-fingers Kmin --max-palm-fingers Kmax`
- `--palm-radius R`
- `--palm-thickness T`

Layout mode:

- `--placement-mode symmetric|asymmetric|anthro-top-heavy`

Symmetric controls:

- `--symmetric-start-deg theta0`
- `--symmetric-jitter-deg sigma`

Anthropomorphic controls:

- `--thumb-bottom-deg lo hi`
- `--top-band-deg lo hi`

Slot constraints:

- `--slot-count S`
- `--min-sep-slots D`
- `--anthro-top-fixed-slots "i1,i2,..."`
- `--thumb-fixed-slot j`

## 4) Finger Morphology Controls

Fingertip selection:

- `--fingertip-type standard|wedged|rounded|thinner`
- `--fingertip-random`
- `--fingertip-f1 TYPE ... --fingertip-f5 TYPE`

Servo/ghost behavior:

- `--min-servos`, `--max-servos`
- `--ghost-mode tiny|zero`
- `--ghost-eps-gen eps_gen`
- `--ghost-eps-conv eps_conv`

`tiny` gives locked joints a small nonzero range; `zero` clamps exactly fixed.

## 5) Reproducibility Controls

- `--seed S` global seed
- `--palm-seed-mode random|fixed|incremental`
- `--palm-base-seed S0`

Seed mode behavior:

- `random`: independent palm seeds per hand
- `fixed`: same palm for all generated hands
- `incremental`: deterministic per-hand seed offset from base seed

## 6) Experiment-Style Commands

Symmetric 4-finger family (2-3 servos, randomized fingertips):

```bash
./converters/batch_build_rand_hands2.sh \
  -o sets/group2 --save-xml \
  --palm-radius 0.07 \
  --mode all-random -n "$COUNT" \
  --min-palm-fingers 4 --max-palm-fingers 4 \
  --min-servos 2 --max-servos 3 \
  --fingertip-random \
  --placement-mode symmetric \
  --symmetric-start-deg 0 --symmetric-jitter-deg 3 \
  --slot-count 36 --min-sep-slots 4 \
  --seed 3
```

Anthropomorphic 5-finger family with fixed thumb slot:

```bash
./converters/batch_build_rand_hands2.sh \
  -o sets/group7 --save-xml \
  --palm-radius 0.07 \
  --mode all-random -n 1500 \
  --min-palm-fingers 5 --max-palm-fingers 5 \
  --min-servos 2 --max-servos 3 \
  --fingertip-random \
  --placement-mode anthro-top-heavy \
  --thumb-bottom-deg 180 360 \
  --top-band-deg 0 180 \
  --slot-count 36 --min-sep-slots 4 \
  --anthro-top-fixed-slots "15,11,7,3" \
  --thumb-fixed-slot 21 --thumb-fixed-servos 3 \
  --seed 3
```

## 7) Downstream Conversion

Generated URDFs are intended to be batch-converted into USD with the same URDF->USD path used by the CAD baseline. This preserves one simulator interface while allowing morphology to vary.
