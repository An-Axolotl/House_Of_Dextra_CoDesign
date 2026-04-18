# MuJoCo Hand Generation Docs

This folder explains the generation pipeline in detail without expanding the top-level README files.

## Document Map

- `Generation/docs/01_cad_to_mjcf_pipeline.md`
  - SolidWorks/CAD export, MJCF modeling, MJCF-to-URDF/xacro conversion, and USD handoff.
- `Generation/docs/02_parametric_hand_generator.md`
  - Design-space definition for palm layout and per-finger morphology.
- `Generation/docs/03_generator_cli_and_experiment_configs.md`
  - Practical usage of `batch_build_rand_hands2.sh`, flag semantics, and experiment-style command lines.
- `Generation/converters/README.md`
  - Script-oriented quick reference and runnable command examples.

## At A Glance

Two related workflows are used in this repo:

1. **CAD-derived baseline hand**
   - SolidWorks geometry/joints -> MJCF -> URDF/xacro -> USD.
2. **Parametric random hand generator**
   - Sample palm + finger morphology -> generate MJCF/xacro/URDF -> USD.

Both workflows converge to URDF, then share the same URDF-to-USD asset path used by IsaacLab/IsaacGym experiments.
