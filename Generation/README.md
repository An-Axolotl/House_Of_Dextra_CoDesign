# MuJoCo Assets and Generation

This directory contains MuJoCo assets plus the hand generation pipeline used to produce MJCF/URDF artifacts for downstream simulation.

## Generation Documentation

Start here for an overview of the full documentation set:

- `Generation/docs/README.md`  
  High-level map of the conceptual docs and how the CAD-derived and parametric workflows connect.

- `Generation/docs/01_cad_to_mjcf_pipeline.md`  
  Explains the baseline CAD -> MJCF -> URDF/xacro -> USD flow, including joint/frame conventions and collision decomposition.

- `Generation/docs/02_parametric_hand_generator.md`  
  Describes the parametric design space: palm placement modes, grammar-based finger morphology, and ghost-joint strategy.

- `Generation/docs/03_generator_cli_and_experiment_configs.md`  
  Practical CLI guide for generator modes, key flags, and experiment-style command configurations.

- `Generation/converters/README.md`  
  Script-level operational reference for running the pipeline in practice (dependencies, outputs, and command examples).

## Main Entrypoint

Current build entrypoint used by optimization and batch-generation workflows:

- `Generation/converters/batch_build_rand_hands2.sh`
