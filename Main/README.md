# Main Directory

This directory contains the Graph Heuristic Search (GHS) orchestration layer for CoDesign-InHand.
It is responsible for:

- sampling candidate hand morphologies,
- invoking the `Generation` pipeline to build assets,
- converting assets for simulation,
- evaluating designs in IsaacLab with pretrained checkpoints,
- logging/search bookkeeping and result packaging.

The main entrypoint for routine runs is:

- `./quick_start_rotation.sh`

## What `./quick_start_rotation.sh` Does

At runtime, the script:

1. Validates expected directory structure (`../IsaacLab`, `../Generation`).
2. Checks required scripts exist:
   - `../Generation/converters/batch_build_rand_hands2.sh`
   - `../IsaacLab/scripts/rl_games/play.py`
3. Verifies Python/Torch availability in the active environment.
4. Exports runtime environment variables for GHS evaluation behavior.
5. Launches `ghs_runner.py` with a configured set of search/evaluation flags.
6. Writes outputs under `Main/ghs_rotation_optimization/run_<timestamp>/`.

## Expected Layout From `Main/`

Run this script from inside `Main/`:

```bash
cd Main
./quick_start_rotation.sh
```

The script expects sibling directories:

- `../IsaacLab`
- `../Generation`

## Key Configuration Knobs

Open `quick_start_rotation.sh` and adjust the top-level config variables:

- `ITERATIONS`: number of GHS iterations.
- `CANDIDATES`: candidate designs evaluated per iteration.
- `EVAL_EPISODES`: episodes per design evaluation.
- `EVAL_ENVS`: number of parallel IsaacLab environments.
- `PLACEMENT_MODE`: hand placement strategy (`symmetric`, `anthro-top-heavy`, `both` via runner behavior).
- `EVAL_GROUPS`: evaluation groups (`all` or subset list).
- `SHUFFLE_GROUPS`: group-order reshuffling behavior for multi-group cycles.
- `EPSILON`: epsilon-greedy exploration rate.
- `BATCH_SIZE`: optimization batch size.
- `DEBUG`: optional debug mode.
- `OUTPUT_DIR`: run artifact path root.

## Environment Notes

This workflow uses two Python contexts:

- Runner Python:
  - executes `ghs_runner.py` and search logic.
- Isaac Python:
  - executes IsaacLab-facing scripts (e.g., `play.py`, URDF->USD conversion).

Isaac interpreter resolution is handled by `Main/utils/isaac_python.py` and can be overridden with:

- CLI: `--isaac-python <path>`
- env vars: `CODESIGN_ISAAC_PYTHON` or `ISAAC_SIM_PYTHON`

## Outputs

Each run creates a timestamped folder in:

- `Main/ghs_rotation_optimization/run_<timestamp>/`

Typical artifacts include:

- per-iteration logs and intermediate build outputs,
- `final_results/` summary files,
- best/worst design exports,
- optional evaluation videos (when enabled),
- generated helper script(s) for best-design re-evaluation.

## Related Files

- `ghs_runner.py`: primary orchestration and evaluation loop.
- `graph_heuristic_search.py`: core GHS search logic.
- `hand_groups.py`: evaluation group definitions.
- `quick_start_rotation.sh`: main run script.
