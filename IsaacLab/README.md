# IsaacLab

This directory contains the Isaac Lab side of the CoDesign-InHand workflow:

- the reorientation task used for training and evaluation,
- the hand asset configuration in `source/codesign/codesign/assets/lego_hand.py`,
- and the RL-Games train/play entrypoints in `scripts/rl_games/`.

For Isaac Sim installation and repo bootstrap steps, start with [`../SETUP.md`](../SETUP.md).

## Main Scripts

Run these with the Isaac Sim Python that `setup.sh` configures:

```bash
cd /root/CoDesign-InHand
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/train.py --task Codesign-Reorientation-Direct-v0 --headless
```

```bash
cd /root/CoDesign-InHand
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/play.py --task Codesign-Reorientation-Direct-v0 --checkpoint /abs/path/to/checkpoint.pth
```

Useful flags:

- `--num_envs <N>`: override the number of parallel environments.
- `--video`: record a rollout.
- `--headless`: run without the viewer.
- `--device cuda:0`: choose a specific device.

Training logs are written under the current working directory in `logs/rl_games/...`.

## Hand Selection

Hand selection is defined in `source/codesign/codesign/assets/lego_hand.py`.

By default, it scans the asset root and collects any hand that follows:

```text
<hand_dir>/<hand_dir>.usd
```

The default scan root is:

```text
IsaacLab/source/codesign/codesign/assets/group4
```

You can now override the hand selection directly from the RL-Games CLI.

### Single Hand By Name

```bash
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/train.py \
  --task Codesign-Reorientation-Direct-v0 \
  --headless \
  --hand hand_f1_110_f2_110_f3_110_f4_110_f5_110
```

### Single Hand By USD Path

```bash
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/play.py \
  --task Codesign-Reorientation-Direct-v0 \
  --checkpoint /abs/path/to/checkpoint.pth \
  --hand /abs/path/to/hand_f1_110_f2_110_f3_110_f4_110_f5_110.usd
```

### Switch Asset Roots

Use `--hand-assets-dir` when the hand lives outside the default `group4` asset folder:

```bash
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/train.py \
  --task Codesign-Reorientation-Direct-v0 \
  --headless \
  --hand-assets-dir IsaacLab/source/codesign/codesign/assets/lego_hand_v2 \
  --hand hand_f1_110_f2_110_f3_110_f4_110_f5_110
```

### Multiple Hands In Parallel

```bash
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/play.py \
  --task Codesign-Reorientation-Direct-v0 \
  --checkpoint /abs/path/to/checkpoint.pth \
  --hand-assets-dir IsaacLab/source/codesign/codesign/assets/lego_hand_v2 \
  --hands hand_a hand_b hand_c
```

When multiple hands are selected, `lego_hand.py` passes all matching USDs to `MultiUsdFileCfg`, and Isaac Lab samples from that list across environments.

## Notes

- `--hand` accepts a hand name, a hand directory, or a direct `.usd` path.
- `--hands` accepts multiple names, directories, or `.usd` paths.
- If a selected hand has metadata sidecars, `reorientation_env.py` will load them from:
  - `metadata/<hand_name>.meta.json`
  - `meshes/<hand_name>_palm.json`
- If you do not pass `--hand` or `--hands`, the current asset root is scanned and all matching hands are used.
