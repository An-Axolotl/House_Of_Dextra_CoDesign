# Setup

This document collects the environment setup steps for CoDesign-InHand.

`setup.sh` prepares the repo and the Python environments, but it does **not** install Isaac Sim itself. Isaac Sim needs to be installed first.

## Assumptions

- Linux, with Ubuntu 22.04 recommended.
- NVIDIA GPU and a compatible driver.
- Isaac Sim 5.0.
- This repo cloned at:

```bash
export REPO_ROOT="$HOME/CoDesign-InHand"
```

- External Isaac Lab checkout at:

```bash
export ISAACLAB_EXTERNAL="${ISAACLAB_EXTERNAL:-$HOME/isaaclab}"
export ISAACLAB_DIR="${ISAACLAB_DIR:-$ISAACLAB_EXTERNAL}"
```

Note: `setup.sh` honors `ISAACLAB_DIR` and `ISAACLAB_EXTERNAL`. If neither is set, it prefers `/workspace/isaaclab` when that directory already exists. Otherwise it uses `$HOME/isaaclab`.

## 1. Install Basic System Packages

If this is a minimal machine or container, install the tools needed for the rest of setup.

Install them only if they are not already present:

```bash
missing_packages=()
for cmd in python3 wget curl unzip git; do
  command -v "$cmd" >/dev/null 2>&1 || missing_packages+=("$cmd")
done

if ! command -v git-lfs >/dev/null 2>&1; then
  missing_packages+=("git-lfs")
fi

dpkg -s python3-venv >/dev/null 2>&1 || missing_packages+=("python3-venv")

if [ ${#missing_packages[@]} -gt 0 ]; then
  apt-get update
  apt-get install -y "${missing_packages[@]}"
fi
```

If you prefer a simple unconditional install, this also works:

```bash
apt-get update
apt-get install -y python3 python3-venv wget curl unzip git git-lfs \
  libgl1 libegl1 libvulkan1 libglu1-mesa vulkan-tools mesa-utils
```

If you see `python3: command not found`, `wget: command not found`, or `unzip: command not found`, this is the step you are missing.

## 2. Install Isaac Sim 5.0

For a headless machine, you can download the Isaac Sim 5.0 Linux standalone zip directly from the command line.

Example install location:

```bash
export ISAACSIM_ROOT="$HOME/isaacsim"
mkdir -p "$ISAACSIM_ROOT"
```

Recommended: run the compatibility checker first:

```bash
mkdir -p "$HOME/isaacsim-downloads"
cd "$HOME/isaacsim-downloads"
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-comp-check-5.0.0-linux-x86_64.zip
unzip -o isaac-sim-comp-check-5.0.0-linux-x86_64.zip -d isaac-sim-comp-check
cd isaac-sim-comp-check
./omni.isaac.sim.compatibility_check.sh
```

If you are running as `root` in a minimal container, rerun it with the root flag and only install the missing graphics runtime packages if they are not already present:

```bash
cd "$HOME/isaacsim-downloads/isaac-sim-comp-check"

missing_runtime_packages=()
for pkg in libgl1 libegl1 libvulkan1 libglu1-mesa vulkan-tools mesa-utils; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing_runtime_packages+=("$pkg")
done

if [ ${#missing_runtime_packages[@]} -gt 0 ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_runtime_packages[@]}"
fi

mkdir -p /tmp/xdg-runtime-root
chmod 700 /tmp/xdg-runtime-root

OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg-runtime-root \
  ./omni.isaac.sim.compatibility_check.sh --allow-root
```

Use that rerun when:

- the checker complains about running as root,
- or it logs errors like `libGL.so.1: cannot open shared object file`, `libGLU.so.1: cannot open shared object file`, or `vkCreateInstance failed`.

On the tested NVIDIA A10 pod, installing those packages changed the compatibility result from `FAILED` to `PASSED`.

Then download the full Isaac Sim 5.0 binary:

```bash
mkdir -p "$HOME/isaacsim-downloads"
cd "$HOME/isaacsim-downloads"
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip
```

Unpack and initialize Isaac Sim:

```bash
unzip -o "$HOME/isaacsim-downloads/isaac-sim-standalone-5.0.0-linux-x86_64.zip" -d "$ISAACSIM_ROOT"
cd "$ISAACSIM_ROOT"
./post_install.sh
test -x "$ISAACSIM_ROOT/python.sh"
"$ISAACSIM_ROOT/python.sh" --version
./isaac-sim.sh --help
```

If `python.sh` or `post_install.sh` is still missing after unzip, check where the archive landed:

```bash
find "$ISAACSIM_ROOT" -maxdepth 2 \( -name python.sh -o -name post_install.sh \)
```

If `wget` is unavailable, `curl` works too:

```bash
mkdir -p "$HOME/isaacsim-downloads"
cd "$HOME/isaacsim-downloads"
curl -L -O https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip
```

If `wget` or `curl` returns an HTTP permission error on your network, open the official download page from a browser session with your NVIDIA account and use the Linux download link from there.

## 3. Prepare the External Isaac Lab Checkout

This repo already contains a project-specific [`IsaacLab/`](./IsaacLab/) directory used by the CoDesign code and scripts.

However, `setup.sh` also uses a full external Isaac Lab checkout for the upstream bootstrap helpers such as `isaaclab.sh`. That external checkout lives at `$ISAACLAB_DIR`.

You do not need to run the commands in this section manually if you are using the current `setup.sh`. The script will clone or update `$ISAACLAB_DIR` and link `_isaac_sim` automatically.

Use the commands below only if you want to pre-create or manually verify the external checkout before running `setup.sh`.

If you opened a new shell, re-export the paths before linking anything:

```bash
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"
export ISAACLAB_EXTERNAL="${ISAACLAB_EXTERNAL:-$HOME/isaaclab}"
export ISAACLAB_DIR="${ISAACLAB_DIR:-$ISAACLAB_EXTERNAL}"
```

Clone Isaac Lab if it is not already present:

```bash
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  git clone --branch main --single-branch https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
```

Point that checkout at the Isaac Sim install:

```bash
test -x "$ISAACSIM_ROOT/python.sh"
ln -sfn "$ISAACSIM_ROOT" "$ISAACLAB_DIR/_isaac_sim"
```

Verify that Isaac Lab can see Isaac Sim Python:

```bash
"$ISAACLAB_DIR/_isaac_sim/python.sh" --version
```

If this file does not exist, `setup.sh` now stops early with a clearer error instead of failing later during Isaac Lab package installation.

## 4. Run Repo Setup

Run the repo setup from the CoDesign-InHand root:

```bash
cd "$REPO_ROOT"
./setup.sh
```

The script will:

- install apt packages used by the repo,
- create or reuse the runner virtual environment at `$HOME/.venvs/codesign`,
- install runner/build dependencies into that venv,
- install repo packages into Isaac Sim Python,
- and write `.codesign_env.sh` for future shells.

## 5. Load the Runner Virtual Environment

After `setup.sh` finishes:

```bash
source "$REPO_ROOT/.codesign_env.sh"
```

`setup.sh` is rerun-safe by default now:

- it only installs missing apt packages,
- it skips `git pull` on existing repos unless `UPDATE_EXISTING_REPOS=1`,
- it skips heavyweight Isaac Lab / Isaac Sim reinstalls on reruns unless explicitly forced,
- it refuses to mutate the live Isaac install if an Isaac/Kit process is already running,
- and it strips any inherited active venv/conda Python activation before calling the upstream Isaac Lab installer.

Force flags if you intentionally want a heavier rerun:

```bash
UPDATE_EXISTING_REPOS=1 ./setup.sh
FORCE_REINSTALL_ISAACLAB_PACKAGES=1 ./setup.sh
FORCE_REINSTALL_ISAACSIM_PYTHON_DEPS=1 ./setup.sh
ALLOW_ACTIVE_ISAAC_PROCESSES=1 ./setup.sh
```

Quick check:

```bash
echo "$VIRTUAL_ENV"
python --version
command -v python
command -v xacrodoc
```

Expected runner venv:

```text
$HOME/.venvs/codesign
```

## 6. Verify IsaacLab Commands

Once setup is complete, the Isaac Lab scripts should run through Isaac Sim Python:

```bash
cd "$REPO_ROOT"
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/train.py --help
"$CODESIGN_ISAAC_PYTHON" IsaacLab/scripts/rl_games/play.py --help
```

## 7. Tested Version Matrix

The setup and quick-start flow in this repo was last validated on `2026-03-28` with the following versions. If a future update breaks the pod, compare against this matrix first.

Important: this project uses two different Python environments on purpose.

- The runner venv is used for design generation and orchestration.
- Isaac Sim Python is used for RL training, URDF->USD conversion, and `play.py`.
- It is normal for these two environments to have different package versions.

Validated platform:

| Component | Version |
| --- | --- |
| Ubuntu | `22.04.5 LTS` |
| GPU | `NVIDIA A10` |
| NVIDIA driver | `590.48.01` |
| CoDesign repo branch | `sim2real` |
| CoDesign repo commit | `ed0e4584f978db7e78dcddf693cfe021ec92975c` |
| External IsaacLab branch | `main` |
| External IsaacLab commit | `f4aa17f87e2e5db5484f0b5974918573e8918ce2` |
| Isaac Sim standalone | `5.0.0` |
| Isaac Sim Python | `Python 3.11.13` |
| Runner venv Python | `Python 3.10.12` |

Runner venv packages:

| Package | Version |
| --- | --- |
| `torch` | `2.11.0+cu130` |
| `numpy` | `2.2.6` |
| `scipy` | `1.15.3` |
| `trimesh` | `4.11.5` |
| `coacd` | `1.0.7` |
| `xacrodoc` | `2.0.0` |
| `wandb` | `0.25.1` |

Isaac Sim Python packages:

| Package | Version |
| --- | --- |
| `torch` | `2.7.0+cu128` |
| `numpy` | `1.26.0` |
| `scipy` | `1.15.3` |
| `trimesh` | `4.5.1` |
| `coacd` | `1.0.7` |
| `xacrodoc` | `2.0.0` |
| `wandb` | `0.25.1` |
| `rl-games` | `1.6.1` |
| `gym` | `0.23.1` |
| `gymnasium` | `1.2.1` |
| `isaaclab` | `0.54.3` |
| `isaaclab_rl` | `0.5.0` |
| `isaaclab_mimic` | `1.0.16` |

System packages that mattered for earlier pod failures:

| Package | Version |
| --- | --- |
| `python3` | `3.10.6-1~22.04.1` |
| `python3-venv` | `3.10.6-1~22.04.1` |
| `libgl1` | `1.4.0-1` |
| `libegl1` | `1.4.0-1` |
| `libglu1-mesa` | `9.0.2-1` |
| `git-lfs` | `3.0.2-1ubuntu0.3` |

Runtime libraries that must exist even if the exact package source differs by base image:

- `libGL.so.1`
- `libEGL.so.1`
- `libGLU.so.1`
- `libvulkan.so.1`

## Common Issues

### `unzip: command not found`

Install it first:

```bash
apt-get update
apt-get install -y unzip
```

### `cd: ~/Downloads: No such file or directory`

That machine just does not have a `Downloads` directory. Use the real location of the Isaac Sim zip file instead.

### `ln: failed to create symbolic link '.../_isaac_sim' -> '': No such file or directory`

`ISAACSIM_ROOT` was empty in that shell, so `ln` had no source path to point at.

Re-export it and verify it contains Isaac Sim before linking:

```bash
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"
echo "$ISAACSIM_ROOT"
test -x "$ISAACSIM_ROOT/python.sh"
```

### `Unable to find any Python executable at path: .../_isaac_sim/python.sh`

Isaac Sim is either not installed yet, or Isaac Lab is not linked to it correctly.

Check:

```bash
ls -la "$ISAACSIM_ROOT"
ls -la "$ISAACLAB_DIR/_isaac_sim"
ls -la "$ISAACLAB_DIR/_isaac_sim/python.sh"
```

Then recreate the symlink if needed:

```bash
test -x "$ISAACSIM_ROOT/python.sh"
ln -sfn "$ISAACSIM_ROOT" "$ISAACLAB_DIR/_isaac_sim"
```

If `"$ISAACSIM_ROOT"` exists but is nearly empty, you likely created the directory before unzipping Isaac Sim into it. Unpack the standalone zip there, run `./post_install.sh`, and verify `python.sh` exists before recreating the symlink.

### `quick_start_rotation.sh` builds URDFs but fails during `Converting all URDFs to USD in batch`

If the run gets past URDF generation and then fails with symptoms like:

- `libGLU.so.1: cannot open shared object file`
- `No valid USD files found after conversion!`
- `Found 0 USD files total`
- `malloc_consolidate(): unaligned fastbin chunk detected`

then the container is usually still missing part of Isaac Sim's graphics runtime userspace.

On older pods that were created before the latest `setup.sh` changes, install the missing package and rerun setup:

```bash
apt-get update
apt-get install -y libglu1-mesa

cd "$REPO_ROOT"
./setup.sh
```

If you are bootstrapping a fresh pod from the current repo state, `setup.sh` already installs `libgl1`, `libegl1`, `libvulkan1`, `libglu1-mesa`, `vulkan-tools`, and `mesa-utils` for you.

The current repo also includes the Isaac Sim 5.0 URDF-to-USD compatibility fixes in the converter and runner, so if you still see this error on the latest checkout, the first thing to verify is that the runtime packages above are actually installed.

### `ModuleNotFoundError: No module named 'isaaclab.utils.pretrained_checkpoint'`

This usually means the local `IsaacLab/scripts/*/play.py` script is older than the external Isaac Lab checkout installed in `$ISAACLAB_DIR`.

In newer Isaac Lab versions, the pretrained checkpoint helper moved from:

```python
isaaclab.utils.pretrained_checkpoint
```

to:

```python
isaaclab_rl.utils.pretrained_checkpoint
```

The current repo includes a compatibility fallback for both module paths. If you still see this error, update to the latest repo state and rerun:

```bash
cd "$REPO_ROOT"
./setup.sh
```

## References

- Isaac Sim 5.0 download page: https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html
- Isaac Sim 5.0 workstation installation: https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_workstation.html
- Isaac Lab installation overview: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- Isaac Lab pip-install guide: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
