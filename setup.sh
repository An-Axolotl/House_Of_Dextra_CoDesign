#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------------------
# config
# ----------------------------
export DEBIAN_FRONTEND=noninteractive

# CoDesign repo/settings
REPO_BRANCH="${REPO_BRANCH:-sim2real}"
REPO_URL="${REPO_URL:-https://github.com/An-Axolotl/CoDesign-InHand.git}"
GIT_PAT="${GIT_PAT:-}"

# IsaacLab repo/settings
ISAACLAB_REPO_URL="${ISAACLAB_REPO_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_BRANCH="${ISAACLAB_BRANCH:-main}"
ISAACLAB_EXTERNAL="${ISAACLAB_EXTERNAL:-}"
UPDATE_EXISTING_REPOS="${UPDATE_EXISTING_REPOS:-0}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(dirname "$SCRIPT_DIR")}"
CO_DESIGN_DIR="${CO_DESIGN_DIR:-$SCRIPT_DIR}"
if [[ -z "${ISAACLAB_DIR:-}" ]]; then
  if [[ -n "$ISAACLAB_EXTERNAL" ]]; then
    ISAACLAB_DIR="$ISAACLAB_EXTERNAL"
  elif [[ -d /workspace/isaaclab ]]; then
    ISAACLAB_DIR="/workspace/isaaclab"
  else
    ISAACLAB_DIR="$WORKDIR/isaaclab"
  fi
fi
CO_DESIGN_ISAACLAB_DIR="${CO_DESIGN_ISAACLAB_DIR:-$CO_DESIGN_DIR/IsaacLab}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-}"
ISAACSIM_SH="${ISAACSIM_SH:-$ISAACLAB_DIR/_isaac_sim/python.sh}"
REQUIRE_ISAACSIM="${REQUIRE_ISAACSIM:-1}"

# Legacy conda env
ENV_NAME="${ENV_NAME:-catchingbot}"
CONDA_PY="/opt/conda/envs/${ENV_NAME}/bin/python3"
CONDA_BIN="/opt/conda/envs/${ENV_NAME}/bin"

# Runner/build virtual environment
VENV_NAME="${VENV_NAME:-codesign}"
VENV_DIR="${VENV_DIR:-$WORKDIR/.venvs/${VENV_NAME}}"
VENV_BASE_PYTHON="${VENV_BASE_PYTHON:-}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-1}"
FORCE_RECREATE_VENV="${FORCE_RECREATE_VENV:-0}"
PERSIST_SHELL_CONFIG="${PERSIST_SHELL_CONFIG:-1}"
CODESIGN_ENV_FILE="${CODESIGN_ENV_FILE:-$CO_DESIGN_DIR/.codesign_env.sh}"
TORCH_PIP_SPEC="${TORCH_PIP_SPEC:-torch}"
INSTALL_HOST_PYTHON_DEPS="${INSTALL_HOST_PYTHON_DEPS:-0}"
INSTALL_CONDA_DEPS="${INSTALL_CONDA_DEPS:-0}"
SKIP_HEAVY_ON_RERUN="${SKIP_HEAVY_ON_RERUN:-1}"
FORCE_REINSTALL_ISAACLAB_PACKAGES="${FORCE_REINSTALL_ISAACLAB_PACKAGES:-0}"
FORCE_REINSTALL_ISAACSIM_PYTHON_DEPS="${FORCE_REINSTALL_ISAACSIM_PYTHON_DEPS:-0}"
ALLOW_ACTIVE_ISAAC_PROCESSES="${ALLOW_ACTIVE_ISAAC_PROCESSES:-0}"
SETUP_STAMP_DIR="${SETUP_STAMP_DIR:-$CO_DESIGN_DIR/.setup-stamps}"

VENV_PY=""
VENV_BIN=""
ISAACSIM_READY=0

# ----------------------------
# logging/helpers
# ----------------------------
log()  { echo "[setup] $*"; }
warn() { echo "[setup][WARN] $*" >&2; }
die()  { echo "[setup][ERROR] $*" >&2; exit 1; }

trap 'echo "[setup][ERROR] failed at line $LINENO: $BASH_COMMAND" >&2' ERR

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

stamp_path() {
  local name="$1"
  printf '%s/%s.stamp' "$SETUP_STAMP_DIR" "$name"
}

have_stamp() {
  [[ -f "$(stamp_path "$1")" ]]
}

write_stamp() {
  local name="$1"
  mkdir -p "$SETUP_STAMP_DIR"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$(stamp_path "$name")"
}

have_dpkg_package() {
  dpkg -s "$1" >/dev/null 2>&1
}

install_apt_packages_if_missing() {
  local -a requested=("$@")
  local -a missing=()
  local pkg

  for pkg in "${requested[@]}"; do
    if ! have_dpkg_package "$pkg"; then
      missing+=("$pkg")
    fi
  done

  if ((${#missing[@]} == 0)); then
    log "Required apt packages are already installed; skipping apt install"
    return 0
  fi

  log "Installing missing apt packages: ${missing[*]}"
  apt-get update -y
  apt-get install -y --no-install-recommends "${missing[@]}"
}

list_active_isaac_processes() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi

  pgrep -fa '(/isaacsim/python\.sh|/isaacsim/kit/python/bin/python3|isaaclab\.sh|omni\.isaac|Isaac-Sim)' || true
}

ensure_no_active_isaac_processes() {
  local active

  [[ "$ALLOW_ACTIVE_ISAAC_PROCESSES" == "0" ]] || return 0

  active="$(list_active_isaac_processes)"
  [[ -z "$active" ]] && return 0

  cat >&2 <<EOF
[setup][ERROR] Active Isaac/Kit processes were detected, so setup.sh is refusing to mutate the live install.
This rerun-safety guard prevents pod instability while Isaac Sim is running.

Active processes:
$active

Stop the active Isaac/Kit jobs first, then rerun setup.sh.
If you intentionally want to override this guard, rerun with:
  ALLOW_ACTIVE_ISAAC_PROCESSES=1 ./setup.sh
EOF
  exit 1
}

git_auth_url() {
  local url="$1"
  if [[ -n "$GIT_PAT" && "$url" =~ ^https://github\.com/(.*)$ ]]; then
    echo "https://x-access-token:${GIT_PAT}@github.com/${BASH_REMATCH[1]}"
  else
    echo "$url"
  fi
}

resolve_isaacsim_root() {
  local -a raw_candidates=()
  local candidate
  local resolved

  [[ -n "$ISAACSIM_ROOT" ]] && raw_candidates+=("$ISAACSIM_ROOT")
  [[ -e "$ISAACLAB_DIR/_isaac_sim" || -L "$ISAACLAB_DIR/_isaac_sim" ]] && raw_candidates+=("$ISAACLAB_DIR/_isaac_sim")
  raw_candidates+=("$WORKDIR/isaacsim" "$HOME/isaacsim" "/isaac-sim" "/opt/nvidia/isaac-sim")

  declare -A seen=()
  for candidate in "${raw_candidates[@]}"; do
    resolved="$(readlink -f "$candidate" 2>/dev/null || true)"
    [[ -n "$resolved" && -d "$resolved" ]] || continue
    [[ -n "${seen[$resolved]:-}" ]] && continue
    seen["$resolved"]=1
    if [[ -x "$resolved/python.sh" ]]; then
      echo "$resolved"
      return 0
    fi
  done

  return 1
}

sync_isaaclab_isaacsim_link() {
  local resolved_root
  local current_link

  if [[ -x "$ISAACSIM_SH" ]]; then
    ISAACSIM_ROOT="${ISAACSIM_ROOT:-$(dirname "$ISAACSIM_SH")}"
    return 0
  fi

  resolved_root="$(resolve_isaacsim_root || true)"
  [[ -n "$resolved_root" ]] || return 1

  mkdir -p "$ISAACLAB_DIR"

  if [[ -e "$ISAACLAB_DIR/_isaac_sim" && ! -L "$ISAACLAB_DIR/_isaac_sim" ]]; then
    if [[ -x "$ISAACLAB_DIR/_isaac_sim/python.sh" ]]; then
      ISAACSIM_ROOT="$ISAACLAB_DIR/_isaac_sim"
      ISAACSIM_SH="$ISAACLAB_DIR/_isaac_sim/python.sh"
      return 0
    fi

    warn "$ISAACLAB_DIR/_isaac_sim exists as a real directory without python.sh; leaving it untouched."
    return 1
  fi

  current_link="$(readlink -f "$ISAACLAB_DIR/_isaac_sim" 2>/dev/null || true)"
  if [[ "$current_link" != "$resolved_root" ]]; then
    log "Linking Isaac Lab checkout to Isaac Sim at $resolved_root"
    ln -sfn "$resolved_root" "$ISAACLAB_DIR/_isaac_sim"
  fi

  ISAACSIM_ROOT="$resolved_root"
  ISAACSIM_SH="$resolved_root/python.sh"
  [[ -x "$ISAACSIM_SH" ]]
}

require_isaacsim_ready() {
  sync_isaaclab_isaacsim_link || true

  if [[ -x "$ISAACSIM_SH" ]]; then
    ISAACSIM_READY=1
    return 0
  fi

  ISAACSIM_READY=0
  warn "Expected Isaac Sim python shim at $ISAACSIM_SH"

  if [[ -n "$ISAACSIM_ROOT" ]]; then
    if [[ -d "$ISAACSIM_ROOT" ]]; then
      warn "ISAACSIM_ROOT is set to $ISAACSIM_ROOT"
      if [[ ! -x "$ISAACSIM_ROOT/python.sh" ]]; then
        warn "ISAACSIM_ROOT exists but does not contain python.sh yet."
      fi
    else
      warn "ISAACSIM_ROOT is set to $ISAACSIM_ROOT but that directory does not exist."
    fi
  fi

  if [[ -L "$ISAACLAB_DIR/_isaac_sim" || -e "$ISAACLAB_DIR/_isaac_sim" ]]; then
    warn "Isaac Lab _isaac_sim currently resolves to: $(readlink -f "$ISAACLAB_DIR/_isaac_sim" 2>/dev/null || echo "<broken>")"
  fi

  if [[ "$REQUIRE_ISAACSIM" == "0" ]]; then
    warn "Continuing without Isaac Sim because REQUIRE_ISAACSIM=0"
    return 0
  fi

  cat >&2 <<EOF
[setup][ERROR] Isaac Sim is not installed or not linked correctly.
Expected an executable python shim at:
  $ISAACSIM_SH

Fix it with:
  export ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"
  mkdir -p "\$ISAACSIM_ROOT"
  unzip -o "\$HOME/isaacsim-downloads/isaac-sim-standalone-5.0.0-linux-x86_64.zip" -d "\$ISAACSIM_ROOT"
  (cd "\$ISAACSIM_ROOT" && ./post_install.sh)
  test -x "\$ISAACSIM_ROOT/python.sh"
  ln -sfn "\$ISAACSIM_ROOT" "$ISAACLAB_DIR/_isaac_sim"

Then rerun ./setup.sh.
Set REQUIRE_ISAACSIM=0 only if you intentionally want to skip Isaac Sim-dependent setup.
EOF
  exit 1
}

clone_or_update_repo() {
  local repo_url="$1"
  local repo_dir="$2"
  local branch="$3"

  local auth_url
  auth_url="$(git_auth_url "$repo_url")"

  if [[ ! -d "$repo_dir/.git" ]]; then
    if [[ -d "$repo_dir" ]] && find "$repo_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      warn "Repo directory exists without git metadata at $repo_dir; using existing contents as-is."
    else
      log "Cloning $(basename "$repo_dir") into $repo_dir"
      git clone --branch "$branch" --single-branch "$auth_url" "$repo_dir"
    fi
  else
    log "Repo already exists at $repo_dir"
    if [[ "$UPDATE_EXISTING_REPOS" != "0" ]]; then
      pushd "$repo_dir" >/dev/null
      git remote set-url origin "$auth_url" || true
      git fetch origin
      if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
        git checkout "$branch"
        git pull --ff-only origin "$branch"
      else
        warn "Branch '$branch' not found on origin for $repo_dir; leaving current branch unchanged."
      fi
      popd >/dev/null
    else
      log "Skipping repo refresh for $repo_dir (set UPDATE_EXISTING_REPOS=1 to pull latest changes)"
    fi
  fi
}

prepare_codesign_repo() {
  if [[ -d "$CO_DESIGN_DIR/.git" ]]; then
    log "Using existing CoDesign repo at $CO_DESIGN_DIR"
  else
    clone_or_update_repo "$REPO_URL" "$CO_DESIGN_DIR" "$REPO_BRANCH"
  fi
}

resolve_python_candidate() {
  local candidate="$1"

  if [[ -z "$candidate" ]]; then
    return 1
  fi

  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] || return 1
    readlink -f "$candidate"
  else
    command -v "$candidate"
  fi
}

python_has_module() {
  local py="$1"
  local module="$2"

  "$py" - "$module" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
}

python_module_origin() {
  local py="$1"
  local module="$2"

  "$py" - "$module" <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec(sys.argv[1])
if spec is None:
    raise SystemExit(1)

candidates = []
if spec.origin not in (None, "built-in", "frozen"):
    candidates.append(spec.origin)
if spec.submodule_search_locations:
    candidates.extend(spec.submodule_search_locations)

for candidate in candidates:
    try:
        print(pathlib.Path(candidate).resolve())
        raise SystemExit(0)
    except Exception:
        continue

raise SystemExit(1)
PY
}

python_module_in_prefix() {
  local py="$1"
  local module="$2"
  local prefix="$3"
  local prefix_real
  local origin

  prefix_real="$(readlink -f "$prefix")"
  origin="$(python_module_origin "$py" "$module" 2>/dev/null || true)"
  [[ -n "$origin" && "$origin" == "$prefix_real"* ]]
}

python_supports_venv() {
  local py="$1"
  python_has_module "$py" venv
}

resolve_venv_base_python() {
  local -a raw_candidates=()
  local candidate
  local resolved

  [[ -n "$VENV_BASE_PYTHON" ]] && raw_candidates+=("$VENV_BASE_PYTHON")
  [[ -x "$CONDA_PY" ]] && raw_candidates+=("$CONDA_PY")
  raw_candidates+=("python3" "python")

  declare -A seen=()
  for candidate in "${raw_candidates[@]}"; do
    resolved="$(resolve_python_candidate "$candidate" 2>/dev/null || true)"
    [[ -n "$resolved" ]] || continue
    [[ -n "${seen[$resolved]:-}" ]] && continue
    seen["$resolved"]=1
    if python_supports_venv "$resolved"; then
      echo "$resolved"
      return 0
    fi
  done

  return 1
}

create_runner_venv() {
  local base_py
  local -a venv_args=()

  base_py="$(resolve_venv_base_python)" || die "Could not find a Python interpreter with the stdlib 'venv' module."

  mkdir -p "$(dirname "$VENV_DIR")"

  if [[ -d "$VENV_DIR" && "$FORCE_RECREATE_VENV" != "0" ]]; then
    log "Recreating virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    [[ "$VENV_SYSTEM_SITE_PACKAGES" == "0" ]] || venv_args+=(--system-site-packages)
    log "Creating runner virtual environment at $VENV_DIR using $base_py"
    "$base_py" -m venv "${venv_args[@]}" "$VENV_DIR"
  else
    log "Using existing runner virtual environment at $VENV_DIR"
  fi

  VENV_PY="$VENV_DIR/bin/python"
  VENV_BIN="$VENV_DIR/bin"

  [[ -x "$VENV_PY" ]] || die "Virtual environment python not found after creation: $VENV_PY"
}

install_runner_venv_deps() {
  local -a packages_to_install=()
  local -a venv_managed_specs=(
    "numpy:numpy"
    "scipy:scipy"
    "trimesh:trimesh"
    "coacd:coacd"
    "wandb:wandb"
    "xacrodoc:xacrodoc"
  )
  local spec
  local module_name
  local package_name

  log "Installing runner/build dependencies into virtual environment"
  "$VENV_PY" -m pip install --upgrade pip setuptools wheel

  for spec in "${venv_managed_specs[@]}"; do
    module_name="${spec%%:*}"
    package_name="${spec#*:}"
    if python_module_in_prefix "$VENV_PY" "$module_name" "$VENV_DIR"; then
      log "Keeping existing venv install for $module_name"
    else
      packages_to_install+=("$package_name")
    fi
  done

  if ((${#packages_to_install[@]} > 0)); then
    # Force installation into the venv even if the base interpreter already
    # provides a copy through --system-site-packages.
    "$VENV_PY" -m pip install --upgrade --ignore-installed "${packages_to_install[@]}"
  else
    log "Venv-managed runner/build packages are already present"
  fi

  if python_module_in_prefix "$VENV_PY" torch "$VENV_DIR"; then
    log "torch is already installed inside the virtual environment"
  elif python_has_module "$VENV_PY" torch; then
    log "torch is available to the virtual environment via inherited site-packages"
  else
    log "Installing torch into the virtual environment"
    "$VENV_PY" -m pip install "$TORCH_PIP_SPEC"
  fi

  if [[ -d "$CO_DESIGN_ISAACLAB_DIR/source/codesign" ]]; then
    log "Installing codesign editable package into runner virtual environment"
    "$VENV_PY" -m pip install -e "$CO_DESIGN_ISAACLAB_DIR/source/codesign"
  else
    warn "codesign source not found at $CO_DESIGN_ISAACLAB_DIR/source/codesign; skipping venv editable install."
  fi
}

upsert_managed_block() {
  local file="$1"
  local start_marker="$2"
  local end_marker="$3"
  local payload="$4"
  local tmp

  tmp="$(mktemp)"

  if [[ -f "$file" ]]; then
    awk -v start="$start_marker" -v end="$end_marker" '
      $0 == start {skip=1; next}
      $0 == end {skip=0; next}
      !skip {print}
    ' "$file" > "$tmp"
  else
    : > "$tmp"
  fi

  {
    cat "$tmp"
    if [[ -s "$tmp" ]]; then
      printf '\n'
    fi
    printf '%s\n' "$start_marker"
    printf '%s\n' "$payload"
    printf '%s\n' "$end_marker"
  } > "$file"

  rm -f "$tmp"
}

write_codesign_env_file() {
  log "Writing shell environment snippet to $CODESIGN_ENV_FILE"
  mkdir -p "$(dirname "$CODESIGN_ENV_FILE")"

  cat > "$CODESIGN_ENV_FILE" <<EOF
# Generated by setup.sh. Source this file to use the CoDesign runner environment.
export CODESIGN_VENV_DIR="$VENV_DIR"
export CODESIGN_RUNNER_PYTHON="$VENV_PY"
export CODESIGN_PYTHON="$VENV_PY"

# IsaacLab bootstrap files often alias python/pip to Isaac Sim. Clear those so
# the CoDesign runner venv wins in interactive shells.
unalias python python3 pip pip3 2>/dev/null || true

if [[ -f "$VENV_BIN/activate" ]]; then
  # Use the standard activation script so shells also show the active venv.
  source "$VENV_BIN/activate"
else
  export VIRTUAL_ENV="$VENV_DIR"
  case ":\$PATH:" in
    *:"$VENV_BIN":*) ;;
    *) export PATH="$VENV_BIN:\$PATH" ;;
  esac
fi

export CODESIGN_RUNNER_PYTHON="$VENV_PY"
export CODESIGN_PYTHON="$VENV_PY"
EOF

  if [[ -x "$ISAACSIM_SH" ]]; then
    printf 'export CODESIGN_ISAAC_PYTHON="%s"\n' "$ISAACSIM_SH" >> "$CODESIGN_ENV_FILE"
  fi

  if [[ -x "$VENV_BIN/xacrodoc" ]]; then
    printf 'export CODESIGN_XACRODOC="%s"\n' "$VENV_BIN/xacrodoc" >> "$CODESIGN_ENV_FILE"
  elif [[ -x "$CONDA_BIN/xacrodoc" ]]; then
    printf 'export CODESIGN_XACRODOC="%s"\n' "$CONDA_BIN/xacrodoc" >> "$CODESIGN_ENV_FILE"
  fi

  chmod 0644 "$CODESIGN_ENV_FILE"
}

persist_shell_setup() {
  local bashrc
  local payload

  [[ "$PERSIST_SHELL_CONFIG" == "0" ]] && return 0

  bashrc="$HOME/.bashrc"
  touch "$bashrc"

  payload="$(cat <<EOF
if [[ -f "$CODESIGN_ENV_FILE" ]]; then
  source "$CODESIGN_ENV_FILE"
fi
EOF
)"

  upsert_managed_block \
    "$bashrc" \
    "# >>> codesign setup >>>" \
    "# <<< codesign setup <<<" \
    "$payload"

  log "Updated $bashrc so new shells use the CoDesign runner virtual environment"
}

verify_python_modules() {
  local py="$1"
  shift
  local module
  local -a missing=()

  for module in "$@"; do
    if ! python_has_module "$py" "$module"; then
      missing+=("$module")
    fi
  done

  if ((${#missing[@]} > 0)); then
    warn "Missing modules for $py: ${missing[*]}"
  fi
}

require_python_modules() {
  local py="$1"
  shift
  local module
  local -a missing=()

  for module in "$@"; do
    if ! python_has_module "$py" "$module"; then
      missing+=("$module")
    fi
  done

  if ((${#missing[@]} > 0)); then
    die "Missing required modules for $py: ${missing[*]}"
  fi
}

remove_path_entry() {
  local path_value="$1"
  local entry_to_remove="$2"
  local cleaned=""
  local entry
  local entry_real
  local target_real
  local first=1

  [[ -n "$path_value" ]] || return 0
  [[ -n "$entry_to_remove" ]] || {
    printf '%s' "$path_value"
    return 0
  }

  target_real="$(readlink -f "$entry_to_remove" 2>/dev/null || printf '%s' "$entry_to_remove")"

  IFS=':' read -r -a _setup_path_entries <<< "$path_value"
  for entry in "${_setup_path_entries[@]}"; do
    [[ -n "$entry" ]] || continue
    entry_real="$(readlink -f "$entry" 2>/dev/null || printf '%s' "$entry")"
    [[ "$entry_real" == "$target_real" ]] && continue

    if [[ "$first" == "1" ]]; then
      cleaned="$entry"
      first=0
    else
      cleaned="${cleaned}:$entry"
    fi
  done

  printf '%s' "$cleaned"
}

build_setup_safe_path() {
  local path_value="${1:-$PATH}"
  local cleaned="$path_value"

  [[ -n "${VIRTUAL_ENV:-}" ]] && cleaned="$(remove_path_entry "$cleaned" "$VIRTUAL_ENV/bin")"
  [[ -n "${CONDA_PREFIX:-}" ]] && cleaned="$(remove_path_entry "$cleaned" "$CONDA_PREFIX/bin")"
  [[ -n "$VENV_DIR" ]] && cleaned="$(remove_path_entry "$cleaned" "$VENV_DIR/bin")"
  [[ -n "$CONDA_BIN" ]] && cleaned="$(remove_path_entry "$cleaned" "$CONDA_BIN")"

  printf '%s' "$cleaned"
}

sanitize_inherited_python_env() {
  local cleaned_path
  local touched=0
  local var_name

  cleaned_path="$(build_setup_safe_path "$PATH")"
  if [[ "$cleaned_path" != "$PATH" ]]; then
    PATH="$cleaned_path"
    export PATH
    touched=1
  fi

  for var_name in VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV _CE_CONDA _CE_M MAMBA_DEFAULT_ENV MAMBA_ROOT_PREFIX PYTHONHOME PYTHONPATH; do
    if [[ -n "${!var_name:-}" ]]; then
      unset "$var_name"
      touched=1
    fi
  done

  if [[ "$touched" == "1" ]]; then
    hash -r
    log "Sanitized inherited Python environment so setup uses Isaac Sim/system Python instead of an active shell venv"
  fi
}

run_isaaclab_rl_games_install() {
  local clean_path

  clean_path="$(build_setup_safe_path "$PATH")"

  env \
    PATH="$clean_path" \
    HOME="$HOME" \
    USER="${USER:-root}" \
    LOGNAME="${LOGNAME:-root}" \
    SHELL="${SHELL:-/bin/bash}" \
    TERM="${TERM:-xterm}" \
    DISPLAY="${DISPLAY:-}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    VIRTUAL_ENV= \
    CONDA_PREFIX= \
    CONDA_DEFAULT_ENV= \
    _CE_CONDA= \
    _CE_M= \
    MAMBA_DEFAULT_ENV= \
    MAMBA_ROOT_PREFIX= \
    PYTHONHOME= \
    PYTHONPATH= \
    "$ISAACLAB_DIR/isaaclab.sh" -i rl_games
}

# ----------------------------
# prepare shell/workspace
# ----------------------------
mkdir -p "$WORKDIR"

if [[ -f ~/.bashrc ]]; then
  set +u
  # shellcheck disable=SC1090
  source ~/.bashrc || true
  set -u
fi

sanitize_inherited_python_env

require_cmd git
require_cmd apt-get

# ----------------------------
# apt packages
# ----------------------------
install_apt_packages_if_missing \
  python3 \
  python3-venv \
  libgl1 \
  libegl1 \
  libvulkan1 \
  libglu1-mesa \
  vulkan-tools \
  mesa-utils \
  xserver-xorg-video-dummy \
  x11vnc \
  net-tools \
  xvfb \
  zenity \
  dialog \
  git-lfs

git lfs install

# ----------------------------
# clone/update repos
# ----------------------------
prepare_codesign_repo
clone_or_update_repo "$ISAACLAB_REPO_URL" "$ISAACLAB_DIR" "$ISAACLAB_BRANCH"

# Keep origin PAT-authenticated for CoDesign if requested
if [[ -d "$CO_DESIGN_DIR/.git" ]]; then
  pushd "$CO_DESIGN_DIR" >/dev/null
  git remote set-url origin "$(git_auth_url "$REPO_URL")" || true
  popd >/dev/null
fi

require_isaacsim_ready

# ----------------------------
# Isaac Lab python packages
# ----------------------------
if [[ "$ISAACSIM_READY" == "1" && -x "$ISAACLAB_DIR/isaaclab.sh" ]]; then
  if [[ "$SKIP_HEAVY_ON_RERUN" == "1" && "$FORCE_REINSTALL_ISAACLAB_PACKAGES" == "0" ]] && [[ -f "$ISAACSIM_SH" ]] && have_stamp isaaclab_rl_games_install; then
    log "Skipping Isaac Lab package reinstall on rerun (set FORCE_REINSTALL_ISAACLAB_PACKAGES=1 to reinstall)"
  else
    ensure_no_active_isaac_processes
    log "Installing Isaac Lab packages into Isaac Sim python"
    run_isaaclab_rl_games_install
    write_stamp isaaclab_rl_games_install
  fi
elif [[ "$ISAACSIM_READY" != "1" ]]; then
  warn "Isaac Sim shim not found at $ISAACSIM_SH; skipping Isaac Lab package install."
else
  warn "Isaac Lab helper not found at $ISAACLAB_DIR/isaaclab.sh; skipping Isaac Lab package install."
fi

# ----------------------------
# Runner/build virtual environment
# ----------------------------
create_runner_venv
install_runner_venv_deps
write_codesign_env_file
persist_shell_setup

set +u
# shellcheck disable=SC1090
source "$CODESIGN_ENV_FILE"
set -u

# ----------------------------
# Optional legacy host python install
# ----------------------------
if [[ "$INSTALL_HOST_PYTHON_DEPS" != "0" ]]; then
  if command -v python >/dev/null 2>&1; then
    if [[ -d "$CO_DESIGN_ISAACLAB_DIR/source/codesign" ]]; then
      log "Installing codesign editable package into host python"
      python -m pip install --upgrade pip
      python -m pip install -e "$CO_DESIGN_ISAACLAB_DIR/source/codesign"
    else
      warn "codesign source not found at $CO_DESIGN_ISAACLAB_DIR/source/codesign; skipping host editable install."
    fi
  else
    warn "'python' not found on PATH; skipping host python install."
  fi
fi

# ----------------------------
# Isaac Sim python deps
# ----------------------------
if [[ "$ISAACSIM_READY" == "1" && -x "$ISAACSIM_SH" ]]; then
  if [[ "$SKIP_HEAVY_ON_RERUN" == "1" && "$FORCE_REINSTALL_ISAACSIM_PYTHON_DEPS" == "0" ]] && have_stamp isaacsim_python_deps; then
    log "Skipping Isaac Sim python reinstall on rerun (set FORCE_REINSTALL_ISAACSIM_PYTHON_DEPS=1 to reinstall)"
  else
    ensure_no_active_isaac_processes
    log "Installing deps into Isaac Sim python via $ISAACSIM_SH"
    "$ISAACSIM_SH" -m pip install --upgrade pip
    "$ISAACSIM_SH" -m pip install xacrodoc coacd scipy trimesh wandb

    if [[ -d "$CO_DESIGN_ISAACLAB_DIR/source/codesign" ]]; then
      log "Installing codesign editable package into Isaac Sim python"
      "$ISAACSIM_SH" -m pip install -e "$CO_DESIGN_ISAACLAB_DIR/source/codesign"
    else
      warn "codesign source not found at $CO_DESIGN_ISAACLAB_DIR/source/codesign; skipping Isaac Sim editable install."
    fi
    write_stamp isaacsim_python_deps
  fi
else
  warn "Isaac Sim shim not found at $ISAACSIM_SH; skipping Isaac Sim pip installs."
fi

# ----------------------------
# Optional legacy conda env deps
# ----------------------------
if [[ "$INSTALL_CONDA_DEPS" != "0" ]]; then
  if [[ -x "$CONDA_PY" ]]; then
    log "Installing deps into legacy conda env $ENV_NAME"
    "$CONDA_PY" -m pip install --upgrade pip
    "$CONDA_PY" -m pip install torch numpy xacrodoc scipy trimesh coacd wandb

    if [[ -d "$CO_DESIGN_ISAACLAB_DIR/source/codesign" ]]; then
      log "Installing codesign editable package into legacy conda env $ENV_NAME"
      "$CONDA_PY" -m pip install -e "$CO_DESIGN_ISAACLAB_DIR/source/codesign"
    else
      warn "codesign source not found at $CO_DESIGN_ISAACLAB_DIR/source/codesign; skipping conda editable install."
    fi
  else
    warn "Conda env python not found at $CONDA_PY; skipping conda installs."
  fi
fi

# ----------------------------
# verification
# ----------------------------
verify_python_modules "$VENV_PY" torch numpy scipy trimesh coacd wandb xacrodoc codesign
require_python_modules "$VENV_PY" torch numpy scipy trimesh coacd wandb xacrodoc codesign

log "Verification summary:"
echo "  CoDesign repo:   $CO_DESIGN_DIR"
echo "  IsaacLab repo:   $ISAACLAB_DIR"
echo "  Runner venv:     $VENV_DIR"
echo "  Runner python:   $VENV_PY"
echo "  Isaac Sim shim:  $ISAACSIM_SH"
echo "  Conda python:    $CONDA_PY"
echo "  Env snippet:     $CODESIGN_ENV_FILE"

if python_module_in_prefix "$VENV_PY" torch "$VENV_DIR"; then
  echo "  torch source:    venv"
elif python_has_module "$VENV_PY" torch; then
echo "  torch source:    inherited from base interpreter"
else
  warn "torch is not importable from the runner virtual environment."
fi

if [[ -x "$VENV_BIN/xacrodoc" ]]; then
  echo "  xacrodoc path:   $VENV_BIN/xacrodoc"
elif command -v xacrodoc >/dev/null 2>&1; then
  echo "  xacrodoc path:   $(command -v xacrodoc)"
else
  warn "xacrodoc is still not on PATH."
fi

write_stamp setup_complete

log "Setup complete"
