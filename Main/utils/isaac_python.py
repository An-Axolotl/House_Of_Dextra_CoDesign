#!/usr/bin/env python3
"""
Helpers for resolving the Python executable used by Isaac Lab subprocesses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _resolve_python_executable(candidate: str) -> Optional[str]:
    """Resolve a Python executable path or command name into an executable path."""
    if not candidate:
        return None

    expanded = os.path.expandvars(os.path.expanduser(candidate))

    # Treat values with path separators as explicit file paths.
    if os.path.sep in expanded:
        path = Path(expanded)
        if path.exists():
            return str(path.resolve())
        return None

    return shutil.which(expanded)


def _python_supports_isaaclab(python_exe: str, env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """
    Validate that a Python executable can resolve Isaac Lab modules.
    We probe module specs instead of importing full stacks to keep startup light.
    """
    probe = (
        "import importlib.util,sys;"
        "mods=('isaaclab','isaaclab_tasks');"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "sys.exit(0 if not missing else 1)"
    )

    try:
        result = subprocess.run(
            [python_exe, "-c", probe],
            capture_output=True,
            text=True,
            timeout=45,
            env=env,
        )
    except Exception as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, "ok"

    tail = (result.stderr or result.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"exit code {result.returncode}"
    return False, detail


def resolve_isaac_python(
    play_script: str,
    env: Optional[Dict[str, str]] = None,
    preferred: Optional[str] = None,
) -> str:
    """
    Resolve the Python executable used for Isaac Lab/Isaac Sim subprocesses.
    Priority:
      1) --isaac-python
      2) CODESIGN_ISAAC_PYTHON / ISAAC_SIM_PYTHON
      3) <IsaacLab>/_isaac_sim/python.sh inferred from --play-script
      4) common Isaac Sim paths
      5) python3/python if Isaac Lab modules are importable
    """
    merged_env: Dict[str, str] = dict(os.environ)
    if env:
        merged_env.update(env)

    candidates: List[Tuple[str, str]] = []

    if preferred:
        candidates.append(("--isaac-python", preferred))

    for var in ("CODESIGN_ISAAC_PYTHON", "ISAAC_SIM_PYTHON"):
        value = merged_env.get(var)
        if value:
            candidates.append((f"env:{var}", value))

    # Infer IsaacLab root from play script path: <IsaacLab>/scripts/rl_games/play.py
    try:
        play_path = Path(play_script).resolve()
        if len(play_path.parents) >= 3:
            isaaclab_root = play_path.parents[2]
            candidates.append(("inferred-from-play-script", str(isaaclab_root / "_isaac_sim/python.sh")))
    except Exception:
        pass

    candidates.extend(
        [
            ("common", "/workspace/isaaclab/_isaac_sim/python.sh"),
            ("common", "/isaac-sim/python.sh"),
            ("common", "/opt/nvidia/isaac-sim/python.sh"),
            ("fallback", "python3"),
            ("fallback", "python"),
        ]
    )

    tried: List[Tuple[str, str, str]] = []
    seen_paths = set()

    for source, raw_candidate in candidates:
        resolved = _resolve_python_executable(raw_candidate)
        if not resolved or resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        ok, detail = _python_supports_isaaclab(resolved, env=merged_env)
        if ok:
            print(f"Using Isaac Python ({source}): {resolved}")
            return resolved

        tried.append((source, resolved, detail))

    tried_msg = "\n".join(f"  - {src}: {path} ({detail})" for src, path, detail in tried)
    raise RuntimeError(
        "Could not find a Python executable with Isaac Lab modules "
        "('isaaclab' and 'isaaclab_tasks').\n"
        f"Tried:\n{tried_msg if tried_msg else '  - no executable candidates found'}\n"
        "Set --isaac-python or CODESIGN_ISAAC_PYTHON to a valid interpreter."
    )

