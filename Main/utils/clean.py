"""Helpers for launching and cleaning up subprocess groups."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Optional


def _terminate_process_group(proc: subprocess.Popen, label: str, grace_seconds: float = 5.0) -> None:
    """Terminate a subprocess group so Isaac/Kit children do not linger after interrupts/timeouts."""
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as exc:
        print(f"[cleanup] Failed to send SIGTERM to {label}: {exc}")
    else:
        try:
            proc.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as exc:
        print(f"[cleanup] Failed to send SIGKILL to {label}: {exc}")
        return

    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        print(f"[cleanup] {label} did not exit after SIGKILL")


def run_managed_subprocess(
    cmd,
    *,
    timeout: Optional[int] = None,
    env=None,
    cwd: Optional[str] = None,
    text: bool = True,
    capture_output: bool = False,
    stdout=None,
    stderr=None,
    check: bool = False,
    label: Optional[str] = None,
):
    """Run a subprocess in its own process group so interrupts and timeouts clean up descendants."""
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("capture_output cannot be used with explicit stdout/stderr")
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    run_label = label or (" ".join(cmd[:2]) if isinstance(cmd, (list, tuple)) else str(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
    )

    try:
        stdout_data, stderr_data = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        print(f"[cleanup] {run_label} timed out after {timeout}s; terminating subprocess group")
        _terminate_process_group(proc, run_label)
        try:
            stdout_data, stderr_data = proc.communicate(timeout=1)
        except Exception:
            stdout_data = exc.output
            stderr_data = exc.stderr
        raise subprocess.TimeoutExpired(proc.args, timeout, output=stdout_data, stderr=stderr_data)
    except KeyboardInterrupt:
        print(f"[cleanup] Interrupted while running {run_label}; terminating subprocess group")
        _terminate_process_group(proc, run_label)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        raise
    except BaseException:
        _terminate_process_group(proc, run_label)
        try:
            proc.communicate(timeout=1)
        except Exception:
            pass
        raise

    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=stdout_data, stderr=stderr_data)

    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout_data, stderr_data)
