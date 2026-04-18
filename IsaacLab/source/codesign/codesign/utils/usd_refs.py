from __future__ import annotations

import os

import omni.usd
from pxr import Sdf


def get_current_stage():
    """Return the active USD stage from the Omniverse context."""
    return omni.usd.get_context().get_stage()


def normalize_asset_path(path: str) -> str:
    """Normalize a filesystem path when possible."""
    try:
        return os.path.realpath(path)
    except Exception:
        return path


def _get_reference_items(list_op: Sdf.ReferenceListOp):
    """Handle USD API differences across versions."""
    try:
        return list_op.GetAddedOrExplicitItems()
    except AttributeError:
        return (
            getattr(list_op, "addedOrExplicitItems", [])
            or getattr(list_op, "GetExplicitItems", lambda: [])()
            or getattr(list_op, "GetAddedItems", lambda: [])()
        )


def get_referenced_usd_from_prim(
    prim,
    *,
    allowed_paths: set[str] | None = None,
    normalize_paths: bool = False,
) -> str | None:
    """Resolve the first referenced USD for a prim."""
    if not prim:
        return None

    list_op = prim.GetMetadata("references")
    if isinstance(list_op, Sdf.ReferenceListOp):
        for ref in _get_reference_items(list_op):
            asset_path = getattr(ref, "assetPath", "")
            if not asset_path:
                continue
            candidate = normalize_asset_path(asset_path) if normalize_paths else asset_path
            if allowed_paths is None or candidate in allowed_paths:
                return candidate

    for spec in prim.GetPrimStack():
        layer = spec.layer
        path = getattr(layer, "realPath", "") or layer.identifier
        if not path or not path.lower().endswith(".usd"):
            continue
        candidate = normalize_asset_path(path) if normalize_paths else path
        if allowed_paths is None or candidate in allowed_paths:
            return candidate
    return None


def get_env_robot_usd_path(stage, env_index: int) -> str | None:
    """Resolve the robot USD path for one environment."""
    prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot")
    return get_referenced_usd_from_prim(prim)


def get_env_object_usd_path(
    stage,
    env_index: int,
    *,
    allowed_paths: set[str] | None = None,
    normalize_paths: bool = False,
) -> str | None:
    """Resolve the object USD path for one environment."""
    for prim_path in (
        f"/World/envs/env_{env_index}/object/object",
        f"/World/envs/env_{env_index}/object",
    ):
        prim = stage.GetPrimAtPath(prim_path)
        path = get_referenced_usd_from_prim(
            prim,
            allowed_paths=allowed_paths,
            normalize_paths=normalize_paths,
        )
        if path:
            return path
    return None
