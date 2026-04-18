#!/usr/bin/env python3
"""
Verify required Python dependencies for the MuJoCo hand generator pipeline.

Checks:
1) Imports for numpy, scipy, trimesh, coacd
2) A minimal CoACD decomposition run using the same API path used by generate_palm_mesh.py
"""

from __future__ import annotations

import sys


def _version(mod) -> str:
    return getattr(mod, "__version__", "unknown")


def main() -> int:
    try:
        import numpy as np
        import scipy
        import trimesh
        import coacd
    except Exception as exc:
        print(f"[FAIL] Import error: {exc}")
        return 1

    print("[OK] Imported required modules:")
    print(f"  numpy   : {_version(np)}")
    print(f"  scipy   : {_version(scipy)}")
    print(f"  trimesh : {_version(trimesh)}")
    print(f"  coacd   : {_version(coacd)}")

    # Use a tiny watertight primitive and run CoACD with the same call style as generate_palm_mesh.py.
    try:
        mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        parts = coacd.run_coacd(
            coacd.Mesh(vertices, faces),
            threshold=0.1,
            max_convex_hull=-1,
            preprocess_mode="auto",
            preprocess_resolution=50,
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=100,
            mcts_max_depth=3,
            pca=False,
            merge=True,
            decimate=False,
            max_ch_vertex=256,
            extrude=False,
            extrude_margin=0.01,
            apx_mode="ch",
            seed=36,
        )
    except Exception as exc:
        print(f"[FAIL] CoACD runtime check failed: {exc}")
        return 1

    if not parts:
        print("[FAIL] CoACD returned no convex parts.")
        return 1

    first_vertices, first_faces = parts[0]
    first_vertices = np.asarray(first_vertices)
    first_faces = np.asarray(first_faces)
    if first_vertices.ndim != 2 or first_vertices.shape[1] != 3:
        print("[FAIL] First convex part has invalid vertex shape.")
        return 1
    if first_faces.ndim != 2 or first_faces.shape[1] != 3:
        print("[FAIL] First convex part has invalid face shape.")
        return 1

    print("[OK] CoACD runtime check passed.")
    print(f"  convex parts produced: {len(parts)}")
    print(f"  first part verts/faces: {first_vertices.shape[0]} / {first_faces.shape[0]}")
    print("[PASS] MuJoCo generator dependency verification succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
