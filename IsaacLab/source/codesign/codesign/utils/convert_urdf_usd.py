# scripts/tools/batch_convert_urdf.py
"""
# Run example:
python source/codesign/codesign/utils/convert_urdf_usd.py \
  --in-dir source/codesign/codesign/assets/group4_test \
  --out-dir source/codesign/codesign/assets/group4_test \
  --merge-joints --headless
  
python scripts/zero_agent.py --task=Codesign-Reorientation_Direct-v0 --num_envs=16

python scripts/tools/batch_convert_urdf.py \
  --in-dir source/codesign/codesign/assets/test \
  --out-dir source/codesign/codesign/assets/test \
  --pattern "*.urdf" \
  --nat-freq 25.0 --zeta 1.0 \
  --merge-joints --headless --no-window
"""

import argparse
import sys
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Batch URDF→USD with Natural gains / force drive / static base")
parser.add_argument("--in-dir", required=True, help="Folder to search for URDFs (recurses)")
parser.add_argument("--out-dir", required=True, help="Base folder to write per-model subfolders")
parser.add_argument("--pattern", default="*.urdf", help="Glob pattern (recurses)")
parser.add_argument("--merge-joints", action="store_true", default=False)
parser.add_argument("--nat-freq", type=float, default=5.0, help="Natural frequency (Hz)")
parser.add_argument("--zeta", type=float, default=1.0, help="Damping ratio")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app  # one Kit session

def _enable_urdf_importer_extension():
    """Enable whichever URDF importer extension is available in this Isaac Sim install."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    candidates = (
        "isaacsim.asset.importer.urdf",
        "isaacsim.asset.importer.urdf-2.4.31",
        "isaacsim.asset.importer.urdf-2.4.19",
        "omni.importer.urdf",
    )

    enabled = []
    for ext_name in candidates:
        try:
            manager.set_extension_enabled_immediate(ext_name, True)
        except Exception:
            continue
        try:
            if manager.is_extension_enabled(ext_name):
                enabled.append(ext_name)
        except Exception:
            continue

    if enabled:
        print(f"Enabled URDF importer extension(s): {', '.join(enabled)}")
    else:
        print("Warning: could not explicitly enable a URDF importer extension; relying on current Kit config.")


exit_code = 0

try:
    _enable_urdf_importer_extension()

    # Import after Kit is up and the importer extension is enabled.
    import omni.kit.commands
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase

    class CompatibleUrdfConverter(UrdfConverter):
        """Tolerate importer API differences across Isaac Sim patch versions."""

        def __init__(self, cfg):
            from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface

            self._urdf_interface = acquire_urdf_interface()
            AssetConverterBase.__init__(self, cfg=cfg)

        def _get_urdf_import_config(self):
            _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

            import_config.set_distance_scale(1.0)
            import_config.set_make_default_prim(True)
            import_config.set_create_physics_scene(False)

            convex_decomp = self.cfg.collider_type == "convex_decomposition"
            import_config.set_density(self.cfg.link_density)
            import_config.set_convex_decomp(convex_decomp)
            import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
            import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)

            # Older importer builds do not expose this helper yet.
            if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
                import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)

            import_config.set_fix_base(self.cfg.fix_base)
            import_config.set_self_collision(self.cfg.self_collision)
            import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
            import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)

            return import_config

    in_root = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    urdfs = sorted(in_root.rglob(args.pattern))
    if not urdfs:
        print(f"No URDFs under {in_root} matching {args.pattern}")
    else:
        print(f"Found {len(urdfs)} URDF(s). Writing per-model subfolders under {out_root} ...")

    success_count = 0
    failure_count = 0

    for i, urdf in enumerate(urdfs, 1):
        model_name = urdf.stem
        model_dir = out_root / model_name            # e.g., .../rounded_hand/
        model_dir.mkdir(parents=True, exist_ok=True)
        usd_filename = model_name + ".usd"           # e.g., rounded_hand.usd
        expected_usd = model_dir / usd_filename

        cfg = UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(model_dir),                  # write into per-model folder
            usd_file_name=usd_filename,
            fix_base=True,                           # Static Base
            merge_fixed_joints=args.merge_joints,
            force_usd_conversion=True,
            link_density=1240,                      # roughly PLA plastic
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",                 # Drive type: Force
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
                    natural_frequency=args.nat_freq,  # Joint config: Natural
                    damping_ratio=args.zeta,
                ),
            ),
        )

        print(f"[{i}/{len(urdfs)}] {urdf} -> {expected_usd}")
        try:
            CompatibleUrdfConverter(cfg)
        except Exception:
            failure_count += 1
            print(f"[{i}/{len(urdfs)}] FAILED to convert {urdf}")
            traceback.print_exc()
            continue

        if expected_usd.exists():
            success_count += 1
        else:
            failure_count += 1
            print(f"[{i}/{len(urdfs)}] FAILED: expected USD was not created at {expected_usd}")

    print(f"Conversion summary: {success_count} succeeded, {failure_count} failed")
    if failure_count:
        exit_code = 1
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    app.close()

raise SystemExit(exit_code)
