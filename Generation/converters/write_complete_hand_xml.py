#!/usr/bin/env python3
"""Write a complete MuJoCo hand XML scene from generated frame data."""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_args():
    parser = argparse.ArgumentParser(description="Write complete hand MJCF scene XML.")
    parser.add_argument("--frames-json", required=True, help="Input palm frames JSON.")
    parser.add_argument("--out", required=True, help="Output complete XML path.")
    parser.add_argument("--hand-name", required=True, help="Generated hand name.")
    parser.add_argument(
        "--actual-finger-count",
        required=True,
        type=int,
        help="Number of active fingers for this hand.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.frames_json, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    mujoco = ET.Element("mujoco", {"model": f"{args.hand_name}_scene"})

    ET.SubElement(
        mujoco,
        "compiler",
        {"angle": "radian", "meshdir": "./", "inertiagrouprange": "1 1"},
    )

    option = ET.SubElement(
        mujoco,
        "option",
        {"integrator": "implicitfast", "cone": "elliptic", "impratio": "100", "timestep": "0.002"},
    )
    ET.SubElement(option, "flag", {"warmstart": "enable", "nativeccd": "enable", "multiccd": "enable"})

    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "mesh", {"name": "palm_vis", "file": f"meshes/{args.hand_name}_palm.obj"})

    palm_base_path = os.path.splitext(
        os.path.join(os.path.dirname(args.frames_json), f"{args.hand_name}_palm.obj")
    )[0]
    palm_part_files = sorted(glob.glob(f"{palm_base_path}_part*.obj"))

    if palm_part_files:
        for i, part_file in enumerate(palm_part_files):
            part_filename = os.path.basename(part_file)
            ET.SubElement(asset, "mesh", {"name": f"palm_part{i}", "file": f"meshes/{part_filename}"})
    else:
        ET.SubElement(asset, "mesh", {"name": "palm_part0", "file": f"meshes/{args.hand_name}_palm.obj"})

    for i in range(1, 6):
        ET.SubElement(asset, "model", {"name": f"finger_model_{i}", "file": f"{args.hand_name}_f{i}.xml"})

    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".1 .2 .3",
            "rgb2": ".2 .3 .4",
            "width": "300",
            "height": "300",
            "mark": "none",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "grid", "texture": "grid", "texrepeat": "1 1", "texuniform": "true", "reflectance": ".2"},
    )

    default = ET.SubElement(mujoco, "default")
    ET.SubElement(
        default,
        "geom",
        {"solimp": "0.999 0.999 0.001 0.0001 1", "solref": "0.0001 1", "friction": ".2", "condim": "6", "margin": "0.002"},
    )
    ET.SubElement(default, "joint", {"damping": "0.03", "frictionloss": "0.001", "armature": "0.0015"})

    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "main_light", "pos": "0 0 4", "dir": "0 0 -1", "diffuse": "1 1 1", "specular": "0.5 0.5 0.5"},
    )
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "floor", "type": "plane", "pos": "0 0 0", "size": "10 10 0.1", "material": "grid"},
    )

    hand_body = ET.SubElement(worldbody, "body", {"name": "hand", "pos": "0 0 0.5", "euler": "3.1415926 0 0"})
    palm_body = ET.SubElement(hand_body, "body", {"name": "palm"})

    if palm_part_files:
        for i, _ in enumerate(palm_part_files):
            ET.SubElement(
                palm_body,
                "geom",
                {"name": f"palm_geom_{i}", "type": "mesh", "mesh": f"palm_part{i}", "group": "1", "rgba": "1 1 1 1"},
            )
    else:
        ET.SubElement(
            palm_body,
            "geom",
            {"name": "palm_geom_0", "type": "mesh", "mesh": "palm_part0", "group": "1", "rgba": "1 1 1 1"},
        )

    ET.SubElement(
        palm_body,
        "geom",
        {"name": "palm_vis_geom", "type": "mesh", "mesh": "palm_vis", "group": "0", "contype": "0", "conaffinity": "0"},
    )

    for i in range(5):
        finger_num = i + 1
        if i < args.actual_finger_count:
            frame = frame_data["frames"][i]
            pos = frame["pos"]
            quat = frame["quat"]
            frame_elem = ET.SubElement(
                hand_body,
                "frame",
                {
                    "pos": f"{pos['x']:.6f} {pos['y']:.6f} {pos['z']:.6f}",
                    "quat": f"{quat['w']:.6f} {quat['x']:.6f} {quat['y']:.6f} {quat['z']:.6f}",
                },
            )
        else:
            frame_elem = ET.SubElement(hand_body, "frame", {"pos": "0 0 0", "quat": "1 0 0 0"})

        ET.SubElement(frame_elem, "attach", {"model": f"finger_model_{finger_num}", "body": "finger_root", "prefix": f"f{finger_num}_"})

    ET.SubElement(worldbody, "camera", {"name": "fixed", "pos": "0 -3 2", "xyaxes": "1 0 0 0 1 2"})

    rough = ET.tostring(mujoco, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write("\n".join(lines))

    print(f"Generated complete hand XML: {args.out}")


if __name__ == "__main__":
    main()
