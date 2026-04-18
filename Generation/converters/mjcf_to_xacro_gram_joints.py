#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MJCF → URDF/Xacro converter (handles randomized grammar stacking & removed joints)

- Detects grammar stacking (nested or sibling) and consolidates grammar meshes into
  their parent link (no grammar links by default).
- Detects "ghosted" joints (tiny range) and keeps their links/joints with tiny
  inertials + tiny range + heavy damping/friction.
- Computes correct URDF revolute joint origins from absolute hinge positions:
    origin(parent→child) = p_child_hinge - p_parent_last_hinge
- Applies visual/collision "compensation" = negative of the last upstream revolute abs pos.
- Inserts fixed-joint offsets δ by summing MJCF body pos along fixed chains (between
  lever and the next servo body). Can be flattened to zero with a flag.

Usage:
  python converters/mjcf_to_xacro_finger.py input.xml -o out.urdf.xacro

Helpful flags:
  --flatten-stack-offsets    : set fixed offsets from stacking to zero (default: False)
  --keep-grammar1-ghost      : insert a ghost link between base_lever and middle (default: False)
  --ghost-range-eps 1e-6     : threshold for detecting ghosted joints
  --tiny-mass 1e-9           : mass used for ghost links
  --tiny-inertia 1e-12       : diagonal inertia used for ghost links
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import re
from typing import Optional, Tuple, List

# ------------------- Vec helpers -------------------

def parse_vec3(s: str) -> Tuple[float, float, float]:
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 numbers, got: '{s}'")
    return (vals[0], vals[1], vals[2])

def vadd(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vsub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vscale(a, k): return (a[0]*k, a[1]*k, a[2]*k)
def f3(v): return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"

# ------------------- MJCF parsing -------------------

def get_worldbody(root: ET.Element) -> ET.Element:
    wb = root.find('worldbody')
    if wb is None:
        raise RuntimeError("No <worldbody> in MJCF.")
    return wb

def find_body(root: ET.Element, name: str) -> Optional[ET.Element]:
    return root.find(f".//body[@name='{name}']")

def list_direct_children(b: ET.Element) -> List[ET.Element]:
    return [x for x in b.findall("./body")]

def find_joint_info(root: ET.Element, jname: str):
    for j in root.findall(".//joint"):
        if j.get('name') == jname:
            pos = parse_vec3(j.get('pos', '0 0 0'))
            axis = tuple(float(x) for x in j.get('axis', '0 0 1').split())
            rng = None
            if j.get('range'):
                lo, hi = [float(x) for x in j.get('range').split()]
                rng = (lo, hi)
            return {"pos": pos, "axis": axis, "range": rng}
    return None

def is_ghost_joint(jinfo, eps: float) -> bool:
    if not jinfo: 
        return True
    if not jinfo.get("range"):
        return False
    lo, hi = jinfo["range"]
    return abs(hi - lo) < eps

def body_has_visual_geom(b: ET.Element) -> bool:
    """Heuristic: any geom referencing a '/visual/' mesh or non-vis mesh counts as visual/collision presence."""
    return b.find(".//geom") is not None

def grammar_site_info(parent_body: ET.Element, site_prefix: str):
    """
    Detect grammar stacks under parent_body (e.g. base_lever→grammar1, middle_lever→grammar2).

    Returns (count, delta, style) where:
      - count: total stacks (includes stack0 if present)
      - delta: Δg taken from the first _1 body (0,0,0 if not found)
      - style: 'nested' | 'sibling' | 'none'
    Handles both:
      1) Geoms directly under parent_body (stack0), and
      2) An intermediate wrapper body named 'finger_{site_prefix}' that holds stack0 geom.
    """
    if parent_body is None:
        return 0, (0.0, 0.0, 0.0), 'none'

    count = 0
    delta = (0.0, 0.0, 0.0)
    style = 'none'

    # Optional wrapper like <body name="finger_grammar2"> ... </body>
    wrapper = parent_body.find(f"./body[@name='finger_{site_prefix}']")

    def has_stack0_here(node: ET.Element) -> bool:
        if node is None:
            return False
        # stack0 must be a geom directly under this node
        for g in node.findall("./geom"):
            m = g.get("mesh", "")
            if m.endswith(f"finger_{site_prefix}"):
                return True
        return False

    # stack0 detection: direct under lever OR direct under wrapper
    if has_stack0_here(parent_body) or has_stack0_here(wrapper):
        count += 1

    # Prefer to search under wrapper if it exists; else under parent
    search_root = wrapper if wrapper is not None else parent_body

    # --- NESTED style: _1 is a descendant of search_root and each _k is a child of _{k-1}
    b1 = search_root.find(f".//body[@name='finger_{site_prefix}_1']")
    if b1 is not None:
        style = 'nested'
        delta = parse_vec3(b1.get('pos', '0 0 0'))
        count += 1  # for _1
        cur = b1
        i = 2
        while True:
            nxt = cur.find(f"./body[@name='finger_{site_prefix}_{i}']")
            if nxt is None:
                break
            count += 1
            cur = nxt
            i += 1
        return count, delta, style

    # --- SIBLING style: numbered bodies are direct children of search_root
    sib_idxs = []
    for child in search_root.findall("./body"):
        nm = child.get('name') or ""
        m = re.match(rf"finger_{site_prefix}_(\d+)$", nm)
        if m:
            sib_idxs.append(int(m.group(1)))
    sib_idxs.sort()

    if sib_idxs:
        style = 'sibling'
        count += len(sib_idxs)
        first = search_root.find(f"./body[@name='finger_{site_prefix}_{sib_idxs[0]}']")
        if first is not None:
            delta = parse_vec3(first.get('pos', '0 0 0'))
        return count, delta, style

    return count, delta, style


def accumulate_fixed_offset(parent: ET.Element, target_name: str) -> Tuple[bool, Tuple[float,float,float]]:
    """
    DFS to find 'target_name' under 'parent'; sum 'pos' along the path (fixed chain).
    Stops at first match; returns (found, sum_pos).
    """
    def dfs(node: ET.Element, acc) -> Optional[Tuple[float,float,float]]:
        # If this node is the target, done
        if node.get('name') == target_name:
            return acc
        # Add children (bodies). For each body, add its 'pos' before recursing.
        for child in node.findall("./body"):
            pos = parse_vec3(child.get('pos', '0 0 0'))
            got = dfs(child, vadd(acc, pos))
            if got is not None:
                return got
        return None

    res = dfs(parent, (0.0,0.0,0.0))
    return (res is not None), (res if res is not None else (0.0,0.0,0.0))

# ------------------- URDF/Xacro emit -------------------

def add(robot, tag, attrib=None, text=None):
    el = ET.SubElement(robot, tag, attrib or {})
    if text is not None:
        el.text = text
    return el

def emit_inertial(link, mass="0.01", inertia="0.00001"):
    inert = add(link, 'inertial')
    add(inert, 'mass', {'value': mass})
    add(inert, 'inertia', {'ixx': inertia, 'ixy': '0', 'ixz': '0', 'iyy': inertia, 'iyz': '0', 'izz': inertia})

def emit_ghost_inertial(link, tiny_mass, tiny_inertia):
    inert = add(link, 'inertial')
    add(inert, 'mass', {'value': f"{tiny_mass}"})
    add(inert, 'inertia', {'ixx': f"{tiny_inertia}", 'ixy': '0', 'ixz': '0',
                           'iyy': f"{tiny_inertia}", 'iyz': '0', 'izz': f"{tiny_inertia}"})

def emit_visual(link, mesh_vis, origin=None):
    vis = add(link, 'visual')
    if origin is not None:
        add(vis, 'origin', {'xyz': f3(origin), 'rpy': '0 0 0'})
    ge = add(vis, 'geometry')
    add(ge, 'mesh', {'filename': f'robot_meshes/visual/{mesh_vis}'})

def emit_collision(link, mesh, origin=None):
    col = add(link, 'collision')
    if origin is not None:
        add(col, 'origin', {'xyz': f3(origin), 'rpy': '0 0 0'})
    ge = add(col, 'geometry')
    add(ge, 'mesh', {'filename': f'robot_meshes/{mesh}'})

def emit_link_functional(parent, name, comp, vis_mesh, coll_meshes, mass="0.01", inertia="0.00001"):
    link = add(parent, 'link', {'name': name})
    emit_visual(link, vis_mesh, comp)
    for cm in coll_meshes:
        emit_collision(link, cm, comp)
    emit_inertial(link, mass, inertia)
    return link

def emit_tip_link(parent, name, comp, vis_mesh, coll_meshes, mass="0.01", inertia="0.00001", is_ghost=False):
    """
    Emit tip link with dynamic collision meshes based on fingertip type.
    For ghost fingers, only add inertial properties.
    """
    link = add(parent, 'link', {'name': name})
    
    if not is_ghost:
        emit_visual(link, vis_mesh, comp)
        for cm in coll_meshes:
            emit_collision(link, cm, comp)
    
    if is_ghost:
        emit_ghost_inertial(link, 1e-9, 1e-12)  # Use ghost inertials
    else:
        emit_inertial(link, mass, inertia)
    return link

def emit_link_ghost(parent, name, tiny_mass, tiny_inertia):
    link = add(parent, 'link', {'name': name})
    emit_ghost_inertial(link, tiny_mass, tiny_inertia)
    return link

def emit_link_with_grammar(parent, name, comp, parent_vis, parent_coll_list,
                           gram_count, gram_delta, gram_vis, gram_coll,
                           mass="0.01", inertia="0.00001"):
    link = add(parent, 'link', {'name': name})
    # parent visuals
    emit_visual(link, parent_vis, comp)
    for cm in parent_coll_list:
        emit_collision(link, cm, comp)
    # grammar visuals/colls
    for i in range(gram_count):
        offset = comp if i == 0 else vadd(comp, vscale(gram_delta, i))
        emit_visual(link, gram_vis, offset)
        emit_collision(link, gram_coll, offset)
    # inertial: bump for grammar
    base_m = float(mass)
    base_I = float(inertia)
    total_m = base_m + gram_count * 0.005
    total_I = base_I + gram_count * 0.000005
    emit_inertial(link, f"{total_m:.5f}", f"{total_I:.8f}")
    return link

def emit_fixed_joint(parent, name, p_link, c_link, origin=(0.0,0.0,0.0)):
    j = add(parent, 'joint', {'name': name, 'type': 'fixed'})
    add(j, 'parent', {'link': p_link})
    add(j, 'child',  {'link': c_link})
    add(j, 'origin', {'xyz': f3(origin), 'rpy': '0 0 0'})
    return j

def emit_revolute_joint(parent, name, p_link, c_link, origin, axis, limits, damping="1.0", friction=None):
    j = add(parent, 'joint', {'name': name, 'type': 'revolute'})
    add(j, 'parent', {'link': p_link})
    add(j, 'child',  {'link': c_link})
    add(j, 'origin', {'xyz': f3(origin), 'rpy': '0 0 0'})
    add(j, 'axis',   {'xyz': f3(axis)})
    lo, hi = limits if limits else (0.0, 0.0)
    add(j, 'limit',  {'lower': f"{lo}", 'upper': f"{hi}", 'effort': '10', 'velocity': '1.0'})
    dyn = add(j, 'dynamics', {'damping': f"{damping}"})
    if friction is not None:
        dyn.set('friction', f"{friction}")
    return j

def prettify(elem) -> str:
    rough = ET.tostring(elem, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    # drop xml decl to match common xacro style
    lines = pretty.splitlines()
    if lines and lines[0].startswith('<?xml'):
        lines = lines[1:]
    return "\n".join(lines)

# ------------------- Conversion -------------------

def detect_fingertip_config(root: ET.Element) -> Tuple[str, List[str]]:
    """
    Detect fingertip configuration from MJCF by examining assets and geoms.
    Returns (visual_file, collision_files_list).
    """
    # Find tip visual mesh from assets
    visual_file = "finger_tip_vis.obj"  # default fallback
    for mesh in root.findall(".//asset/mesh"):
        mesh_name = mesh.get("name", "")
        if mesh_name == "finger_tip_vis":
            file_path = mesh.get("file", "")
            if file_path.startswith("robot_meshes/visual/"):
                visual_file = file_path.replace("robot_meshes/visual/", "")
            break
    
    # Find collision meshes by examining tip body geoms
    collision_files = []
    tip_body = root.find(".//body[@name='finger_tip']")
    if tip_body is not None:
        # Collect all collision geoms (group != "0")
        for geom in tip_body.findall("./geom"):
            group = geom.get("group", "1")
            if group != "0":  # not visual
                mesh_name = geom.get("mesh", "")
                if mesh_name.startswith("finger_tip_part"):
                    # Find corresponding file in assets
                    for asset_mesh in root.findall(".//asset/mesh"):
                        if asset_mesh.get("name") == mesh_name:
                            file_path = asset_mesh.get("file", "")
                            if file_path.startswith("robot_meshes/"):
                                collision_files.append(file_path.replace("robot_meshes/", ""))
                            break
    
    # Fallback if nothing found
    if not collision_files:
        collision_files = ["finger_tip_part0.obj"]
    
    return visual_file, collision_files

def convert(mjcf_path: str, out_path: str,
            flatten_stack_offsets: bool = False,
            keep_grammar1_ghost: bool = False,
            ghost_eps: float = 1e-6,
            tiny_mass: float = 1e-9,
            tiny_inertia: float = 1e-12):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    wb = get_worldbody(root)

    # Detect fingertip configuration
    tip_vis_file, tip_coll_files = detect_fingertip_config(root)

    # Bodies we care about (by your naming)
    palm_servo = find_body(wb, "finger_palm_servo")
    base       = find_body(wb, "finger_base")
    base_lev   = find_body(wb, "finger_base_lever")
    middle     = find_body(wb, "finger_middle")
    mid_lev    = find_body(wb, "finger_middle_lever")
    end        = find_body(wb, "finger_end")
    end_lev    = find_body(wb, "finger_end_lever")
    tip        = find_body(wb, "finger_tip")

    # Joints (abs hinge positions & axes)
    jpalm = find_joint_info(wb, "finger_palm_joint") or {"pos":(0,0,0),"axis":(0,0,1),"range":(-0.698,0.698)}
    jbase = find_joint_info(wb, "finger_base_joint")
    jmid  = find_joint_info(wb, "finger_middle_joint")
    jend  = find_joint_info(wb, "finger_end_joint")

    p0 = jpalm["pos"]
    p1 = jbase["pos"] if jbase else p0
    p2 = jmid["pos"]  if jmid  else p1
    p3 = jend["pos"]  if jend  else p2

    # Ghost detection
    base_ghost   = is_ghost_joint(jbase, ghost_eps) if jbase else True
    middle_ghost = is_ghost_joint(jmid,  ghost_eps) if jmid  else True
    end_ghost    = is_ghost_joint(jend,  ghost_eps) if jend  else False  # usually real

    # Check if this is a ghost finger by looking for minimal structure
    is_ghost_finger = (base_ghost and middle_ghost and 
                       not body_has_visual_geom(palm_servo) and
                       not body_has_visual_geom(tip))

    # Grammar discovery (counts & Δg)
    g1_count, g1_dg, g1_style = grammar_site_info(base_lev, "grammar1")
    g2_count, g2_dg, g2_style = grammar_site_info(mid_lev,  "grammar2")

    # Fixed offsets (δ) along chains (lever → next servo body)
    found_mid_off,  mid_off  = accumulate_fixed_offset(base_lev, "finger_middle") if base_lev is not None else (False,(0,0,0))
    found_end_off,  end_off  = accumulate_fixed_offset(mid_lev,  "finger_end")    if mid_lev  is not None else (False,(0,0,0))
    # sanity: if we detect grammar stacking but didn't find target bodies, fallback to i*Δg for last stack
    if g1_count > 1 and not found_mid_off:
        mid_off = vscale(g1_dg, g1_count-1)
    if g2_count > 1 and not found_end_off:
        end_off = vscale(g2_dg, g2_count-1)

    if flatten_stack_offsets:
        mid_off = (0.0,0.0,0.0)
        end_off = (0.0,0.0,0.0)

    # Output Xacro
    robot = ET.Element('robot', {'xmlns:xacro': 'http://www.ros.org/wiki/xacro'})
    robot.append(ET.Comment(
f"""Auto-converted from {Path(mjcf_path).name}
Grammar: grammar1={g1_style}, grammar2={g2_style}; counts: g1={g1_count}, g2={g2_count}
Δg1={g1_dg}, Δg2={g2_dg}
Offsets: δ_middle={mid_off}, δ_end={end_off}; flatten={flatten_stack_offsets}
Ghost joints: base={base_ghost}, middle={middle_ghost}, end={end_ghost}"""
    ))

    macro_name = Path(mjcf_path).stem or "finger_auto"
    macro = add(robot, 'xacro:macro', {'name': macro_name, 'params': 'prefix parent *origin'})

    # Root link + mount
    link_root = add(macro, 'link', {'name': '${prefix}finger_root'})
    emit_inertial(link_root, "0.001", "0.000001")
    jmount = add(macro, 'joint', {'name': '${prefix}root_joint', 'type': 'fixed'})
    add(jmount, 'parent', {'link': '${parent}'})
    add(jmount, 'child',  {'link': '${prefix}finger_root'})
    add(jmount, 'xacro:insert_block', {'name': 'origin'})

    # Palm servo (fixed to root)
    link_ps = add(macro, 'link', {'name': '${prefix}finger_palm_servo'})
    if not is_ghost_finger:
        emit_visual(link_ps, 'finger_palm_servo_vis.obj', None)
        emit_collision(link_ps, 'finger_palm_servo_part0.obj', None)
        emit_inertial(link_ps, "0.01", "0.00001")
    else:
        emit_ghost_inertial(link_ps, tiny_mass, tiny_inertia)
    emit_fixed_joint(macro, '${prefix}palm_servo_joint', '${prefix}finger_root', '${prefix}finger_palm_servo', (0.0,0.0,0.0))

    # Base (link)
    if base_ghost:
        emit_link_ghost(macro, '${prefix}finger_base', tiny_mass, tiny_inertia)
    else:
        emit_link_functional(macro, '${prefix}finger_base', vscale(p0,-1.0),
                             'finger_base_vis.obj', ['finger_base_servo_part0.obj'])
    # Palm joint (always present as revolute in your models)
    emit_revolute_joint(macro, '${prefix}finger_palm_joint', '${prefix}finger_palm_servo', '${prefix}finger_base',
                        origin=p0, axis=jpalm['axis'], limits=jpalm.get('range', (-0.698,0.698)))

    # Base lever
    if base_ghost:
        emit_link_ghost(macro, '${prefix}finger_base_lever', tiny_mass, tiny_inertia)
        # Ghosted base joint
        emit_revolute_joint(macro, '${prefix}finger_base_joint', '${prefix}finger_base', '${prefix}finger_base_lever',
                            origin=vsub(p1, p0), axis=(jbase['axis'] if jbase else (-1,0,0)),
                            limits=(-0.001, 0.001), damping="10.0", friction="1.0")
    else:
        # functional + consolidated grammar1
        emit_link_with_grammar(macro, '${prefix}finger_base_lever', vscale(p1,-1.0),
                               'finger_base_lever_vis.obj',
                               ['finger_base_lever_part0.obj','finger_base_lever_part1.obj','finger_base_lever_part2.obj'],
                               gram_count=g1_count, gram_delta=g1_dg,
                               gram_vis='finger_grammar1_vis.obj', gram_coll='finger_grammar1_part0.obj')
        # Real base joint
        emit_revolute_joint(macro, '${prefix}finger_base_joint', '${prefix}finger_base', '${prefix}finger_base_lever',
                            origin=vsub(p1, p0), axis=jbase['axis'], limits=jbase['range'])

    # Optional grammar1 ghost slot (keeps naming slot, even when consolidated)
    if keep_grammar1_ghost:
        emit_link_ghost(macro, '${prefix}finger_grammar1', tiny_mass, tiny_inertia)
        emit_fixed_joint(macro, '${prefix}grammar1_joint', '${prefix}finger_base_lever', '${prefix}finger_grammar1', (0.0,0.0,0.0))
        mid_parent = '${prefix}finger_grammar1'
    else:
        mid_parent = '${prefix}finger_base_lever'

    # Middle servo link
    if middle_ghost:
        emit_link_ghost(macro, '${prefix}finger_middle', tiny_mass, tiny_inertia)
    else:
        emit_link_functional(macro, '${prefix}finger_middle', vscale(p1,-1.0),
                             'finger_middle_vis.obj', ['finger_middle_servo_part0.obj'])
    # Fixed from (base_lever or grammar1) → middle, with δ
    emit_fixed_joint(macro, '${prefix}middle_joint', mid_parent, '${prefix}finger_middle', origin=mid_off)

    # Middle lever (with consolidated grammar2)
    if middle_ghost:
        emit_link_ghost(macro, '${prefix}finger_middle_lever', tiny_mass, tiny_inertia)
        # Ghosted middle revolute
        emit_revolute_joint(macro, '${prefix}finger_middle_joint', '${prefix}finger_middle', '${prefix}finger_middle_lever',
                            origin=vsub(p2, p1), axis=(jmid['axis'] if jmid else (-1,0,0)),
                            limits=(-0.001, 0.001), damping="10.0", friction="1.0")
    else:
        emit_link_with_grammar(macro, '${prefix}finger_middle_lever', vscale(p2,-1.0),
                               'finger_middle_lever_vis.obj',
                               ['finger_middle_lever_part0.obj','finger_middle_lever_part1.obj','finger_middle_lever_part2.obj'],
                               gram_count=g2_count, gram_delta=g2_dg,
                               gram_vis='finger_grammar2_vis.obj', gram_coll='finger_grammar2_part0.obj')
        emit_revolute_joint(macro, '${prefix}finger_middle_joint', '${prefix}finger_middle', '${prefix}finger_middle_lever',
                            origin=vsub(p2, p1), axis=jmid['axis'], limits=jmid['range'])

    # End (fixed to middle lever) — fixed origin is the cumulative grammar2 offset unless flattened
    if end_ghost:
        emit_link_ghost(macro, '${prefix}finger_end', tiny_mass, tiny_inertia)
    else:
        emit_link_functional(macro, '${prefix}finger_end', vscale(p2,-1.0),
                             'finger_end_vis.obj', ['finger_end_servo_part0.obj'])
    emit_fixed_joint(macro, '${prefix}end_joint', '${prefix}finger_middle_lever', '${prefix}finger_end', origin=end_off)

    # End lever + revolute
    if end_ghost:
        emit_link_ghost(macro, '${prefix}finger_end_lever', tiny_mass, tiny_inertia)
        emit_revolute_joint(macro, '${prefix}finger_end_joint', '${prefix}finger_end', '${prefix}finger_end_lever',
                            origin=vsub(p3, p2), axis=(jend['axis'] if jend else (-1,0,0)),
                            limits=(-0.001, 0.001), damping="10.0", friction="1.0")
    else:
        emit_link_functional(macro, '${prefix}finger_end_lever', vscale(p3,-1.0),
                             'finger_end_lever_vis.obj',
                             ['finger_end_lever_part0.obj','finger_end_lever_part1.obj','finger_end_lever_part2.obj'])
        emit_revolute_joint(macro, '${prefix}finger_end_joint', '${prefix}finger_end', '${prefix}finger_end_lever',
                            origin=vsub(p3, p2), axis=jend['axis'], limits=jend['range'])

    # Tip (fixed) - use detected configuration, handle ghost case
    if is_ghost_finger:
        emit_tip_link(macro, '${prefix}finger_tip', vscale(p3,-1.0),
                      tip_vis_file, tip_coll_files, is_ghost=True)
    else:
        emit_tip_link(macro, '${prefix}finger_tip', vscale(p3,-1.0),
                      tip_vis_file, tip_coll_files)
    emit_fixed_joint(macro, '${prefix}tip_joint', '${prefix}finger_end_lever', '${prefix}finger_tip', (0.0,0.0,0.0))

    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + prettify(robot)
    Path(out_path).write_text(xml, encoding='utf-8')

    # stdout summary
    print(f"Converted {mjcf_path} → {out_path}")
    print(f"p0={p0}, p1={p1}, p2={p2}, p3={p3}")
    print(f"grammar1: count={g1_count}, Δg={g1_dg}, style={g1_style}")
    print(f"grammar2: count={g2_count}, Δg={g2_dg}, style={g2_style}")
    print(f"δ_middle={mid_off}, δ_end={end_off}, flatten={flatten_stack_offsets}")
    print(f"ghost(base)={base_ghost}, ghost(middle)={middle_ghost}, ghost(end)={end_ghost}")
    print(f"fingertip: visual={tip_vis_file}, collision_parts={len(tip_coll_files)}")

def main():
    ap = argparse.ArgumentParser(description="Convert MJCF finger → URDF/Xacro (random grammar stacks & ghost joints).")
    ap.add_argument("mjcf", help="Input MJCF XML file")
    ap.add_argument("-o","--out", default=None, help="Output URDF/Xacro path (default: <stem>.urdf.xacro)")
    ap.add_argument("--flatten-stack-offsets", action="store_true",
                    help="Set fixed offsets from stacking to zero (default: False)")
    ap.add_argument("--keep-grammar1-ghost", action="store_true",
                    help="Insert a ghost grammar1 link between base_lever and middle (default: False)")
    ap.add_argument("--ghost-range-eps", type=float, default=1e-6,
                    help="Threshold for treating a joint as ghosted (upper-lower < eps)")
    ap.add_argument("--tiny-mass", type=float, default=1e-9, help="Ghost link mass")
    ap.add_argument("--tiny-inertia", type=float, default=1e-12, help="Ghost link diagonal inertia")
    args = ap.parse_args()

    out = args.out or str(Path(args.mjcf).with_suffix(".urdf.xacro"))
    convert(args.mjcf, out,
            flatten_stack_offsets=args.flatten_stack_offsets,
            keep_grammar1_ghost=args.keep_grammar1_ghost,
            ghost_eps=args.ghost_range_eps,
            tiny_mass=args.tiny_mass,
            tiny_inertia=args.tiny_inertia)

if __name__ == "__main__":
    main()
