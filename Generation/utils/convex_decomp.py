import trimesh
from trimesh.util import concatenate
import numpy as np
import sys
import os
import argparse
import coacd

# NOTE: not using --input-dir uses the file thresholds shown below

# mesh = trimesh.load(input_file, force="mesh")
# mesh = coacd.Mesh(mesh.vertices, mesh.faces)
# parts = coacd.run_coacd(mesh) # a list of convex hulls.


file_thresholds = {
    # "palm": 0.08, # good enough, outputs 4 parts of palm
    # "finger_palm_servo": 0.5, # good enough but needs collision exclusion between palm and intra-finger
    # "finger_base_servo": 0.13, # ok ish
    # "finger_base_lever": 0.2, # ok ish
    # "finger_grammar1": 0.2, # OK
    # "finger_middle_servo": 0.13, # ok ish
    # "finger_middle_lever": 0.2,
    # "finger_grammar2": 0.2, # OK
    # "finger_end_servo": 0.13, # ok ish
    # "finger_tip": 0.2, # perfect!
    # "finger_end_lever": 0.2,
    
    # "finger1_tip2": 0.1,
    # "finger_tip_rounded": 0.1,
    # "finger_tip_wedged":0.1,
    # "finger_tip_thinner": 0.1,
    
    # "palm_hull": 0.1,
    "palm1": 0.1,
    "palm2": 0.1
}
input_file_formats = list(file_thresholds.keys())

split_files = True
merge_meshes = False  # merge all convex hulls into one mesh - only for non-split files
input_file_format = "./meshes/lego_hand/robot_meshes/{f_name}.obj"
output_file_format = "./meshes/lego_hand/robot_meshes/{f_name}.obj"
quiet = False
max_convex_hull = -1
preprocess_mode = 'auto'
prep_resolution = 50
resolution = 2000
mcts_node = 20
mcts_iteration = 100
mcts_max_depth = 3
pca = False
no_merge = False
decimate = False
max_ch_vertex = 256
extrude = False
extrude_margin = 0.01
apx_mode = 'ch'
seed = 36

if split_files:
    for f_name, threshold in file_thresholds.items():
        input_file  = input_file_format.format(f_name=f_name)
        out_dir     = "./meshes/lego_hand/robot_meshes"   # keep directory once
        stem        = f_name                          # filename stem

        if not os.path.isfile(input_file):
            print(f"missing {input_file}")
            continue
        os.makedirs(out_dir, exist_ok=True)

        # === run CoACD exactly as before ===
        raw = trimesh.load(input_file, force="mesh")
        parts = coacd.run_coacd(
            coacd.Mesh(raw.vertices, raw.faces),
            threshold=threshold, max_convex_hull=max_convex_hull,
            preprocess_mode=preprocess_mode, preprocess_resolution=prep_resolution,
            resolution=resolution, mcts_nodes=mcts_node,
            mcts_iterations=mcts_iteration, mcts_max_depth=mcts_max_depth,
            pca=pca, merge=not no_merge, decimate=decimate,
            max_ch_vertex=max_ch_vertex, extrude=extrude,
            extrude_margin=extrude_margin, apx_mode=apx_mode, seed=seed)

        # === export each hull ===
        for idx, (vs, fs) in enumerate(parts):
            hull = trimesh.Trimesh(vs, fs, process=False)
            out_path = os.path.join(out_dir, f"{stem}_part{idx}.obj")
            hull.export(out_path)
            print(f"   wrote {out_path}")
else:
    for f_name, threshold in file_thresholds.items():
        input_file = input_file_format.format(f_name=f_name)
        output_file = output_file_format.format(f_name=f_name)

        if not os.path.isfile(input_file):
            print(input_file, "is not a file")
            continue

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
            
        if quiet:
            coacd.set_log_level("error")

        mesh = trimesh.load(input_file, force="mesh")
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(
            mesh,
            threshold=threshold,
            max_convex_hull=max_convex_hull,
            preprocess_mode=preprocess_mode,
            preprocess_resolution=prep_resolution,
            resolution=resolution,
            mcts_nodes=mcts_node,
            mcts_iterations=mcts_iteration,
            mcts_max_depth=mcts_max_depth,
            pca=pca,
            merge=not no_merge,
            decimate=decimate,
            max_ch_vertex=max_ch_vertex,
            extrude=extrude,
            extrude_margin=extrude_margin,
            apx_mode=apx_mode,
            seed=seed,
        )
        mesh_parts = []
        for vs, fs in result:
            mesh_parts.append(trimesh.Trimesh(vs, fs))
        
        if merge_meshes:
            # merge them into one mesh
            merged = concatenate(mesh_parts)

            # export a single OBJ containing all faces
            merged.export(output_file)
        else:
            scene = trimesh.Scene()
            np.random.seed(0)
            for p in mesh_parts:
                p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
                scene.add_geometry(p)
                scene.export(output_file)
            print(f"Exported {len(mesh_parts)} parts to {output_file}")

def process_mesh_directory(input_dir: str, output_dir: str = None, threshold: float = 0.1):
    """
    Process all .obj files in a directory with CoACD decomposition.
    Useful for batch processing existing mesh collections.
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .obj files
    obj_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.obj') and '_part' not in file:  # Skip already processed parts
                obj_files.append(os.path.join(root, file))
    
    print(f"Found {len(obj_files)} .obj files to process")
    
    for obj_file in obj_files:
        print(f"\nProcessing: {obj_file}")
        
        # Get relative path structure
        rel_path = os.path.relpath(obj_file, input_dir)
        output_base = os.path.join(output_dir, os.path.splitext(rel_path)[0])
        
        try:
            raw = trimesh.load(obj_file, force="mesh")
            parts = coacd.run_coacd(
                coacd.Mesh(raw.vertices, raw.faces),
                threshold=threshold,
                max_convex_hull=-1,
                preprocess_mode='auto',
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
                apx_mode='ch',
                seed=36
            )
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_base), exist_ok=True)
            
            # Export parts
            for idx, (vs, fs) in enumerate(parts):
                hull = trimesh.Trimesh(vs, fs, process=False)
                part_path = f"{output_base}_part{idx}.obj"
                hull.export(part_path)
                print(f"   wrote {part_path}")
                
        except Exception as e:
            print(f"   Error processing {obj_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convex decomposition using CoACD")
    parser.add_argument("--input-dir", type=str, help="Input directory containing .obj files")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: same as input)")
    parser.add_argument("--threshold", type=float, default=0.1, help="CoACD threshold")
    
    args = parser.parse_args()
    
    if args.input_dir:
        process_mesh_directory(args.input_dir, args.output_dir, args.threshold)
    else:
        # Run original code
        if split_files:
            for f_name, threshold in file_thresholds.items():
                input_file  = input_file_format.format(f_name=f_name)
                out_dir     = "./meshes/lego_hand/robot_meshes"   # keep directory once
                stem        = f_name                          # filename stem

                if not os.path.isfile(input_file):
                    print(f"missing {input_file}")
                    continue
                os.makedirs(out_dir, exist_ok=True)

                # === run CoACD exactly as before ===
                raw = trimesh.load(input_file, force="mesh")
                parts = coacd.run_coacd(
                    coacd.Mesh(raw.vertices, raw.faces),
                    threshold=threshold, max_convex_hull=max_convex_hull,
                    preprocess_mode=preprocess_mode, preprocess_resolution=prep_resolution,
                    resolution=resolution, mcts_nodes=mcts_node,
                    mcts_iterations=mcts_iteration, mcts_max_depth=mcts_max_depth,
                    pca=pca, merge=not no_merge, decimate=decimate,
                    max_ch_vertex=max_ch_vertex, extrude=extrude,
                    extrude_margin=extrude_margin, apx_mode=apx_mode, seed=seed)

                # === export each hull ===
                for idx, (vs, fs) in enumerate(parts):
                    hull = trimesh.Trimesh(vs, fs, process=False)
                    out_path = os.path.join(out_dir, f"{stem}_part{idx}.obj")
                    hull.export(out_path)
                    print(f"   wrote {out_path}")
        else:
            for f_name, threshold in file_thresholds.items():
                input_file = input_file_format.format(f_name=f_name)
                output_file = output_file_format.format(f_name=f_name)

                if not os.path.isfile(input_file):
                    print(input_file, "is not a file")
                    continue

                if not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))
                    
                if quiet:
                    coacd.set_log_level("error")

                mesh = trimesh.load(input_file, force="mesh")
                mesh = coacd.Mesh(mesh.vertices, mesh.faces)
                result = coacd.run_coacd(
                    mesh,
                    threshold=threshold,
                    max_convex_hull=max_convex_hull,
                    preprocess_mode=preprocess_mode,
                    preprocess_resolution=prep_resolution,
                    resolution=resolution,
                    mcts_nodes=mcts_node,
                    mcts_iterations=mcts_iteration,
                    mcts_max_depth=mcts_max_depth,
                    pca=pca,
                    merge=not no_merge,
                    decimate=decimate,
                    max_ch_vertex=max_ch_vertex,
                    extrude=extrude,
                    extrude_margin=extrude_margin,
                    apx_mode=apx_mode,
                    seed=seed,
                )
                mesh_parts = []
                for vs, fs in result:
                    mesh_parts.append(trimesh.Trimesh(vs, fs))
                
                if merge_meshes:
                    # merge them into one mesh
                    merged = concatenate(mesh_parts)

                    # export a single OBJ containing all faces
                    merged.export(output_file)
                else:
                    scene = trimesh.Scene()
                    np.random.seed(0)
                    for p in mesh_parts:
                        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
                        scene.add_geometry(p)
                        scene.export(output_file)
                    print(f"Exported {len(mesh_parts)} parts to {output_file}")