import trimesh
import os

file_thresholds = {
    "palm": 0.2,
    "finger1_palm_servo": 0.18, # perfect!
    "finger1_base": 0.1,
    "finger1_bg": 0.2,
    "finger1_grammar1": 0.2,
    "finger1_middle": 0.1,
    "finger1_mg": 0.2,
    "finger1_grammar2": 0.2,
    "finger1_end": 0.1,
    "finger1_tip": 0.1, # perfect!
}

input_file_format = "./meshes/robot_meshes2/{f_name}.obj"

for f_name, threshold in file_thresholds.items():
    input_file = input_file_format.format(f_name=f_name)
    
    if not os.path.isfile(input_file):
        print(input_file, "is not a file")
        continue
    
    mesh = trimesh.load(input_file, force="mesh")
    
    # Check if the mesh is watertight
    print(f"Checking {f_name}:")
    print("Watertight?", mesh.is_watertight)
    # print("Non-manifold edges:", mesh.non_manifold_edges.shape[0])


