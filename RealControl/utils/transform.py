"""
Coordinate transformation utilities for converting between Isaac Lab and MuJoCo orderings.
"""
import numpy as np


def reorder_isaaclab_to_mujoco(a):
    """
    Convert from Isaac Lab ordering to MuJoCo ordering.
    
    Isaac Lab ordering: [palm0-4, base0-4, middle0-4, end0-4] (20 elements)
    MuJoCo ordering: [palm0,base0,middle0,end0, palm1,base1,middle1,end1, ...] (20 elements)
    
    Args:
        a: Array-like of shape (20,) in Isaac Lab ordering
        
    Returns:
        np.ndarray: Array in MuJoCo ordering
    """
    a = np.asarray(a).reshape(-1)
    if a.size != 20:
        raise ValueError(f"Expected 20-dim, got {a.size}")
    
    out = np.empty_like(a)
    for i in range(5):  # 5 fingers
        b = 4 * i  # base index for finger i in MuJoCo ordering
        out[b + 0] = a[0 * 5 + i]  # palm
        out[b + 1] = a[1 * 5 + i]  # base
        out[b + 2] = a[2 * 5 + i]  # middle
        out[b + 3] = a[3 * 5 + i]  # end
    return out


def reorder_mujoco_to_isaaclab(a):
    """
    Convert from MuJoCo ordering to Isaac Lab ordering.
    
    MuJoCo ordering: [palm0,base0,middle0,end0, palm1,base1,middle1,end1, ...] (20 elements)
    Isaac Lab ordering: [palm0-4, base0-4, middle0-4, end0-4] (20 elements)
    
    Args:
        a: Array-like of shape (20,) in MuJoCo ordering
        
    Returns:
        np.ndarray: Array in Isaac Lab ordering
    """
    a = np.asarray(a).reshape(-1)
    if a.size != 20:
        raise ValueError(f"Expected 20-dim, got {a.size}")
    
    out = np.empty_like(a)
    for i in range(5):  # 5 fingers
        b = 4 * i  # base index for finger i in MuJoCo ordering
        out[0 * 5 + i] = a[b + 0]  # palm
        out[1 * 5 + i] = a[b + 1]  # base
        out[2 * 5 + i] = a[b + 2]  # middle
        out[3 * 5 + i] = a[b + 3]  # end
    return out


def get_default_joint_positions_sim_mj(config):
    """
    Get default joint positions in MuJoCo ordering and simulation radians.
    
    Args:
        config: Config object containing default Isaac Lab joint positions
        
    Returns:
        np.ndarray: Default joint positions in MuJoCo ordering (sim-rad)
    """
    isaac_order = config.default_joint_positions_isaac
    return reorder_isaaclab_to_mujoco(isaac_order)

if __name__ == "__main__":
    mujoco_tensor = np.array([
        0.11168, 1.15337, 1.05087, 1.48077,    
        0.18148, -0.000187692, 1.33139, 1.28281,
        0.37692, 1.07247, 1.11696, 1.25157,     
        0.41182, 0.804082, 1.24389, 1.30853,    
        1.95882e-14, 5.96076e-08, 6.02116e-08, 4.35014e-08
    ])

    mujoco_tensor = np.array([-3.68089e-10, 0.927515, 1.18507, 1.5711, 
                              -2.32723e-10, -0.000228377, 1.29877, 1.57103, 
                              5.30275e-10, 0.686113, 1.32886, 1.5203, 
                              1.9589e-14, 5.96076e-08, 6.02116e-08, 4.35014e-08, 
                              1.95882e-14, 5.96076e-08, 6.02116e-08, 4.35014e-08])
    mujoco_tensor = np.array([-1.16345e-10, -0.000113806, 1.23859, 1.46726, 
                              -1.12204e-07, 0.807236, 1.31, 1.5711, 
                              -0.0348999, 0.486601, 1.22414, 1.56224, 
                              5.97316e-08, -0.000216751, 1.17354, 1.57103, 
                              0.000151382, 0.782352, 1.2881, 1.34602])

    mujoco_tensor = np.array([-2.21991e-10, 0.701115, 1.16988, 1.57103, 4.28419e-05, 0.899371, 0.970408, 1.57094, 1.21269e-08, 0.803137, 1.17948, 1.57102, 3.85905e-05, 0.908186, 0.869432, 1.57104, 1.82049e-07, 1.02404, 0.759448, 1.57104])

    mujoco_tensor = np.array([6.596e-10, 0.813297, 1.4288, 1.38664, -0.000150261, 1.05466, 1.13151, 1.37826, 7.32339e-10, 0.88774, 1.26957, 1.18286, 1.94647e-09, 0.737906, 0.978077, 1.50499, 3.70264e-08, 0.765509, 0.970433, 1.57104])

    mujoco_tensor = np.array([-1.59701e-09, 0.494768, 1.08501, 1.57104, -4.9303e-10, 0.704919, 1.06308, 1.57095, -1.31244e-09, 0.578609, 1.0309, 1.57104, 1.53977e-07, 0.710607, 0.86498, 1.57104, 3.55728e-09, 0.661388, 1.06465, 1.57104])

    isaaclab_tensor = reorder_mujoco_to_isaaclab(mujoco_tensor)
    
        # Print in ready-to-paste format
    print("Isaac Lab ordering (ready to paste):")
    print(", ".join(f"{x:.8f}" if abs(x) > 1e-6 else f"{x:.8e}" for x in isaaclab_tensor))
    
    # Also print with square brackets for array format
    print("\nAs array:")
    print("[" + ", ".join(f"{x:.8f}" if abs(x) > 1e-6 else f"{x:.8e}" for x in isaaclab_tensor) + "]")