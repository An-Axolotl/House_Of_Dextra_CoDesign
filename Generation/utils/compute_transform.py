import numpy as np

def compute_z_constrained_finger_transform(p1a, p1b, p2a, p2b):
    """
    Compute finger transformation constrained to Z-axis rotation only.
    This eliminates spurious X/Y rotations caused by measurement imprecision.
    
    Args:
        p1a, p1b: Two corresponding points on finger 1
        p2a, p2b: Two corresponding points on finger 2
    
    Returns:
        position: Translation vector [x, y, z]
        quaternion: Rotation quaternion [w, x, y, z] (pure Z-rotation)
    """
    # Convert to numpy arrays
    p1a = np.array(p1a, dtype=float)
    p1b = np.array(p1b, dtype=float)
    p2a = np.array(p2a, dtype=float)
    p2b = np.array(p2b, dtype=float)
    
    # Compute direction vectors
    v1 = p1b - p1a
    v2 = p2b - p2a
    
    # Project vectors onto XY plane (ignore Z for rotation calculation)
    v1_xy = np.array([v1[0], v1[1], 0.0])
    v2_xy = np.array([v2[0], v2[1], 0.0])
    
    # Normalize XY vectors
    v1_xy_norm = v1_xy / np.linalg.norm(v1_xy)
    v2_xy_norm = v2_xy / np.linalg.norm(v2_xy)
    
    # Compute Z-axis rotation angle
    cos_angle = np.dot(v1_xy_norm, v2_xy_norm)
    sin_angle = np.cross(v1_xy_norm, v2_xy_norm)[2]  # Z-component of cross product
    
    angle = np.arctan2(sin_angle, cos_angle)
    
    # Create pure Z-rotation quaternion
    quaternion = [np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]  # [w, x, y, z]
    
    # Create rotation matrix for this pure Z-rotation
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Compute translation using the constraint method
    T_from_a = p2a - rotation_matrix @ p1a
    T_from_b = p2b - rotation_matrix @ p1b
    
    translation = (T_from_a + T_from_b) / 2
    
    # Report the constraint
    error = np.linalg.norm(T_from_a - T_from_b)
    print(f"Z-rotation angle: {np.degrees(angle):.2f}°")
    print(f"Transformation consistency: {error:.6f}")
    
    return translation.tolist(), quaternion

def compute_unconstrained_vs_constrained(p1a, p1b, p2a, p2b):
    """
    Compare unconstrained vs Z-constrained methods
    """
    print("=== COMPARISON: Unconstrained vs Z-Constrained ===")
    
    # Unconstrained method (your original)
    pos_unc, quat_unc = compute_correct_finger_transform(p1a, p1b, p2a, p2b)
    
    # Z-constrained method  
    pos_con, quat_con = compute_z_constrained_finger_transform(p1a, p1b, p2a, p2b)
    
    print(f"\nUnconstrained result:")
    print(f'pos="{pos_unc[0]:.6f} {pos_unc[1]:.6f} {pos_unc[2]:.6f}" quat="{quat_unc[0]:.6f} {quat_unc[1]:.6f} {quat_unc[2]:.6f} {quat_unc[3]:.6f}"')
    
    print(f"\nZ-constrained result:")
    print(f'pos="{pos_con[0]:.6f} {pos_con[1]:.6f} {pos_con[2]:.6f}" quat="{quat_con[0]:.6f} {quat_con[1]:.6f} {quat_con[2]:.6f} {quat_con[3]:.6f}"')
    
    # Show the difference in quaternions
    quat_diff = np.array(quat_unc) - np.array(quat_con)
    print(f"\nQuaternion difference: {quat_diff}")
    print(f"X-rotation eliminated: {quat_diff[1]:.6f}")
    print(f"Y-rotation eliminated: {quat_diff[2]:.6f}")
    
    return pos_con, quat_con

def compute_correct_finger_transform(p1a, p1b, p2a, p2b):
    """Original unconstrained method for comparison"""
    p1a = np.array(p1a, dtype=float)
    p1b = np.array(p1b, dtype=float)
    p2a = np.array(p2a, dtype=float)
    p2b = np.array(p2b, dtype=float)
    
    v1 = p1b - p1a
    v2 = p2b - p2a
    
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Original quaternion computation (can have X/Y components due to noise)
    dot_product = np.dot(v1_norm, v2_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    if dot_product > 0.9999:
        quaternion = [1.0, 0.0, 0.0, 0.0]
    elif dot_product < -0.9999:
        quaternion = [0.0, 0.0, 0.0, 1.0]  # 180° around Z
    else:
        cross_product = np.cross(v1_norm, v2_norm)
        w = 1.0 + dot_product
        quat = np.array([w, cross_product[0], cross_product[1], cross_product[2]])
        quaternion = quat / np.linalg.norm(quat)
    
    # Rotation matrix
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    T_from_a = p2a - rotation_matrix @ p1a
    T_from_b = p2b - rotation_matrix @ p1b
    translation = (T_from_a + T_from_b) / 2
    
    return translation.tolist(), quaternion.tolist()

def test_middle_finger():
    """Test with your middle finger data"""
    # finger1_base = [-0.06945, 0.04928, 0.00731]
    # finger1_tip = [-0.1135, 0.04599, 0.00464]
    
    # index
    # finger2_base = [-0.06549, 0.10719, 0.00731]
    # finger2_tip = [-0.10198, 0.13208, 0.00464]
      
    # middle finger
    # finger2_base = [-0.00782, 0.11965, 0.00731]
    # finger2_tip = [-0.0099, 0.16377, 0.00464]
    
    # ring
    # finger2_base = [0.03123, 0.11978, 0.00731]
    # finger2_tip = [0.03128, 0.16395, 0.00464]
    
    # pinky
    # finger2_base = [0.07536, 0.10049, 0.00731]
    # finger2_tip = [0.09608, 0.1395, 0.00464]
    
    
    # BELOW IS FOR THE LEGO HAND
    
    finger1_base = [0.03662, 0.05775, 0.00872]
    finger1_tip = [0.03519, 0.12336, 0.00592]
    
    # finger2_base = [0.08547, 0.0546, 0.00872]
    # finger2_tip = [0.11903, 0.111, 0.00592]
    
    # finger2_base = [0.10521, 0.00376, 0.00872]
    # finger2_tip = [0.16967, 0.01611, 0.00592]
    
    # finger2_base = [0.09439, -0.04397, 0.00872]
    # finger2_tip = [0.15681, -0.06426, 0.00592]
    
    finger2_base = [0.054, -0.07495, 0.00872]
    finger2_tip = [0.08588, -0.13232, 0.00592]
    
    # BELOW IS FOR THE LATEST LEGO HAND

    finger1_base = [0.01946, 0.03744, 0.0]
    finger1_tip = [0.01838, 0.06902, 0.0]
    
    finger2_base = [61.43 / 1000, 46.26 / 1000, 0.0]
    finger2_tip = [77.26 / 1000, 73.61 / 1000, 0.0]
    
    # finger2_base = [81.79 / 1000, 18.16 / 1000, 0.0]
    # finger2_tip = [112.75 / 1000, 24.49 / 1000, 0.0]
    
    # finger2_base = [81.39 / 1000, -20.06 / 1000, 0.0]
    # finger2_tip = [111.56 / 1000, -29.46 / 1000, 0.0]
    
    # finger2_base = [59.58 / 1000, -47.58 / 1000, 0.0]
    # finger2_tip = [75.27 / 1000, -75.01 / 1000, 0.0]

    pos, quat = compute_unconstrained_vs_constrained(
        finger1_base, finger1_tip, finger2_base, finger2_tip
    )
    
    print(f"\n=== RECOMMENDED FOR MUJOCO ===")
    print(f'<frame pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">')
    print(f'    <attach model="finger_model" body="finger_root" prefix="f3_"/>')
    print(f'</frame>')
    
    return pos, quat

if __name__ == "__main__":
    test_middle_finger()
    
# index
# finger2_base = [-0.06549, 0.10719, 0.00731]
# finger2_tip = [-0.10198, 0.13208, 0.00464]
    
    