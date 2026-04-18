import numpy as np

def unit_vector_between_mm(p1_mm, p2_mm, eps=1e-12):
    """
    Unit vector from p1 -> p2 expressed in the world/origin frame.

    Parameters
    ----------
    p1_mm, p2_mm : array-like length 3
        Coordinates in millimeters.
    eps : float
        Small threshold to guard against zero-length vectors.

    Returns
    -------
    u : np.ndarray shape (3,)
        Unit (dimensionless) vector pointing from p1 to p2.
    """
    p1 = np.asarray(p1_mm, dtype=float)
    p2 = np.asarray(p2_mm, dtype=float)
    v = p2 - p1                # still in mm
    n = np.linalg.norm(v)      # mm
    if n < eps:
        raise ValueError("p1 and p2 are identical or too close; direction undefined.")

    u = v / n
    print("Unit vector between points:", u)
    return u

# # finger 1 palm servo joint axis
# # unit vector: 0 0 1
# # position:  -0.06832 0.03225 0.06643

# # finger 1 base joint
# unit_vector_between_mm([-69.45, 49.28, 7.31], [-66.88, 15.34, 7.31])
# # unit vector: 0.0755057  -0.99714537  0.
# # position: -0.06688 0.001534 0.00731

# # finger 1 middle joint
# unit_vector_between_mm([-113.5, 45.99, 4.64], [-110.93, 12.06, 4.64])
# # unit vector: 0.07552783 -0.99714369 0.
# # position: -0.11093 0.01206 0.00464

# # finger 1 end joint
# unit_vector_between_mm([-154.95, 8.77, 1.54], [-157.52, 42.71, 1.54])
# # unit vector: 0.0755057   -0.99714537  0.
# # position: -0.15752 0.04271 0.00154


# LEGO HAND FIRST ITERATION
# # finger 1 palm servo joint axis
# # unit vector: 0 0 1
# # position:  -0.06832 0.03225 0.06643

# # finger 1 base joint
# unit_vector_between_mm([36.62, 57.75, 8.72], [2.6, 57, 8.72])
# # unit vector: -0.99975708 -0.0220405  0.
# # position: 0.03662 0.05775 0.00872

# # finger 1 middle joint
# unit_vector_between_mm([35.91, 90.56, 7.43], [1.88, 89.82, 7.43])
# # unit vector: -0.99976365 -0.02174038 0.
# # position: 0.03591 0.09056 0.00743

# # finger 1 end joint
# unit_vector_between_mm([35.19, 123.36, 5.92], [1.16, 122.62, 5.92])
# # unit vector: -0.99976365 -0.02174038 0.
# # position: 0.03519 0.12336 0.00592


# LEGO HAND FINAL ITERATION
"""palm: (18.63, 61.72, -2.8), (18.63, 61.72, 26)

base: (35.72, 62.15, -32.64), (1.7, 60.99, -32.64)

middle: (34.57, 93.62, -33.78), (0.55, 92.46, -33.78)

end: (33.53, 125.1, -34.78), (-0.48, 123.94, -34.78)"""

# finger 1 palm servo joint axis
# unit vector: 0 0 1
# position: 0.01863, 0.06172, -0.028

# finger 1 base joint
unit_vector_between_mm([35.72, 62.15, -32.64], [1.7, 60.99, -32.64])
# unit vector: -0.99941918 -0.03407779  0.
# position: 0.03572 0.06215 -0.03264

# finger 1 middle joint
unit_vector_between_mm([34.57, 93.62, -33.78], [0.55, 92.46, -33.78])
# unit vector: -0.99941918 -0.03407779  0.
# position: 0.03457 0.09362 -0.03378

# finger 1 end joint
unit_vector_between_mm([33.53, 125.1, -34.78], [-0.48, 123.94, -34.78])
# unit vector: -0.99941884 -0.03408779  0.
# position: 0.03353 0.1251 -0.03478