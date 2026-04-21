import numpy as np

# ---------------------------
# axis helpers
# ---------------------------
def axis_vec(axis: int):
    if axis == 0:
        return np.array([1.0, 0.0, 0.0])
    elif axis == 1:
        return np.array([0.0, 1.0, 0.0])
    elif axis == 2:
        return np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError("Invalid axis")


def get_axes(order: str):
    mapping = {
        "XYZ": [0, 1, 2],
        "XZY": [0, 2, 1],
        "YXZ": [1, 0, 2],
        "YZX": [1, 2, 0],
        "ZXY": [2, 0, 1],
        "ZYX": [2, 1, 0],
    }
    return mapping[order]


# ---------------------------
# rotation basics
# ---------------------------
def angle_axis_to_matrix(angle, axis):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])


# ---------------------------
# euler -> matrix
# ---------------------------
def euler2matrix(angles, order="ZYX"):
    axes = get_axes(order)
    R = np.eye(3)

    for i in range(3):
        axis = axis_vec(axes[i])
        Ri = angle_axis_to_matrix(angles[i], axis)
        R = R @ Ri  # 对应 C++: q = q * ...

    return R


# ---------------------------
# matrix -> euler（完整6种）
# ---------------------------
def matrix2euler(R, order="ZYX"):
    eps = 1e-6
    angles = np.zeros(3)

    if order == "XYZ":
        sy = -R[2, 0]
        if abs(sy) < 1 - eps:
            angles[1] = np.arcsin(sy)
            angles[0] = np.arctan2(R[2, 1], R[2, 2])
            angles[2] = np.arctan2(R[1, 0], R[0, 0])
        else:
            angles[1] = np.arcsin(sy)
            angles[0] = np.arctan2(-R[1, 2], R[1, 1])
            angles[2] = 0

    elif order == "ZYX":
        sy = -R[2, 0]
        if abs(sy) < 1 - eps:
            angles[1] = np.arcsin(sy)
            angles[0] = np.arctan2(R[1, 0], R[0, 0])
            angles[2] = np.arctan2(R[2, 1], R[2, 2])
        else:
            angles[1] = np.arcsin(sy)
            angles[0] = np.arctan2(-R[0, 1], R[1, 1])
            angles[2] = 0

    elif order == "XZY":
        sz = R[1, 0]
        if abs(sz) < 1 - eps:
            angles[2] = np.arcsin(sz)
            angles[0] = np.arctan2(-R[1, 2], R[1, 1])
            angles[1] = np.arctan2(-R[2, 0], R[0, 0])
        else:
            angles[2] = np.arcsin(sz)
            angles[0] = np.arctan2(R[2, 1], R[2, 2])
            angles[1] = 0

    elif order == "YXZ":
        sx = -R[1, 2]
        if abs(sx) < 1 - eps:
            angles[0] = np.arcsin(sx)
            angles[1] = np.arctan2(R[0, 2], R[2, 2])
            angles[2] = np.arctan2(R[1, 0], R[1, 1])
        else:
            angles[0] = np.arcsin(sx)
            angles[1] = np.arctan2(-R[2, 0], R[0, 0])
            angles[2] = 0

    elif order == "YZX":
        sz = -R[0, 1]
        if abs(sz) < 1 - eps:
            angles[2] = np.arcsin(sz)
            angles[1] = np.arctan2(R[0, 2], R[0, 0])
            angles[0] = np.arctan2(R[2, 1], R[1, 1])
        else:
            angles[2] = np.arcsin(sz)
            angles[1] = np.arctan2(-R[2, 0], R[2, 2])
            angles[0] = 0

    elif order == "ZXY":
        sx = R[2, 1]
        if abs(sx) < 1 - eps:
            angles[0] = np.arcsin(sx)
            angles[2] = np.arctan2(-R[0, 1], R[1, 1])
            angles[1] = np.arctan2(-R[2, 0], R[2, 2])
        else:
            angles[0] = np.arcsin(sx)
            angles[2] = np.arctan2(R[1, 0], R[0, 0])
            angles[1] = 0

    else:
        raise ValueError("Unsupported order")

    return angles



if __name__ == "__main__":
    angles = np.array([(107 * np.pi / 180), 0.0, 0.0])

    order = "ZYX"
    R = euler2matrix(angles, order)
    angles_rec = matrix2euler(R, order)

    print(f"\nOrder: {order}")
    print("R =\n", R)
    print("angles_rec =", angles_rec)