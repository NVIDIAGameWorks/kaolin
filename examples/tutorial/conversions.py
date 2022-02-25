import torch

# EULER input to various targets

# functions are all modified from pyrr to use torch Tensor
#   and preserve gradient functions
def euler_to_matrix33(pitch, roll, yaw) -> torch.Tensor:
    sP = torch.sin(pitch)
    cP = torch.cos(pitch)
    sR = torch.sin(roll)
    cR = torch.cos(roll)
    sY = torch.sin(yaw)
    cY = torch.cos(yaw)

    m1 = torch.stack([cY * cP, -cY * sP * cR + sY * sR, cY * sP * sR + sY * cR])
    m2 = torch.stack([sP, cP * cR, -cP * sR])
    m3 = torch.stack([-sY * cP, sY * sP * cR + cY * sR, -sY * sP * sR + cY * cR])
    mat33 = torch.stack([m1, m2, m3]).cuda()
    return mat33


def euler_to_matrix44(pitch, roll, yaw) -> torch.Tensor:
    mat33 = euler_to_matrix33(pitch, roll, yaw)
    return _pad_mat33_to_mat44(mat33)


def _pad_mat33_to_mat44(mat33: torch.Tensor) -> torch.Tensor:
    col = torch.zeros(size=(3, 1), device=mat33.device)
    row = torch.tensor([[0, 0, 0, 1]], device=mat33.device)
    mat44 = torch.vstack([torch.hstack([mat33, col]), row])
    return mat44


def euler_to_quaternion(pitch, roll, yaw) -> torch.Tensor:
    # TODO: avoid moving data to cuda by hand?
    halfRoll = roll * 0.5
    sR = torch.sin(halfRoll)
    cR = torch.cos(halfRoll)

    halfPitch = pitch * 0.5
    sP = torch.sin(halfPitch)
    cP = torch.cos(halfPitch)

    halfYaw = yaw * 0.5
    sY = torch.sin(halfYaw)
    cY = torch.cos(halfYaw)

    q = torch.stack(
        [
            (sR * cP * cY) + (cR * sP * sY),
            (cR * sP * cY) - (sR * cP * sY),
            (cR * cP * sY) + (sR * sP * cY),
            (cR * cP * cY) - (sR * sP * sY),
        ]
    ).cuda()
    return q


# QUATERNION


def vector_normalize(vec: torch.Tensor) -> torch.Tensor:
    """normalizes an Nd list of vectors or a single vector
    to unit length.

    The vector is **not** changed in place.

    For zero-length vectors, the result will be np.nan.

    :param torch.tensor vec: an Nd array with the final dimension
        being vectors
        ::

            torch.tensor([ x, y, z ])

        Or an NxM array::

            torch.tensor([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).

    :rtype: torch.Tensor
    :return: The normalized vector/s
    """
    # calculate the length
    # this is a duplicate of length(vec) because we
    # always want an array, even a 0-d array.
    return (vec.T / torch.sqrt(torch.sum(vec ** 2, axis=-1))).T


def quaternion_to_matrix33(quat: torch.Tensor) -> torch.Tensor:
    q = vector_normalize(quat)

    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    sqw = qw ** 2
    sqx = qx ** 2
    sqy = qy ** 2
    sqz = qz ** 2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = (sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs
    r0 = torch.stack([m00, m01, m02])
    r1 = torch.stack([m10, m11, m12])
    r2 = torch.stack([m20, m21, m22])
    mat33 = torch.stack([r0, r1, r2]).T
    return mat33


def quaternion_to_matrix44(quat: torch.Tensor) -> torch.Tensor:
    mat33 = quaternion_to_matrix33(quat)
    return _pad_mat33_to_mat44(mat33)


def quaternion_to_matrix44_v2(quat: torch.Tensor) -> torch.Tensor:
    r0 = torch.stack(
        [
            1.0 - 2.0 * quat[1] ** 2 - 2.0 * quat[2] ** 2,
            2.0 * quat[0] * quat[1] - 2.0 * quat[2] * quat[3],
            2.0 * quat[0] * quat[2] + 2.0 * quat[1] * quat[3],
        ]
    )
    r1 = torch.stack(
        [
            2.0 * quat[0] * quat[1] + 2.0 * quat[2] * quat[3],
            1.0 - 2.0 * quat[0] ** 2 - 2.0 * quat[2] ** 2,
            2.0 * quat[1] * quat[2] - 2.0 * quat[0] * quat[3],
        ]
    )
    r2 = torch.stack(
        [
            2.0 * quat[0] * quat[2] - 2.0 * quat[1] * quat[3],
            2.0 * quat[1] * quat[2] + 2.0 * quat[0] * quat[3],
            1.0 - 2.0 * quat[0] ** 2 - 2.0 * quat[1] ** 2,
        ]
    )
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]]).cuda()], dim=1)  # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]]).cuda()], dim=0)  # Pad bottom row.
    return rr


# MATRIX


def matrix33_to_quaternion(mat33: torch.Tensor) -> torch.Tensor:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    mat = mat33.T
    trace = mat[0][0] + mat[1][1] + mat[2][2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        qx = (mat[2][1] - mat[1][2]) * s
        qy = (mat[0][2] - mat[2][0]) * s
        qz = (mat[1][0] - mat[0][1]) * s
        qw = 0.25 / s
    elif mat[0][0] > mat[1][1] and mat[0][0] > mat[2][2]:
        s = 2.0 * torch.sqrt(1.0 + mat[0][0] - mat[1][1] - mat[2][2])
        qx = 0.25 * s
        qy = (mat[0][1] + mat[1][0]) / s
        qz = (mat[0][2] + mat[2][0]) / s
        qw = (mat[2][1] - mat[1][2]) / s
    elif mat[1][1] > mat[2][2]:
        s = 2.0 * torch.sqrt(1.0 + mat[1][1] - mat[0][0] - mat[2][2])
        qx = (mat[0][1] + mat[1][0]) / s
        qy = 0.25 * s
        qz = (mat[1][2] + mat[2][1]) / s
        qw = (mat[0][2] - mat[2][0]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + mat[2][2] - mat[0][0] - mat[1][1])
        qx = (mat[0][2] + mat[2][0]) / s
        qy = (mat[1][2] + mat[2][1]) / s
        qz = 0.25 * s
        qw = (mat[1][0] - mat[0][1]) / s

    quat = torch.stack([qx, qy, qz, qw])
    return quat


def translation_to_matrix44(vec) -> torch.Tensor:
    """Creates an identity matrix with the translation set.

    :param numpy.array vec: The translation vector (shape 3 or 4).
    :rtype: numpy.array
    :return: A matrix with shape (4,4) that represents a matrix
        with the translation set to the specified vector.
    """
    mat = torch.eye(4).cuda()
    mat[0][-1] = vec[0]
    mat[1][-1] = vec[1]
    mat[2][-1] = vec[2]
    return mat


def scale_to_matrix44(scale) -> torch.Tensor:
    mat = torch.eye(4).cuda()
    mat[0][0] *= scale[0]
    mat[1][1] *= scale[1]
    mat[2][2] *= scale[2]
    return mat
