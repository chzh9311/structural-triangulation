import os
import numpy as np
from numpy.linalg import norm


def linear_eigen_method_pose(n_cams, Xs, Ps, confidences=None):
    """
    linear eigen triangulation method for the whole human pose.
    :n_cams:      <int> the number of cameras
    :Xs:          <numpy.ndarray> of n_camera x n_joint x 2. The 2D pose estimations.
    :Ps:          <numpy.ndarray> of n_camera x 3 x 4. The camera reprojection matrices.
    :confidences: <numpy.ndarray> of n_camera x n_joint. The confidences for each
        joint on each view.
    return:       <numpy.ndarray> of n_joint x 3. The 3D joint position estimations.
    """
    Nj = Xs.shape[1]
    linear_X = np.zeros((Nj, 3))
    for i in range(Nj):
        linear_X[i, :] = linear_eigen_method(n_cams, Xs[:, i, :],
            Ps, confidences[:, i]).reshape(3,)
    return linear_X


def linear_eigen_method(n_cams, Xs, Ps, confidences=None):
    """
    linear eigen triangulation method for the whole human pose.
    :n_cams:      <int> the number of cameras
    :Xs:          <numpy.ndarray> of n_camera x 2. The 2D pose estimations.
    :Ps:          <numpy.ndarray> of n_camera x 3 x 4. The camera reprojection matrices.
    :confidences: <numpy.ndarray> of n_camera. The confidences for each view.
    return:       <numpy.ndarray> of 3. The 3D joint position estimations.
    """
    A_rows = []
    if confidences is None:
        confidences = np.ones(n_cams)
    for i in range(n_cams):
        conf = confidences[i]
        A_rows += [(Xs[i, 0] * Ps[i, 2:3, :] - Ps[i, 0:1, :])*conf, (Xs[i, 1] * Ps[i, 2:3, :] - Ps[i, 1:2, :])*conf]

    A = np.concatenate(tuple(A_rows), 0)
    ev, em = np.linalg.eig(A.T @ A)
    sln = em[:, np.argmin(ev):np.argmin(ev)+1]
    sln = sln[0:3, :] / sln[3]
    return sln


def linear_LS_method_pose(n_cams, Xs, Ps, confidences):
    """
    linear Least-Squares triangulation method for the whole human pose.
    :n_cams:      <int> the number of cameras
    :Xs:          <numpy.ndarray> of n_camera x n_joint x 2. The 2D pose estimations.
    :Ps:          <numpy.ndarray> of n_camera x 3 x 4. The camera reprojection matrices.
    :confidences: <numpy.ndarray> of n_camera x n_joint. The confidences for each
        joint on each view.
    return:       <numpy.ndarray> of n_joint x 3. The 3D joint position estimations.
    """
    Nj = Xs.shape[1]
    linear_X = np.zeros((Nj, 3))
    for i in range(Nj):
        linear_X[i, :] = linear_LS_method(n_cams, Xs[:, i, :], Ps, confidences[:, i]).reshape(3,)
    return linear_X


def linear_LS_method(n_cams, Xs, Ps, confidences):
    """
    linear Least-Squares triangulation method for the whole human pose.
    :n_cams:      <int> the number of cameras
    :Xs:          <numpy.ndarray> of n_camera x 2. The 2D pose estimations.
    :Ps:          <numpy.ndarray> of n_camera x 3 x 4. The camera reprojection matrices.
    :confidences: <numpy.ndarray> of n_camera. The confidences for each view.
    return:       <numpy.ndarray> of 3. The 3D joint position estimations.
    """
    A_rows = []
    Bs = []
    for i in range(n_cams):
        conf = confidences[i]
        A_rows += [(Xs[i, 0] * Ps[i, 2:3, 0:3] - Ps[i, 0:1, 0:3])*conf, (Xs[i, 1] * Ps[i, 2:3, 0:3] - Ps[i, 1:2, 0:3])*conf]
        Bs += [(Xs[i, 0] * Ps[i, 2, 3] - Ps[i, 0, 3])*conf, (Xs[i, 1] * Ps[i, 2, 3] - Ps[i, 1, 3])*conf]
    A = np.concatenate(tuple(A_rows), 0)

    B = -np.array(Bs).reshape(-1, 1)
    # sln = np.linalg.solve(A.T @ A, A.T @ B)
    sln = np.linalg.inv(A.T @ A) @ A.T @ B
    # p = SII(A.T @ A, np.array([[0], [0], [1]]))
    return sln


def vec2rot(vec1, vec2):
    """
    :vec1, vec2: <numpy.ndarray> of 3, the input vectors.
    return: <numpy.ndarray> of 3 x 3, the rotation matrix to rotate the axes
        system to make the local direction change from vec1 to vec2.
    """
    vec1 = np.reshape(vec1, (3,))
    vec2 = np.reshape(vec2, (3,))
    theta = np.arccos(np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)))
    if theta < 0.01:
        return np.eye(3)
    else:
        n = np.cross(vec2, vec1)
        n = n / norm(n)
        nx = cross_mat(n)
        R12 = np.eye(3) + np.sin(theta) * nx + (1-np.cos(theta)) * nx @ nx
        return R12


def cross_mat(vec):
    """
    The matrix [n]_x of a vector n which makes the cross product n x v equals the
    the matrix-vector multiplication [n]_x v
    :vec:    <numpy.ndarray> of 3, the vector n.
    :return: <numpy.ndarray> of 3 x 3, the matrix [n]_x.
    """
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])


def repr_mat(pt_2d, P):
    """
    One step in Structural Triangulation
    :pt_2d: <numpy.ndarray> of 2 x 1, 2D re-projection point.
    :P:     <numpy.ndarray> of projection matrix.
    :return: R'K'MKR
    """
    KR = P[:, 0:3]
    u = pt_2d[0, 0]
    v = pt_2d[1, 0]
    M = np.array([[1, 0, -u],
                  [0, 1, -v],
                  [-u, -v, u**2+v**2]])
    result = KR.T @ M @ KR
    return result


def repr_err(pts_2d, P1, P2, pts_3d):
    """
    Calculate the average re-projection error
    pts_2d: 2 x N x 2
    P1, P2: 3 x 4
    pts_3d: N x 3
    """
    N = pts_3d.shape[0]
    repr = np.zeros((2, N, 2))
    P = [P1, P2]
    homo_3d = np.concatenate((pts_3d, np.ones((N, 1))), axis=1).T
    for i in range(2):
        homo_2d = P[i] @ homo_3d
        flat_2d = homo_2d[0:2, :] / homo_2d[2, :]
        repr[i, :, :] = flat_2d.T

    return np.mean(np.mean(norm(repr - pts_2d, axis=2)))


def batch_data_iterator(ORDER, n_frames, batch_size, kps, Ps, confs):
    """
    Iterate over batches of data.
    ORDER: the list of indices where 0 represents the root joint.
    n_frames: <int> frame number
    batch_size: <int> 
    kps: list of <numpy.ndarray> of batch_size x n_cams x n_joints x n_dims: key points
    Ps:  list of <numpy.ndarray> of batch_size x n_cams x 3 x 4.
    confs: list of <numpy.ndarray> of n_cameras x n_joints.
    """
    for i in range(n_frames):
        batch_idx = int(i / batch_size)
        inner_idx = i - batch_size * batch_idx
        n_cams = kps[batch_idx].shape[1]
        yield i, n_cams, kps[batch_idx][inner_idx, ...][:, ORDER, :],\
            Ps[batch_idx][inner_idx, :, :, :], confs[batch_idx][inner_idx, ...][:, ORDER]


def draw_vec_pose(ax, mid_points, vec3D, color):
    """
    ax: <matplotlib.Axes> the ax to draw pose on.
    mid_poins: <numpy.ndarray> of bone number x 3
    vec3D: <numpy.ndarray> of bone number x 3
    color: <string>: a specific color for all bones or "auto"
    """
    for i in range(mid_points.shape[0]):
        p1 = mid_points[i, :] - vec3D[i, :] / 2
        p2 = mid_points[i, :] + vec3D[i, :] / 2
        ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))
    ax.view_init(elev=0, azim=0)