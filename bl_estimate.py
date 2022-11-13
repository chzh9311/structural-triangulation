import os
import pickle
import numpy as np
from scipy.linalg import block_diag as block_diag

from structural_triangulation import get_inner_mat, create_human_tree
from config import get_config
from easydict import EasyDict as edict


def bone_length_estimate(pts_2d, confs, human_tree, Projections, ORDER, samples=None):
    """
    Estimate the bone lengths by taking average over all symmetric bones
    amoung all frames.
    pts_2d: n_frames x n_cams x N_joints x 2
    confs:  n_frames x n_cams x N_joints
    human_tree: <DictTree> object, the tree representing human structure
    Projections: n_frames x n_cams x 3 x 4
    ORDER: the index list of all human joints where 0 represents the root.
    samples: the index list of all T-pose frames. If None, take only the first frame.
    """
    batch_size = pts_2d[0].shape[0]
    if samples is None:
        n_samples = 1
        samples = np.arange(n_samples)
    else:
        n_samples = len(samples)

    Nj = pts_2d[0].shape[2]
    E1 = np.zeros((3*Nj-3, Nj-1))
    for i in range(Nj-1):
        E1[3*i:3*i+3, i:i+1] = np.ones((3, 1))

    Nj = human_tree.size

    L_mean = np.zeros((Nj-1, 1))
    for f in samples:
        poses_2d = pts_2d[f, :, :, :][:, ORDER, :]
        n_cams = poses_2d.shape[0]
        P = np.zeros((Nj * n_cams * 3, Nj * n_cams * 3))
        for i in range(Nj):
            for j in range(n_cams):
                P[3*(i*n_cams + j):3*(i*n_cams + j)+3, 3*(i*n_cams + j):3*(i*n_cams + j)+3]\
                    = confs[f, j, i] * get_inner_mat(poses_2d[j, i, 0], poses_2d[j, i, 1])

        G = human_tree.conv_B2J

        tmp = []
        for j in range(n_cams):
            tmp.append(Projections[f, j, :, 0:3])
        KR_diag = [np.concatenate(tmp, axis=0)]*Nj
        KR = block_diag(*KR_diag)
        Lam = 2 * KR.T @ P @ KR
        Irow = np.concatenate((np.eye(3),)*Nj, axis=1)
        Mrow = Irow @ Lam
        TrLam = Mrow @ Irow.T
        Mrow = Mrow[:, 3:]
        TrM_inv = np.linalg.inv(TrLam)

        tmp = []
        for j in range(n_cams):
            tmp.append(Projections[f, j, :, 3:4])
        KRT = -np.concatenate(tuple(tmp)*Nj, axis=0)
        mt = 2 * (KRT.T @ P @ KR).T

        Q = np.concatenate((-TrM_inv @ Mrow @ G[3:, 3:], np.eye(Nj*3-3)), axis=0)
        p = np.concatenate((-TrM_inv @ Irow @ mt, np.zeros((Nj*3-3, 1))), axis=0)

        A = Q.T @ G.T @ Lam @ G @ Q
        e = (p.T @ G.T @ Lam @ G @ Q + mt.T @ G @ Q).T
        b0 = np.linalg.inv(A) @ e
        L0 = np.linalg.norm(b0.reshape(Nj - 1, 3), axis=1).reshape(-1, 1)
        L_mean += L0

    L_mean /= n_samples
    for i in human_tree.left_bones:
        pair_idx = human_tree.right_bones[human_tree.left_bones.index(i)]
        L_mean[i-1, 0] = L_mean[pair_idx-1, 0] = (L_mean[i-1, 0] + L_mean[pair_idx-1, 0])/2
    return L_mean


def get_bl(cfg_file, output_dir):
    config = get_config(cfg_file)
    with open(config.file_paths.detected_data, 'rb') as f:
        data = edict(pickle.load(f))

    ORDER = np.arange(config.data.n_joints)
    for i, j in config.data.flip_pairs:
        ORDER[[i, j]] = ORDER[[j, i]]
    for sub_idx in [5, 6]:
        mask = data.subject_idx == sub_idx
        keypoints_2d = data.keypoints_2d[mask, ...]  # n_frames x n_cam x n_joints x 2
        projs = data.proj_mats[mask, ...]  # n_frames x n_cam x 3 x 4
        confidences = data.confidences[mask, ...]  # n_frames x n_cam x n_joints
        human_tree = create_human_tree("human36m")

        action_idx = data.action_idx[mask]

        # T poses are the first frames of each action
        Tposes = [i for i in range(0, len(action_idx)) if (i == 0 or action_idx[i] != action_idx[i-1]) and (action_idx[i] % 2 == 0)]
        bl_mean = bone_length_estimate(keypoints_2d, confidences, human_tree, projs, ORDER, Tposes)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, f"S{9 if sub_idx == 5 else 11}_bl_estimated.npy"), bl_mean)

        # Get ground truth bone lengths from data.
        a = human_tree.get_bl_mat(data.keypoints_3d_gt[mask, ...][:, ORDER, :])
        bl_gt = np.mean(human_tree.get_bl_mat(data.keypoints_3d_gt[mask, ...][:, ORDER, :]), axis=0)
        np.save(os.path.join(output_dir, f"S{9 if sub_idx == 5 else 11}_bl_gt.npy"), bl_gt)


if __name__ == "__main__":
    get_bl("config.yaml", os.path.join("data", "bone_lengths", "h36m"))