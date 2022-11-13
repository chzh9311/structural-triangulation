import os
import pickle
import numpy as np
from scipy.linalg import block_diag as block_diag

from structural_triangulation import get_inner_mat, create_human_tree
from config import get_config

def bone_length_esti_batch(pts_2d, confs, human_tree, Projections, ORDER, samples=None):
    """
    Estimate the bone lengths by taking average over all symmetric bones
    amoung all frames.
    pts_2d: list of n_batches x n_cams x N_joints x 2
    confs:  list of n_batches x n_cams x N_joints
    human_tree: <DictTree> object, the tree representing human structure
    Projections: list of n_batches x n_cams x 3 x 4
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
        batch_idx = int(f/batch_size)
        inner_idx = int(f - batch_size * batch_idx)
        poses_2d = pts_2d[batch_idx][inner_idx, :, :, :][:, ORDER, :]
        n_cams = pts_2d[batch_idx].shape[1]
        P = np.zeros((Nj * n_cams * 3, Nj * n_cams * 3))
        for i in range(Nj):
            for j in range(n_cams):
                P[3*(i*n_cams + j):3*(i*n_cams + j)+3, 3*(i*n_cams + j):3*(i*n_cams + j)+3]\
                    = confs[batch_idx][inner_idx, j, i] * get_inner_mat(poses_2d[j, i, 0], poses_2d[j, i, 1])

        G = human_tree.conv_B2J

        tmp = []
        for j in range(n_cams):
            tmp.append(Projections[batch_idx][inner_idx, j, :, 0:3])
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
            tmp.append(Projections[batch_idx][inner_idx, j, :, 3:4])
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


def get_bl(output_dir):
    config = get_config("config.yaml")
    pkl_file = open(config.output_file_path, 'rb')
    data = pickle.load(pkl_file)
    for sub_idx in [5, 6]:
        keypoints_2d = data["keypoints_2d"] # list of batch_size x n_cam x n_joints x 2, length is n_frame
        Ps = data["Projections"] # list of batch_size x n_cam x 3 x 4, len = n_frame
        confidences = data["confidences"] # list of batch_size x n_cam x n_joints, len = n_frame
        human_tree = create_human_tree("human36m")

        action_idx = data['action_idx'].flatten()

        # T poses are the first frames of each action
        Tposes = [i for i in range(0, len(action_idx)) if (i == 0 or action_idx[i] != action_idx[i-1]) and (action_idx[i] % 2 == 1) and (data["subject_idx"][i] == sub_idx)]
        bl_mean = bone_length_esti_batch(keypoints_2d, confidences, human_tree, Ps, config["order"], Tposes)
        np.save(os.path.join(output_dir, f"S{9 if sub_idx == 5 else 11}_bl_mean_w_conf.npy"), bl_mean)