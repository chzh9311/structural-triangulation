import os
import time
import numpy as np
import pickle
from tqdm import tqdm
from config import get_config

from utils import linear_eigen_method_pose, batch_data_iterator
from structural_triangulation import Pose3D_inference, create_human_tree


def test():
    # TODO: specify configurations in terminal.
    config = get_config("config.yaml")

    ORDER = np.arange(config.n_joints)
    for i, j in config.flip_pairs:
        ORDER[[i, j]] = ORDER[[j, i]]

    bl_S9 = np.load(config.S9_bl_path)
    bl_S11 = np.load(config.S11_bl_path)
    pkl_file = open(config.output_file_path, 'rb')
    data = pickle.load(pkl_file)

    # list of batch_size x n_cam x n_joints x 2, length is n_frame
    keypoints_2d = data["keypoints_2d"]
    subject_idx = data["subject_idx"]

    # list of batch_size x n_cam x 3 x 4, len = n_frame
    Ps = data["Projections"]
    # list of batch_size x n_cam x n_joints, len = n_frame
    confidences = data["confidences"]
    n_batches = len(keypoints_2d)
    batch_size = keypoints_2d[0].shape[0]
    n_frames = sum([keypoints_2d[n].shape[0] for n in range(n_batches)])
    human_tree = create_human_tree()
    gt = data["keypoints_3d_gt"][:, ORDER, :]
    gt_relative = gt - gt[:, 0:1, :]

    # Initialization
    linear_X = np.zeros((n_frames, config.n_joints, 3))
    ST_X = np.zeros((n_frames, config.n_joints, 3))
    start = time.time()

    # Baseline method: linear eigen method
    for idx, n_cams, kps, P, confs in tqdm(batch_data_iterator(ORDER, n_frames,
            batch_size, keypoints_2d, Ps, confidences)):
        linear_X[idx, ...] = linear_eigen_method_pose(n_cams, kps, P, confs)
    print((time.time() - start) / n_frames)

    start = time.time()
    # Structural Triangulation
    for idx, n_cams, kps, P, confs in tqdm(batch_data_iterator(ORDER, n_frames,
            batch_size, keypoints_2d, Ps, confidences)):
        ST_X[idx, ...] = Pose3D_inference(n_cams, human_tree, kps, confs,
                bl_S9 if subject_idx[idx] == 5 else bl_S11, P, "st", config.SCA_step)
    
    print((time.time() - start) / n_frames)
    linear_X_relative = linear_X - linear_X[:, 0:1, :]
    ST_X_relative = ST_X - ST_X[:, 0:1, :]

    # On bone lengths:
    bv_linear = (human_tree.conv_J2B @ linear_X.reshape(n_frames, -1).T)[3:, :].T.reshape(n_frames, 16, 3)
    bl_linear = np.linalg.norm(bv_linear, axis=2)
    bl_linear = [bl_linear[subject_idx == 5], bl_linear[subject_idx == 6]]

    bv_st = (human_tree.conv_J2B @ ST_X.reshape(n_frames, -1).T)[3:, :].T.reshape(n_frames, 16, 3)
    bl_st = np.linalg.norm(bv_st, axis=2)
    bl_st = [bl_st[subject_idx == 5], bl_st[subject_idx == 6]]
    pib_th = [0.8, 1.2]

    # Show results
    print(f"""
    Total Absolute MPJPE:
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X - gt, axis=2), axis=0))}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X - gt, axis=2), axis=0))}
    Relative MPJPE:
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X_relative - gt_relative, axis=2), axis=0))}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X_relative - gt_relative, axis=2), axis=0))}
    Bone length criteria:
    S9:
        Baseline:
            mean: {np.mean(np.abs(bl_linear[0] - bl_S9.T)):.2f}
            std: {np.sqrt(np.mean(np.var(bl_linear[0], axis=0))):.2f}
            pib: {np.sum((bl_linear[0].T > pib_th[0] * bl_S9)*(bl_linear[0].T < pib_th[1] * bl_S9)) / (bl_linear[0].shape[0] * 16) * 100:.2f}%
        ST:
            mean: {np.mean(np.abs(bl_st[0] - bl_S9.T)):.2f}
            std: {np.sqrt(np.mean(np.var(bl_st[0], axis=0))):.2f}
            pib: {np.sum((bl_st[0].T > pib_th[0] * bl_S9)*(bl_st[0].T < pib_th[1] * bl_S9)) / (bl_st[0].shape[0] * 16) * 100:.2f}%
    S11:
        Baseline:
            mean: {np.mean(np.abs(bl_linear[1] - bl_S11.T)):.2f}
            std: {np.sqrt(np.mean(np.var(bl_linear[1], axis=0))):.2f}
            pib: {np.sum((bl_linear[1].T > pib_th[0] * bl_S9)*(bl_linear[1].T < pib_th[1] * bl_S9)) / (bl_linear[1].shape[0] * 16) * 100:.2f}%
        ST:
            mean: {np.mean(np.abs(bl_st[1] - bl_S11.T)):.2f}
            std: {np.sqrt(np.mean(np.var(bl_st[1], axis=0))):.2f}
            pib: {np.sum((bl_st[1].T > pib_th[0] * bl_S9)*(bl_st[1].T < pib_th[1] * bl_S9)) / (bl_st[1].shape[0] * 16) * 100:.2f}%
""")

    for i in range(len(config.actions)):
        print(f""" MPJPE-re of action {config.actions[i]}
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X_relative - gt_relative, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X_relative - gt_relative, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        """)
    for i in range(len(config.actions)):
        print(f""" MPJPE-abs of action {config.actions[i]}
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X - gt, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X - gt, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        """)


if __name__ == "__main__":
    test()
