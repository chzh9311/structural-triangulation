import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from config import config

from utils import linear_eigen_method_pose, batch_data_iterator
from structural_triangulation import Pose3D_inference, create_human_tree
from bl_estimate import bone_length_esti_batch


def test():
# H36M
    ORDER = config["order"]
    ACTIONS = config["actions"]
    bl_S9 = np.load(config["S9 bone lengths path"])
    bl_S11 = np.load(config["S11 bone lengths path"])
    pkl_file = open(config["output file path"], 'rb')
    data = pickle.load(pkl_file)
    n_joints = config["joint number"]
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
    linear_X = np.zeros((n_frames, n_joints, 3))
    ST_X = np.zeros((n_frames, n_joints, 3))
    start = time.time()

    ## Test for segment triangulation
    for idx, n_cams, kps, P, confs in tqdm(batch_data_iterator(ORDER, n_frames,
            batch_size, keypoints_2d, Ps, confidences)):
        linear_X[idx, ...] = linear_eigen_method_pose(n_cams, kps, P, confs)
    print((time.time() - start) / n_frames)

    start = time.time()
    for idx, n_cams, kps, P, confs in tqdm(batch_data_iterator(ORDER, n_frames,
            batch_size, keypoints_2d, Ps, confidences)):
        ST_X[idx, ...] = Pose3D_inference(n_cams, human_tree, kps, confs,
                bl_S9 if subject_idx[idx] == 5 else bl_S11, P, "st", 3)
    
    print((time.time() - start) / n_frames)
    linear_X_relative = linear_X - linear_X[:, 0:1, :]
    ST_X_relative = ST_X - ST_X[:, 0:1, :]

    ### On bone lengths:
    bv_linear = (human_tree.conv_J2B @ linear_X.reshape(n_frames, -1).T)[3:, :].T.reshape(n_frames, 16, 3)
    bl_linear = np.linalg.norm(bv_linear, axis=2)
    bl_linear = [bl_linear[subject_idx == 5], bl_linear[subject_idx == 6]]

    bv_st = (human_tree.conv_J2B @ ST_X.reshape(n_frames, -1).T)[3:, :].T.reshape(n_frames, 16, 3)
    bl_st = np.linalg.norm(bv_st, axis=2)
    bl_st = [bl_st[subject_idx == 5], bl_st[subject_idx == 6]]
    pib_th = [0.8, 1.2]

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

    for i in range(len(ACTIONS)):
        print(f""" MPJPE-re of action {ACTIONS[i]}
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X_relative - gt_relative, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X_relative - gt_relative, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        """)
    for i in range(len(ACTIONS)):
        print(f""" MPJPE-abs of action {ACTIONS[i]}
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X - gt, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X - gt, axis=2)[np.logical_or(data["action_idx"] == 2*i, data["action_idx"] == 2*i+1).flatten(), :], axis=0))}
        """)


def get_bl():
    path = config["output path"]
    for sub_idx in config["subject ids"]:
        pkl_file = open(path, 'rb')
        data = pickle.load(pkl_file)
        keypoints_2d = data["keypoints_2d"] # list of batch_size x n_cam x n_joints x 2, length is n_frame
        Ps = data["Projections"] # list of batch_size x n_cam x 3 x 4, len = n_frame
        confidences = data["confidences"] # list of batch_size x n_cam x n_joints, len = n_frame
        human_tree = create_human_tree("human36m")

        action_idx = data['action_idx'].flatten()
        Tposes = [i for i in range(0, len(action_idx)) if (i == 0 or action_idx[i] != action_idx[i-1]) and (action_idx[i] % 2 == 1) and (data["subject_idx"][i] == sub_idx)]
        bl_mean = bone_length_esti_batch(keypoints_2d, confidences, human_tree, Ps, config["order"], Tposes)
        # np.save(f"bone_lengths/h36m/S{9 if sub_idx == 5 else 11}_bl_mean_w_conf.npy", bl_mean)


if __name__ == "__main__":
    test()
    # get_bl()
