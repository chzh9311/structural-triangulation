import time
import numpy as np
import pickle
from tqdm import tqdm
from config import get_config
from easydict import EasyDict as edict

from utils import linear_eigen_method_pose, data_iterator
from structural_triangulation import Pose3D_inference, create_human_tree


def test():
    config = get_config("config.yaml")

    ORDER = np.arange(config.data.n_joints)
    for i, j in config.data.flip_pairs:
        ORDER[[i, j]] = ORDER[[j, i]]

    bl_S9 = np.load(config.file_paths.bl_S9).reshape(config.data.n_joints-1, 1)
    bl_S11 = np.load(config.file_paths.bl_S11).reshape(config.data.n_joints-1, 1)
    with open(config.file_paths.detected_data, 'rb') as pkl_file:
        detected = edict(pickle.load(pkl_file))

    n_frames = detected.keypoints_2d.shape[0]
    human_tree = create_human_tree()
    gt = detected.keypoints_3d_gt[:, ORDER, :]
    gt_relative = gt - gt[:, 0:1, :]

    # Initialization
    linear_X = np.zeros((n_frames, config.data.n_joints, 3))
    ST_X = np.zeros((n_frames, config.data.n_joints, 3))
    start = time.time()

    # Baseline method: linear eigen method
    for idx, n_cams, kps, P, confs in tqdm(data_iterator(ORDER, n_frames,
            detected.keypoints_2d, detected.proj_mats, detected.confidences)):
        linear_X[idx, ...] = linear_eigen_method_pose(n_cams, kps, P, confs)
    print((time.time() - start) / n_frames)

    start = time.time()
    # Structural Triangulation
    for idx, n_cams, kps, P, confs in tqdm(data_iterator(ORDER, n_frames,
            detected.keypoints_2d, detected.proj_mats, detected.confidences)):
        ST_X[idx, ...] = Pose3D_inference(n_cams, human_tree, kps, confs,
                bl_S9 if detected.subject_idx[idx] == 5 else bl_S11, P, config.test.method, config.test.SCA_step)
    
    print((time.time() - start) / n_frames)
    linear_X_relative = linear_X - linear_X[:, 0:1, :]
    ST_X_relative = ST_X - ST_X[:, 0:1, :]

    # On bone lengths:
    bl_linear = human_tree.get_bl_mat(linear_X)
    bl_linear = {"S9": bl_linear[detected.subject_idx == 5, :], "S11":bl_linear[detected.subject_idx == 6, :]}

    bl_st = human_tree.get_bl_mat(ST_X)
    bl_st = {"S9": bl_st[detected.subject_idx == 5, :], "S11": bl_st[detected.subject_idx == 6, :]}
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
    """)

    for s in ["S9", "S11"]:
        print(f"""
            {s}:
            Baseline:
                mean: {np.mean(np.abs(bl_linear[s] - bl_S9.T)):.2f}
                std: {np.sqrt(np.mean(np.var(bl_linear[s], axis=0))):.2f}
                pib: {np.sum((bl_linear[s].T > pib_th[0] * bl_S9)*(bl_linear[s].T < pib_th[1] * bl_S9)) / (bl_linear[s].shape[0] * 16) * 100:.2f}%
            ST:
                mean: {np.mean(np.abs(bl_st[s] - bl_S9.T)):.2f}
                std: {np.sqrt(np.mean(np.var(bl_st[s], axis=0))):.2f}
                pib: {np.sum((bl_st[s].T > pib_th[0] * bl_S9)*(bl_st[s].T < pib_th[1] * bl_S9)) / (bl_st[s].shape[0] * 16) * 100:.2f}%
            """
        )
    for i in range(len(config.data.actions)):
        print(f""" MPJPE-abs of action {config.data.actions[i]}
        Baseline:{np.mean(np.mean(np.linalg.norm(linear_X - gt, axis=2)[np.logical_or(detected.action_idx == 2*i, detected.action_idx == 2*i+1).flatten(), :], axis=0)):.2f}
        ST:{np.mean(np.mean(np.linalg.norm(ST_X - gt, axis=2)[np.logical_or(detected.action_idx == 2*i, detected.action_idx == 2*i+1).flatten(), :], axis=0)):.2f}
        """)


if __name__ == "__main__":
    test()
