import os
import pickle
import numpy as np
import pandas as pd
from math import pi, sin, cos
from tqdm import tqdm
from easydict import EasyDict as edict

from structural_triangulation import create_human_tree, Pose3D_inference
from utils import linear_eigen_method_pose, MPJPE
from config import get_config


def main():
    config = get_config(os.path.join("configs", "virtual_config.yaml"))

    ORDER = np.arange(config.data.n_joints)
    for i, j in config.data.flip_pairs:
        ORDER[[i, j]] = ORDER[[j, i]]

    # parameters
    n_cams_array = np.array(config.test.n_cams)
    sigmas_array = np.array(config.test.n_sigmas)

    # take notes of the results
    tri_result = pd.DataFrame(index=sigmas_array, columns=n_cams_array)
    opt_result = pd.DataFrame(index=sigmas_array, columns=n_cams_array)
    outperform_rate = pd.DataFrame(index=sigmas_array, columns=n_cams_array)

    with open(config.file_paths.detected_data, "rb") as f:
        labels = edict(pickle.load(f))
    human_tree = create_human_tree()

    if config.test.seed is not None:
        np.random.seed(config.test.seed)
    poses3D = labels.keypoints_3d_gt
    lengths = {
        5: np.load(config.file_paths.bl_S9).reshape(-1, 1),
        6: np.load(config.file_paths.bl_S11).reshape(-1, 1)
    }
    for i, j in config.data.flip_pairs:
        poses3D[:, [j, i], :] = poses3D[:, [i, j], :]

    for cam_type in ["round", "half"]:
        for n_cams in tqdm(n_cams_array):
            P_list = generate_cam_systems(n_cams, pi if cam_type == "half" else 2*pi)
            n_frames, n_joints, _ = poses3D.shape
            poses2D = np.zeros((n_frames, n_cams, n_joints, 2))
            for i in range(n_frames):
                X3d = poses3D[i, ...]
                for c in range(n_cams):
                    X2d = P_list[c] @ np.concatenate((X3d, np.ones((n_joints, 1))), axis=1).T
                    X2d = (X2d[0:2, :] / X2d[2, :]).T
                    poses2D[i, c, :, :] = X2d
            for sigma in sigmas_array:
                tri_X = []
                optim_X = []
                estim2D = poses2D + sigma*np.random.randn(*poses2D.shape)
                for i in range(n_frames):
                    tri_X.append(linear_eigen_method_pose(n_cams, estim2D[i, ...], np.stack(tuple(P_list), axis=0)))
                    if config.test.method == "ST":
                        n_steps = 1 if (n_cams == 2 and cam_type == "round") else config.test.n_steps
                    else:
                        n_steps = config.test.n_steps
                    optim_X.append(Pose3D_inference(n_cams, human_tree, estim2D[i, ...], None,
                                                    lengths[labels.subject_idx[i]], np.stack(tuple(P_list), axis=0),
                                                    config.test.method, n_steps))
                tri_X = np.stack(tri_X, axis=0)
                optim_X = np.stack(optim_X, axis=0)
                # print(MPJPE(tri_X, poses3D))
                # print(MPJPE(optim_X, poses3D))
                
                tri_result[n_cams][sigma] = MPJPE(tri_X, poses3D)
                opt_result[n_cams][sigma] = MPJPE(optim_X, poses3D)
                outperform_rate[n_cams][sigma] = OUT_rate(tri_X, optim_X, poses3D)

        if not os.path.exists(config.test.output_dir):
            os.makedirs(config.test.output_dir)
        tri_result.to_csv(os.path.join(config.test.output_dir, f"{cam_type}_LEM.csv"), ",")
        opt_result.to_csv(os.path.join(config.test.output_dir, f"{cam_type}_{config.test.method}.csv"), ",")
        outperform_rate.to_csv(os.path.join(config.test.output_dir, f"{cam_type}_outperform_rate.csv"), ",")


def OUT_rate(pose1, pose2, gt):
    """
    pose1, pose2, gt: <numpy.ndarray> of n_frames x n_joints x n_dim, referring
    to the rate that pose2 is better than pose1 in terms of MPJPE.
    """
    mpjpe1 = np.mean(np.linalg.norm(pose1 - gt, axis=2), axis=1)
    mpjpe2 = np.mean(np.linalg.norm(pose2 - gt, axis=2), axis=1)
    return np.sum(mpjpe2 <= mpjpe1) / gt.shape[0]


def generate_cam_systems(n_cams, rot_range):
    """
    Generate the camera system with predefined camera number n_cams and rotation
    range rot_range.
    """
    assert n_cams > 1
    K = np.array([[900, 0.5, 500], [0, 900, 500], [0, 0, 1]])
    T = np.array([[0], [0], [2000]])
    thetas = [i*rot_range/n_cams for i in range(n_cams)]
    Rs = [np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]) for theta in thetas]
    Ps = [K @ np.concatenate((Rs[i], T), axis=1) for i in range(n_cams)]
    return Ps


if __name__ == "__main__":
    main()