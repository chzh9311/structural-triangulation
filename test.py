import os
import time
import pickle
import yaml

import numpy as np
from tqdm import tqdm
from config import get_config
from easydict import EasyDict as edict

from utils import linear_eigen_method_pose, data_iterator
from structural_triangulation import Pose3D_inference, create_human_tree


def test():
    config = get_config(os.path.join("configs", "h36m_config.yaml"))
    exp_name = f"h36m_{config.test.method}_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting experiment {exp_name}...")

    ORDER = np.arange(config.data.n_joints)
    for i, j in config.data.flip_pairs:
        ORDER[[i, j]] = ORDER[[j, i]]

    bl_S9 = np.load(config.file_paths.bl_S9).reshape(config.data.n_joints - 1, 1)
    bl_S11 = np.load(config.file_paths.bl_S11).reshape(config.data.n_joints - 1, 1)
    with open(config.file_paths.detected_data, 'rb') as pkl_file:
        detected = edict(pickle.load(pkl_file))
        if not config.test.with_damaged_actions:
            da_list = []
            for a in config.test.damaged_actions:
                action_name, subaction = a.split("_")
                da_list.append(2 * config.data.actions.index(action_name) + eval(subaction) - 1)
            mask = np.logical_not(np.isin(detected.action_idx, da_list))
            for k in detected.keys():
                detected[k] = detected[k][mask, ...]

    n_frames = detected.keypoints_2d.shape[0]
    human_tree = create_human_tree()
    gt = detected.keypoints_3d_gt[:, ORDER, :]
    gt_relative = gt - gt[:, 0:1, :]

    # Initialization
    Pose3D = np.zeros((n_frames, config.data.n_joints, 3))

    start = time.time()
    for idx, n_cams, kps, P, confs in tqdm(
        data_iterator(ORDER, n_frames, detected.keypoints_2d, detected.proj_mats, detected.confidences),
        desc="Processing frame data...",
        total=n_frames
    ):
        if config.test.method == "Baseline":
            Pose3D[idx, ...] = linear_eigen_method_pose(n_cams, kps, P, confs)
        else:
            Pose3D[idx, ...] = Pose3D_inference(n_cams, human_tree, kps, confs,
                                                bl_S9 if detected.subject_idx[idx] == 5 else bl_S11, P,
                                                config.test.method, config.test.n_steps)

    Pose3D_relative = Pose3D - Pose3D[:, 0:1, :]

    # On bone lengths:
    bl_estimate = human_tree.get_bl_mat(Pose3D)
    bl_estimate = {"S9": bl_estimate[detected.subject_idx == 5, :], "S11": bl_estimate[detected.subject_idx == 6, :]}

    pib_th = [0.8, 1.2]

    # Dump result to json file:
    result = {}
    result["Method"] = config.test.method
    result["n_steps"] = config.test.n_steps
    result["With_damaged_actions"] = config.test.with_damaged_actions
    result["Joint-relative metrics"] = {
        "Absolute MPJPE (mm)": {"Average": np.mean(np.mean(np.linalg.norm(Pose3D - gt, axis=2), axis=0)).item()},
        "Relative MPJPE (mm)": {
            "Average": np.mean(np.mean(np.linalg.norm(Pose3D_relative - gt_relative, axis=2), axis=0)).item()}
    }
    result["Time-relative metrics"] = {
        "Inference time per frame (ms)": (time.time() - start) / n_frames * 1000,
    }
    result["Bone-relative metrics"] = {
        "Mean Per Bone Length Error (MEBLE)": {
            "S9": np.mean(np.abs(bl_estimate["S9"] - bl_S9.T)).item(),
            "S11": np.mean(np.abs(bl_estimate["S11"] - bl_S11.T)).item(),
        },
        "Mean Bone Length Standard deviation (MBLS)": {
            "S9": np.sqrt(np.mean(np.var(bl_estimate["S9"], axis=0))).item(),
            "S11": np.sqrt(np.mean(np.var(bl_estimate["S11"], axis=0))).item()
        },
        "Percentage of Inlier Bones (PIB)": {
            "S9": np.sum((bl_estimate["S9"].T > pib_th[0] * bl_S9) * (bl_estimate["S9"].T < pib_th[1] * bl_S9)).item() \
                  / (bl_estimate["S9"].shape[0] * 16) * 100,
            "S11": np.sum((bl_estimate["S11"].T > pib_th[0] * bl_S11) * (bl_estimate["S11"].T < pib_th[1] * bl_S11)).item() \
                   / (bl_estimate["S11"].shape[0] * 16) * 100
        }
    }

    for i in range(len(config.data.actions)):
        result["Joint-relative metrics"]["Absolute MPJPE (mm)"][config.data.actions[i]] = \
            np.mean(np.mean(np.linalg.norm(Pose3D - gt, axis=2)[
                            np.logical_or(detected.action_idx == 2 * i, detected.action_idx == 2 * i + 1).flatten(), :],
                            axis=0)).item()
        result["Joint-relative metrics"]["Relative MPJPE (mm)"][config.data.actions[i]] = \
            np.mean(np.mean(np.linalg.norm(Pose3D_relative - gt_relative, axis=2)[
                            np.logical_or(detected.action_idx == 2 * i, detected.action_idx == 2 * i + 1).flatten(), :],
                            axis=0)).item()

    if not os.path.exists(os.path.join(config.test.out_dir, exp_name)):
        os.makedirs(os.path.join(config.test.out_dir, exp_name))
    with open(os.path.join(config.test.out_dir, exp_name, "result.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(result, f)
        print(f"Written result to {os.path.join(config.test.out_dir, exp_name, 'result.yaml')}")


if __name__ == "__main__":
    test()
