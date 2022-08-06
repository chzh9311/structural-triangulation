import numpy as np
import pandas as pd
from math import pi, sin, cos
from tqdm import tqdm
from structural_triangulation import create_human_tree, Pose3D_inference
from utils import linear_eigen_method
from config import config

ORDER = config["order"]
def main():
    test_subjects = [9, 11]
    n_cams_list = np.arange(2, 11)
    sigma_list = np.arange(2, 22, 2)
    tri_result = pd.DataFrame(index=sigma_list, columns=n_cams_list)
    opt_result = pd.DataFrame(index=sigma_list, columns=n_cams_list)
    outperform_rate = pd.DataFrame(index=sigma_list, columns=n_cams_list)
    tri_X = np.zeros((1253 + 928, 17, 3))
    optim_X = np.zeros((1253 + 928, 17, 3))
    lengths = {}
    labels_path=config["label path"]
    labels = np.load(labels_path, allow_pickle=True).item()
    human_tree = create_human_tree()
    poses3D = []
    np.random.seed(0)
    for k in range(len(test_subjects)):
        poses3D.append(labels['table']['keypoints'][labels['table']['subject_idx'] == labels['subject_names'].index(f"S{test_subjects[k]}")])
        poses3D[k] = poses3D[k][:, ORDER, :]
        lengths[test_subjects[k]] = np.zeros((16, 1))
        for i in range(1, 17):
            lengths[test_subjects[k]][i - 1] = np.mean(np.linalg.norm(poses3D[k][:, i, :] - poses3D[k][:, human_tree.node_list[i]["parent"], :], axis=1), axis=0)
    poses3D = np.concatenate(tuple(poses3D), axis=0)
    for cam_type in ["round", "half"]:
        for n_cams in tqdm(n_cams_list):
            P_list = generate_cam_systems(n_cams, pi if cam_type == "half" else 2*pi)
            Nf, Nj, _ = poses3D.shape
            poses2D = np.zeros((Nf, n_cams, Nj, 2))
            for i in range(Nf):
                X3d = poses3D[i, ...]
                for c in range(n_cams):
                    X2d = P_list[c] @ np.concatenate((X3d, np.ones((Nj, 1))), axis=1).T
                    X2d = (X2d[0:2, :] / X2d[2, :]).T
                    poses2D[i, c, :, :] = X2d
            for sigma in sigma_list:
                estim2D = poses2D + sigma*np.random.randn(*poses2D.shape)
                for i in range(Nf):
                    for j in range(17):
                        tri_X[i, j, :] = linear_eigen_method(n_cams, estim2D[i, :, j, :], np.stack(tuple(P_list), axis=0), np.ones((n_cams,))/2).reshape(3,)
                    optim_X[i, ...] = Pose3D_inference(n_cams, human_tree, estim2D[i, ...], np.ones((n_cams, Nj))/n_cams, lengths[9 if i < 1253 else 11], np.stack(tuple(P_list), axis=0), "st", 1 if (n_cams==2 and cam_type=="round") else 3)
                print(MPJPE(tri_X, poses3D))
                print(MPJPE(optim_X, poses3D))
                
                tri_result[n_cams][sigma] = MPJPE(tri_X, poses3D)
                opt_result[n_cams][sigma] = MPJPE(optim_X, poses3D)
                outperform_rate[n_cams][sigma] = OUT_rate(tri_X, optim_X, poses3D)

        tri_result.to_csv(f"vir_result/{cam_type}_triangulation.csv", ",")
        opt_result.to_csv(f"vir_result/{cam_type}_optimization.csv", ",")
        outperform_rate.to_csv(f"vir_result/{cam_type}_outperform_rate.csv", ",")

def MPJPE(pose, gt):
    """
    pose, gt: <numpy.ndarray> of n_frames x n_joints x n_dim, referring to the
    estimated 3D pose and ground truth.
    """
    return np.mean(np.mean(np.linalg.norm(pose - gt, axis=2), axis=1))

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