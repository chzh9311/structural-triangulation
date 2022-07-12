"""
The configuration file. You can config the file paths.
"""
h36m_config = {
    # pickle file given by 2D backbones, including the following contents:
    # keypoints_2d: a list of numpy.ndarray of size batch_size x n_cameras x n_joints x 2. The 2D keypoint estimations.
    # keypoints_3d_gt: a list of numpy.ndarray of size batch_size x n_joints x 3. The 3D ground truth.
    # confidences: a list of numpy.ndarray of batch_size x n_cam x 17, The 2D confidences of each joint.
    # Projections: a list of numpy.ndarray of size batch_size x n_cameras x 3 x 4. The camera projection matrices.
    # action_idx: a vector of dimension n_frames, indicating the action indices. 
    # subject_idx: a vector of dimension n_frames, indicating the subject indices. 
    "output file path": "../detected_data/h36m_full.pkl",

    # The file containing the ground truth labels and bboxes.
    # For human3.6M tests, the file is generated following this guide:
    # https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md
    "label path": "../human36m-multiview-labels-GTbboxes.npy",

    # The bone lengths of test subjects.
    "S9 bone lengths path": "../bone_lengths/h36m/S9_bl_gt.npy",
    "S11 bone lengths path": "../bone_lengths/h36m/S11_bl_gt.npy",
    
    # The correspondence between indices of joints in our test and that of MPII model.
    "order": [6] + list(range(1, 6)) + [0] + list(range(7, 17)),
    "actions": ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
                'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
                'TakingPhoto', 'Waiting', 'Walking', 'WalkingDog', 'WalkingTogether']
}

ttc_config = {
    "output file path": "detected_data/ttc.pkl",
    "label path": "totalcapture-multiview-labels-GTbboxes.npy",
    "S1 bone lengths path": "bone_lengths/S1/bl_gt.npy",
    "S2 bone lengths path": "bone_lengths/S2/bl_gt.npy",
    "S3 bone lengths path": "bone_lengths/S3/bl_gt.npy",
    "S4 bone lengths path": "bone_lengths/S4/bl_gt.npy",
    "S5 bone lengths path": "bone_lengths/S5/bl_gt.npy",
    "order": [6] + list(range(1, 6)) + [0] + list(range(7, 16)),
    "actions": ["Walking", "Freestyle", "Acting"]
}