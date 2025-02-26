import os
import munch 

import cv2 as cv
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg') 
# import matplotlib.pyplot as plt

from tools import *

# wandb.init(project="visual_odometry")  

# Frames path
num_seq      = "00"
dir_path     = "/Users/sergio/Documents/kitti/gray_images/sequences/" + num_seq + "/image_0"
path_frames  = sorted(os.listdir(dir_path))
num_frames   = 50 # len(path_frames)


# Read calibration
path_calib_seq = "/Users/sergio/Documents/kitti/gray_images/sequences/" + num_seq + "/calib.txt"
P, K           = read_calib(path_calib_seq)

# Read the first frame
old_frame = cv.imread(os.path.join(dir_path, path_frames[0]), cv.IMREAD_GRAYSCALE)

# Initialize a dictionary to store the cameras 
cameras = dict()

# Load ground truth trajectory
poses_path = "/Users/sergio/Documents/kitti/dataset_poses/poses/" + num_seq + ".txt"
poses_df   = pd.read_csv(poses_path, header=None, sep=' ')
poses      = poses_df.apply(lambda row: read_pose(row.values), axis=1)

for idx, pose in enumerate(poses[:num_frames]):
    R_gt, T_gt = pose
    T_gt = T_gt.flatten()
    cameras[idx] = munch.munchify({"R" : None, "T" : None, "R_gt" : R_gt, "T_gt" : T_gt, "uid" : idx})

# Real-time pose initialization
rt_pose = np.eye(4, dtype=np.float32)
cameras[0].R = rt_pose[:3, :3]
cameras[0].T = rt_pose[:3, 3]

# Create window to display
win_name  = "KITTI Sequence"
traj_name = "Trajectory"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)


for i in range(1, num_frames):

    # Read current frame
    curr_frame = cv.imread(os.path.join(dir_path, path_frames[i]), cv.IMREAD_GRAYSCALE)

    # Get correspondences
    match_dict = get_matches(old_frame, curr_frame)
    pts1, pts2 = match_dict["matches"]
    kps1, kps2 = match_dict["keypoints"]

    # Get relative pose
    R, T     = get_pose(pts1, pts2, K)

    delta       = transf_hom(R, T)
    prev_pose   = transf_hom(cameras[i-1].R, cameras[i-1].T)
    update_pose = delta @ prev_pose

    cameras[i].R = update_pose[:3, :3]
    cameras[i].T = update_pose[:3, 3]

    # Display frames and trajectory
    frame = cv.drawKeypoints(curr_frame, kps1, None, color=(0,255,0), flags=0)
    drawBannerText(frame, f'Frame: {i}, Pose: {T}')
    cv.imshow(win_name, frame)

    # Update old frame
    old_frame = curr_frame.copy()

    # Handle key events
    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()

eval_ate(
    frames=cameras, 
    kf_ids=range(num_frames), 
    save_dir="results_" + num_seq, 
    iterations=0, 
    final=True, 
    monocular=True
)