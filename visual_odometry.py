from tools import *
import cv2 as cv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Frames path
num_seq      = "00"
dir_path     = "./gray_images/sequences/" + num_seq + "/image_0"
path_frames  = sorted(os.listdir(dir_path))
num_frames   = len(path_frames)

# Read calibration
path_calib_seq = "./gray_images/sequences/" + num_seq + "/calib.txt"
P, K           = read_calib(path_calib_seq)

# Read the first frame
old_frame = cv.imread(os.path.join(dir_path, path_frames[0]), cv.IMREAD_GRAYSCALE)

# Real-time pose initialization
rt_pose = np.eye(4, dtype=np.float32)

# Initialize list to store the estimated trajectory
estimated_trajectory = []

# Load ground truth trajectory
poses_path = "./dataset_poses/poses/00.txt"
poses_df   = pd.read_csv(poses_path, header=None, sep=' ')
poses      = poses_df.apply(lambda row: read_pose(row.values), axis=1)

# Extract ground truth translation components
gt_trajectory = []
for pose in poses:
    _, T = pose
    x, y, z = T
    gt_trajectory.append([x, z])

gt_trajectory = np.array(gt_trajectory)

# Create window to display
win_name = "KITTI Sequence"
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
    hom_pose = transf_hom(R, T)

    # Update real-time pose
    rt_pose = np.matmul(rt_pose, np.linalg.inv(hom_pose))

    # Extract translation vector
    x, y, z = rt_pose[0, 3], rt_pose[1, 3], rt_pose[2, 3]

    # Store the current estimated position
    estimated_trajectory.append([x, z])

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

# Convert estimated trajectory to numpy array
estimated_trajectory = np.array(estimated_trajectory)

# Plot ground truth vs. estimated trajectory
plt.figure(figsize=(10, 5))
plt.subplot(121); plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label='Ground Truth'); plt.legend()
plt.subplot(122); plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label='Estimated'); plt.legend()
plt.savefig("path_comparison.jpg")
plt.show()
