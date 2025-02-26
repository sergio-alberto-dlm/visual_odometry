# tools 
import os 
import json 
import rich 
import copy 

# math and vision 
import numpy as np 
import pandas as pd 
import cv2 as cv 

# visualization 
import matplotlib
matplotlib.use('Agg') 
from matplotlib import pyplot as plt

# evaluation 
# import wandb 
from evo.core import metrics
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.tools import plot
plot.apply_settings(SETTINGS)
from evo.core.trajectory import PosePath3D

# ---> functions
def read_calib(path_calib_seq):

    """
        Function to read the intrinsic parameters 
        and the projection matrix 
    """

    params = pd.read_csv(path_calib_seq, header=None, sep=' ').to_numpy()[:, 1:]
    P      = np.array(params[0].reshape(3, 4), dtype=np.float32)
    K      = P[:3, :3]

    return P, K

def read_pose(pose):
    
    """
        This function takes the pose and returns 
        the rotation matrix (R) and translation (T)
    """

    pose = pose.reshape(3, 4)
    R    = pose[:3, :3]
    T    = pose[:, -1]
    
    return R, T
    
def transf_hom(rotation : np.array, translation : np.array):

    """
        Function to recover homogeneous coordinates 
    """

    pose_hom         = np.eye(4, dtype=np.float32)
    pose_hom[:3, :3] = rotation
    pose_hom[:3, -1] = translation.flatten()

    return pose_hom

def get_matches(img1, img2):
    
    """
        Function to find correspondances between two frames 

        input  : two frames 
        output : pair of corresponding points and keypoints 
    """

    # Detect features 
    max_num_features         = 3000
    orb                      = cv.ORB_create(max_num_features)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # FLANN parameters for LSH (suitable for binary descriptors like ORB)
    FLANN_INDEX_LSH = 6
    index_params    = dict(
                        algorithm         = FLANN_INDEX_LSH,
                        table_number      = 6,  
                        key_size          = 12,     
                        multi_probe_level = 1
                        ) 
    search_params   = dict(checks=50)  # or pass empty dictionary

    # Create FLANN-based matcher
    flann   = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Find corresponding points 
    pts1 = []
    pts2 = []
    
    # Ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(keypoints2[m.trainIdx].pt)
            pts1.append(keypoints1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return {"matches" : (pts1, pts2), "keypoints" : (keypoints1, keypoints2)}

def get_pose(pts1, pts2, cameraMatrix):
    
    """
        Function to recover the rootation and translation between 
        two corresponding points 
    """

    # Compute essential matrix 
    E, mask = cv.findEssentialMat(pts1, pts2, cameraMatrix)

    # recover pose 
    _, R, T, mask = cv.recoverPose(E, pts1, pts2, cameraMatrix)

    return R, T

def drawBannerText(frame, text, banner_height_percent = 0.07, text_color = (0,255,0)):

    """
    Function to annotate a frame 
    """
    # Draw a black filled banner across the top of the image frame.
    # percent: set the banner height as a percentage of the frame height.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv.rectangle(frame, (0,0), (frame.shape[1],banner_height), (0,0,0), thickness=-1)
    
    # Draw text on banner.
    left_offset = 20
    location = (left_offset, int( 5 + (banner_height_percent * frame.shape[0])/2 ))
    fontScale = 1.5
    fontThickness = 2
    cv.putText(frame, text, location, cv.FONT_HERSHEY_PLAIN, fontScale, text_color, fontThickness, cv.LINE_AA)

def mkdir_p(folder_path):
    from errno import EEXIST
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        os.makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise

_log_styles = {
    "MonoGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"

def Log(*args, tag="MonoGS"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)

def compute_metric(data: tuple):
    pe_metric = metrics.APE(metrics.PoseRelation.translation_part)
    pe_metric.process_data(data)
    pe_stat = pe_metric.get_statistic(metrics.StatisticsType.rmse)
    pe_stats = pe_metric.get_all_statistics()
    return pe_stat, pe_stats, pe_metric


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular):
    # sincronize data & align 
    max_diff = 0.01
    traj_ref, traj_est = PosePath3D(poses_se3=poses_gt), PosePath3D(poses_se3=poses_est)
    # traj_ref, traj_est = sync.associate_trajectories(poses_gt, poses_est, max_diff)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    ## RMSE
    data = (traj_ref, traj_est_aligned)
    ate_stat, ate_stats, ate_metric = compute_metric(data)
    # log out 
    Log(f"RMSE ATE [m]", ate_stat, tag="Eval")
    # plot 
    plot_mode = plot.PlotMode.xy
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est_aligned, ate_metric.error, 
                       plot_mode, min_map=ate_stats["min"], max_map=ate_stats["max"])
    ax.legend()
    plt.show()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ate_stats, f, indent=4)

    return ate_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R#.cpu().numpy()
        pose[0:3, 3] = T#.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    #wandb.log({"frame_idx": latest_frame_idx, "ate_trans": stat_full["APE"][str(metrics.PoseRelation.translation_part)]})
    # wandb.log({"frame_idx": latest_frame_idx, "ate_rot": stat_full["APE"][str(metrics.PoseRelation.rotation_angle_rad)]})
    #wandb.log({"frame_idx": latest_frame_idx, "rte_trans": stat_full["RPE"][str(metrics.PoseRelation.translation_part)]})
    # wandb.log({"frame_idx": latest_frame_idx, "rte_rot": stat_full["RPE"][str(metrics.PoseRelation.rotation_angle_rad)]})

    return ate