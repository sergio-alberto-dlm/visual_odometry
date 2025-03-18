import os 
import json 
import copy 
import wandb 
from utils.logging_utils import Log

import numpy as np 
# import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from utils.general_utils import mkdir_p

from evo.core import metrics 
from evo.core.trajectory import PosePath3D
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.tools import plot
plot.apply_settings(SETTINGS)

from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)

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
    traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)

    ## RMSE
    data = (traj_ref, traj_est_aligned)
    ate_stat, ate_stats, ate_metric = compute_metric(data)
    # log out 
    Log(f"RMSE ATE [m]", ate_stat, tag="Eval")
    # plot 
    plot_mode = plot.PlotMode.xz
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
    wandb.log({"frame_idx": latest_frame_idx, "ate_trans": ate})

    return ate