import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify
import numpy as np 

import wandb
from utils.general_utils import mkdir_p, load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate
from utils.logging_utils import Log
from utils.vo_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        start_time = time.time()

        self.config = config
        self.save_dir = save_dir

        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.dataset = load_dataset(args=None, path=None, config=config)

        frontend_queue = mp.Queue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.frontend_queue = frontend_queue
        self.frontend.set_hyperparams()


        self.frontend.run()

        # end.record()
        # torch.cuda.synchronize()
        end_time = time.time()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        # elapsed_time = start.elapsed_time(end)
        elapsed_time = end_time - start_time
        FPS = N_frames / (elapsed_time * 0.001)
        Log("Total time", elapsed_time * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (elapsed_time * 0.001), tag="Eval")

        ATE = eval_ate(
            self.frontend.cameras,
            np.arange(0, len(self.dataset), 20), #self.frontend.kf_indices,
            self.save_dir,
            0,
            final=True,
            monocular=self.monocular,
        )

        columns = ["RMSE ATE", "FPS"]
        metrics_table = wandb.Table(columns=columns)
        metrics_table.add_data(
                ATE,
                FPS,
            )
        wandb.log({"Metrics": metrics_table})

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None
    PROJECT_NAME = "visual_odometry_kitti"

    if args.eval:
        Log("Running Visual Odometry in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-1] + "_" + config["Dataset"]["sequence"], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project=PROJECT_NAME,
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")