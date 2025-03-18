import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from utils.pose_utils import focal2fov

class KITTIParser:
    def __init__(self, input_folder, sequence):
        self.input_folder = input_folder
        self.sequence = sequence
        self.load_poses(self.input_folder)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=float, skiprows=skiprows)
        return data
    
    def read_pose(self, pose_vec):
        """
        Returns a pose matrix from a 12 lenght vector, 
        first 9 elements for rotation, the rest for translation 
        """
        pose_vec = pose_vec.reshape(3, 4)
        return np.linalg.inv(np.vstack([pose_vec, np.array([0, 0, 0, 1], dtype=float)]))

    def load_poses(self, datapath):
        pose_list = os.path.join(datapath, f"dataset_poses/poses/" + self.sequence + ".txt")
        assert os.path.isfile(pose_list), print(f"poses didn't find in {pose_list}")

        color_list = os.path.join(datapath, f"gray_images/sequences/" + self.sequence + "/image_0")
        assert os.path.isdir(color_list), f"images didn't find in {color_list}"

        self.poses = np.apply_along_axis(lambda row: self.read_pose(row), axis=1, arr=self.parse_list(pose_list))
        rgb_names = sorted(os.listdir(color_list))
        self.color_paths = [os.path.join(color_list, rgb_name) for rgb_name in rgb_names]
        self.depth_paths = None 

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cpu"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        if len(image.shape) == 2: image = image[None].reshape(self.height, self.width, -1)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose

class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        sequence = config["Dataset"]["sequence"]
        parser = KITTIParser(dataset_path, sequence)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "kitti":
        return KITTIDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")