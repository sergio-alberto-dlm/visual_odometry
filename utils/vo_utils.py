import math 
import numpy as np 

import torch 
import cv2

from utils.pose_utils import getWorld2View2

DEVICE="cpu"

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device=DEVICE
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device=DEVICE
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=DEVICE)
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=DEVICE)
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)

def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()

def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def get_matches(viewpoint, prev):
    
    """
        Function to find correspondances between two frames 

        input  : two frames 
        output : pair of corresponding points and keypoints 
    """
    # FLANN parameters for LSH (suitable for binary descriptors like ORB)
    FLANN_INDEX_LSH = 6
    index_params    = dict(
                        algorithm         = FLANN_INDEX_LSH,
                        table_number      = 6,  
                        key_size          = 12,     
                        multi_probe_level = 1
                        ) 
    search_params = dict(checks=50)  # or pass empty dictionary

    # Create FLANN-based matcher
    flann   = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(prev.descriptors, viewpoint.descriptors, k=2)

    # Find corresponding points 
    pts1 = []
    pts2 = []
    
    # Ratio test as per Lowe's paper
    for i, m_n in enumerate(matches):
        if len(m_n) < 2:
            continue  # Not enough matches to apply ratio test
        m, n = m_n
        if m.distance < 0.8*n.distance:
            pts1.append(prev.keypoints[m.queryIdx].pt)
            pts2.append(viewpoint.keypoints[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return {"matches" : (pts1, pts2), "keypoints" : (prev.keypoints, viewpoint.keypoints)}

def get_pose(pts1, pts2, cameraMatrix):
    
    """
        Function to recover the rootation and translation between 
        two corresponding points 
    """

    # Compute essential matrix 
    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix)

    # recover pose 
    _, R, T, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix)

    return getWorld2View2(R=torch.from_numpy(R), t=torch.from_numpy(T).squeeze())

def wrapping():
    pass 