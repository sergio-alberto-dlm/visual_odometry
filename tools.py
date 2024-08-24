import numpy as np 
import pandas as pd 
import cv2 as cv 

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
    index_params    = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,     # 20
                        multi_probe_level=1)  # 2
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

