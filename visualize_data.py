# tools
from tools import drawBannerText
import numpy as np 
import cv2 as cv 
import os 


# frames path
num_seq   = "00"
dir_path  = "./gray_images/sequences/" + num_seq + "/image_0"
img_list  = sorted(os.listdir(dir_path))

# process frames 
win_name = "kitti sequence"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

frame_count = 0
for frame_name in img_list:
    frame = cv.imread(os.path.join(dir_path, frame_name), cv.IMREAD_COLOR)
    if frame is None:
        print(f"Warning: Could not read {frame_name}. Skipping...")
        continue

    frame_count += 1
    drawBannerText(frame, f'Frame: {frame_count}')

    cv.imshow(win_name, frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyWindow(win_name)
