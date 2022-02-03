# 4

import  cv2
import  numpy as np


  
depth_extr =  cv2.imread("images/depth_extr.png")#, cv2.IMREAD_GRAYSCALE);

cv2.imshow("Depth thres",depth_extr)
kernel = np.ones((3, 3), 'uint8')
depth_extr = cv2.erode(depth_extr, kernel, iterations=1)
cv2.imshow("Depth thres 2",depth_extr)

camera_seg_align =  cv2.imread("images/camera_seg_align.png")#, cv2.IMREAD_GRAYSCALE);

cv2.imshow("Camera segm",camera_seg_align)
kernel = np.ones((3, 3), 'uint8')
camera_seg_align = cv2.erode(camera_seg_align, kernel, iterations=1)
cv2.imshow("Camera segm 2",camera_seg_align)

intersection = cv2.bitwise_and(depth_extr,camera_seg_align)

intersection = cv2.dilate(intersection, kernel, iterations=2)

cv2.imshow("Intersection",intersection)

cv2.imwrite("images\intersection.png",intersection)

cv2.waitKey(0)


# Fabian Chersi. 30.09.2021
# Free for non-commercial personal use only.

