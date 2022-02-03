# 2

import numpy as np
import cv2 

#---------------------------------------

# Read the images to be aligned
camera_lapl_thr =  cv2.imread("images/camera_lapl_thr.png");


depth_lapl_thr =  cv2.imread("images/depth_lapl_thr.png");


camera_segm = cv2.imread("images/camera_seg.png")
#camera_segm = cv2.imread("images/camera_corrupt_seg.png")



# Convert images to grayscale
camera_lapl_thr_gray = cv2.cvtColor(camera_lapl_thr,cv2.COLOR_BGR2GRAY)
depth_lapl_thr_gray = cv2.cvtColor(depth_lapl_thr,cv2.COLOR_BGR2GRAY)

sz = camera_lapl_thr.shape	
warp_mode = cv2.MOTION_TRANSLATION  #cv2.MOTION_HOMOGRAPHY

if warp_mode == cv2.MOTION_HOMOGRAPHY :
	warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
	warp_matrix = np.eye(2, 3, dtype=np.float32)
number_of_iterations = 500
termination_eps = 1e-10

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC (depth_lapl_thr_gray,camera_lapl_thr_gray,warp_matrix, warp_mode, criteria)


if warp_mode == cv2.MOTION_HOMOGRAPHY :	# Use warpPerspective for Homography
	camera_seg_align = cv2.warpPerspective (camera_segm, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

else :	# Use warpAffine for Translation, Euclidean and Affine
	camera_seg_align = cv2.warpAffine(camera_segm, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
		

kernel = np.ones((3, 3), 'uint8')
camera_seg_align = cv2.dilate(camera_seg_align, kernel, iterations=1)

# Show final results
cv2.imshow("Image 1", camera_lapl_thr)
cv2.imshow("Image 2", depth_lapl_thr)
cv2.imshow("Aligned camera image", camera_seg_align)

cv2.waitKey(0)

cv2.imwrite("images/camera_seg_align.png", camera_seg_align)
