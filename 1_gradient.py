# 1

import numpy as np
import cv2 


img_camera = cv2.imread('image_cam\camera_seg.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow("Camera image", img_camera)

img_depth = cv2.imread('images\depth_colored.jpg')

for x in range(0,img_depth.shape[0]):
	for y in range(0,img_depth.shape[1]):
		img_depth[x,y,2] = 0.90 * img_depth[x,y,2] + 0.10 * img_depth[x,y,1];
		img_depth[x,y,1] = img_depth[x,y,2] ;
		img_depth[x,y,0] = img_depth[x,y,2] ;
depth_gray = cv2.cvtColor(img_depth,cv2.COLOR_BGR2GRAY)

cv2.imshow("Depth image",img_depth)
cv2.imwrite("images\depth_gray.png",depth_gray)




img_camera_lapl = cv2.Laplacian(img_camera,cv2.CV_64F, ksize=7)
img_depth_lapl = cv2.Laplacian(img_depth,cv2.CV_64F, ksize=7)



_, img_camera_lapl_thr = cv2.threshold(img_camera_lapl, 1700, 5000, cv2.THRESH_BINARY) #cv2.THRESH_BINARY #cv2.THRESH_TOZERO

cv2.imshow("Thresholded camera laplacian", img_camera_lapl_thr)
cv2.imwrite("images\camera_lapl_thr.png", img_camera_lapl_thr)

_, img_depth_lapl_thr = cv2.threshold(img_depth_lapl, 1700, 5000, cv2.THRESH_BINARY)
cv2.imshow("Thresholded depth laplacian", img_depth_lapl_thr)
cv2.imwrite("images\depth_lapl_thr.png", img_depth_lapl_thr)

print("Press any key to continue")
cv2.waitKey(0)

