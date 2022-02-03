# 3

import  cv2
import  numpy as np


  
depth_gray =  cv2.imread("images/depth_gray.png")#, cv2.IMREAD_GRAYSCALE);


camera_seg_align =  cv2.imread("images/camera_seg_align.png")#, cv2.IMREAD_GRAYSCALE);

#Calculate centerX and centerY
M = cv2.moments(camera_seg_align[:,:,0])
Xmed = int(M["m10"]/M["m00"])
Ymed = int(M["m01"]/M["m00"])
print(Xmed,Ymed)

# Mask used to flood filling. (mask size needs to be 2 pixels larger)

h, w = depth_gray.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)



pixel = depth_gray[Ymed,Xmed]  #[320,180]
print(pixel)
#low_lim = ( pixel[0], pixel[1], pixel[2])
low_lim = (int(pixel[0]-30),int(pixel[1]-30),int(pixel[2]-30))
print(low_lim)

#th, depth_thr = cv2.threshold(depth_gray, 80, 255, cv2.THRESH_BINARY_INV);

#pixel = img_depth[180,320]
#print(img_depth[)

    
#cv2.floodFill(depth_gray, mask, (0,0), 255, flags=255<<8);
cv2.floodFill(depth_gray, mask, (180, 320), (255, 255, 255), low_lim, (150, 150, 150), cv2.FLOODFILL_FIXED_RANGE)

cv2.imshow("Floodfilled image", depth_gray)

_, depth_gray_extr = cv2.threshold(depth_gray, 254, 500, cv2.THRESH_BINARY) #cv2.THRESH_BINARY #cv2.THRESH_TOZERO
cv2.imshow("Floodfilled image binary", depth_gray_extr)

cv2.imwrite("images\depth_extr.png",depth_gray_extr)

cv2.waitKey(0)
