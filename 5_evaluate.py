# 5

import  cv2
import  numpy as np


my_silh =  cv2.imread("images/intersection.png")#, cv2.IMREAD_GRAYSCALE);


gr_truth = cv2.imread("images/ground_truth.png")
#gr_truth = cv2.imread("images/ground_truth_cut.png")


sz = my_silh.shape


        
        
img_inters = cv2.bitwise_and(my_silh,gr_truth)
img_union = cv2.bitwise_or(my_silh,gr_truth)

count_int = np.count_nonzero(img_inters[:,:,0]==255)
count_uni = np.count_nonzero(img_union[:,:,0]==255)

print("Intersection pixels:",count_int,"  Union pixels:",count_uni)
print()
print(" IOU = %1.3f" % (count_int/count_uni) )


# Fabian Chersi. 30.09.2021
# Free for non-commercial personal use only.

