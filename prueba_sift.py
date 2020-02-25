import split_data
import cv2
import numpy as np
import os
import shutil
import wget
import tarfile
from itertools import compress
from matplotlib import pyplot as plt
import time
#Split the data
# folder='101_ObjectCategories/'
# split_data.splitDatabase(folder)
start_time = time.time()
img=cv2.imread('101_ObjectCategories/train/accordion/image_0001.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp,descriptors = sift.detectAndCompute(img,None)



print(descriptors)

img=cv2.drawKeypoints(gray,kp,img)


# plt.imshow(descriptors[0], cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show()

cv2.imshow('ImageWindow', img)
cv2.waitKey(0)
# plt.imshow(gray)
# plt.colorbar()
# plt.show()




print("--- %s seconds ---" % (time.time() - start_time))
