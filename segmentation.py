import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from rotated_rect_crop import crop_rotated_rectangle
import hog


img_original = cv.imread('ewq.jpg')
hhh=hog.hog(img_original)
print(np.shape(hhh))
height, width = img_original.shape[:2]
img_original = cv.resize(img_original,(800, 800), interpolation = cv.INTER_CUBIC)
#print(np.shape(res))
img = cv.GaussianBlur(img_original, (5,5), 0)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#ret, img = cv.threshold(img,0,1,cv.THRESH_OTSU)
ret, img = cv.threshold(img, 127, 255, 0)

img, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#cont=contours[1:len(contours)]
cont=contours
#cv.drawContours(img_original, cont, -1, (0,255,0), 3)


rect = cv.minAreaRect(contours[16])
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img_original,[box],0,(0,0,255),5)

rect2 = cv.minAreaRect(contours[5])
box2 = cv.boxPoints(rect2)
box2 = np.int0(box2)
cv.drawContours(img_original,[box2],0,(0,0,255),5)
imgplot = plt.imshow(img_original,cmap='Greys_r')
plt.show()


#image_cropped = crop_rotated_rectangle(img, rect)
#rows,cols=np.shape(image_cropped)
#M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
#image_cropped = cv.warpAffine(image_cropped,M,(cols,rows))
#crop_symbol = cv.resize(image_cropped, (45, 45))
#imgplot = plt.imshow(image_cropped,cmap='Greys_r')
#plt.show()
# max_area=-1
# arg_max=-1
# for i in range(len(contours)):
#      x,y,w,h = cv.boundingRect(contours[i])
#      area=w*h
#      cv.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),5)
#      imgplot = plt.imshow(img_original)
#      plt.show()
#      if area!=800**2:
#          if area > max_area:
#              max_area=area
#              arg_max=i
#
#
#
#x,y,w,h = cv.boundingRect(contours[15])
#margin=5
#crop_symbol = img[y-margin:y+h+margin, x-margin:x+w+margin]
#crop_symbol = cv.resize(crop_symbol, (80, 80))
#kernel = np.ones((3,3),np.uint8)
#crop_symbol = cv.dilate(crop_symbol,kernel,iterations = 1)

#cv.imwrite('seg_sum1_80.jpg', crop_symbol)
#imgplot = plt.imshow(crop_symbol,cmap='Greys_r')
#plt.show()

#cv.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),5)
#imgplot = plt.imshow(img_original)
#plt.show()