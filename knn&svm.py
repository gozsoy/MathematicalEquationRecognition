import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_prep
import network
import matplotlib.pyplot as plt
import torchvision
import hog
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
import scipy.stats
import sklearn.svm as svm

# training data ready
training_data,labels,total_class_size,test_data,test_labels=data_prep.readData()

training_data=torch.stack(training_data)
test_data=torch.stack(test_data)

training_data=training_data.numpy()
test_data=test_data.numpy()


labels=labels.numpy()
labels=np.transpose(labels)

test_labels=test_labels.numpy()

hog_features=np.zeros((len(training_data),80*80))

for i in range(len(training_data)):
    hog_features[i]=hog.hog(training_data[i])



test_hog_features=np.zeros((len(test_data),80*80))

for i in range(len(test_data)):
    test_hog_features[i]=hog.hog(test_data[i])

# segmentation test data ready
#segmentation_data=data_prep.readSegmentationTestData()
#segmentation_data=torch.stack(segmentation_data)
#segmentation_data=segmentation_data.numpy()
#
#test_hog_features=np.zeros((len(segmentation_data),64))
#
#for i in range(len(segmentation_data)):
#    test_hog_features[i]=hog.hog(segmentation_data[i])

# create knn model
# neigh1 = KNeighborsClassifier(n_neighbors=7)
# neigh1.fit(hog_features,labels)
# predictions1=neigh1.predict(test_hog_features)
#
#
# neigh2 = KNeighborsClassifier(n_neighbors=3)
# neigh2.fit(hog_features,labels)
# predictions2=neigh2.predict(test_hog_features)
#
# neigh3 = KNeighborsClassifier(n_neighbors=11)
# neigh3.fit(hog_features,labels)
# predictions3=neigh3.predict(test_hog_features)
#
# final_predictions=np.zeros((1,len(predictions1)),int)
#
# for j in range(len(predictions1)):
#     temp=np.zeros((3,1))
#     temp[0]=predictions1[j]
#     temp[1]=predictions2[j]
#     temp[2]=predictions3[j]
#     temp=np.transpose(temp)
#     mode_label=int(scipy.stats.mode(temp[0]).mode[0])
#     final_predictions[0][j]=mode_label
#
#
# accuracy=    ((sum(sum(final_predictions==test_labels[0]))) / (len(test_labels[0])))*100
# print(accuracy  )


# create SVM model
svm=svm.LinearSVC()
svm.fit(hog_features,labels)
predictions=svm.predict(test_hog_features)
accuracy=    ( sum(predictions==test_labels[0]) / (len(test_labels[0])))*100
print(accuracy  )
