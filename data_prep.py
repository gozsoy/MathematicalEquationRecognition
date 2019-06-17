import numpy as np
import os
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import cv2 as cv

# consider subtracting mean and dividing by variance if everything goes not well
# images are not binary here but your test images will be binary

LIMIT_PER_CLASS=200
LIMIT_PER_CLASS_TEST=10
train_samples=[]
test_samples=[]

#pil2tensor=transforms.Compose([transforms.ToTensor()])
#instead of above. calculate the mean and std yourself and do the job.
pil2tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])

def readData():
    mypath1='/Users/gokberk/Desktop/crohme-data-extractor-master/extracted_images_80/'
    mypath2=['int','theta','sum','alpha','beta','Delta','gamma','lambda','mu','phi','pi','sigma','It','0','1','2','3','4','5','6','7','8','9']

    #Image._show(Image.open('rr.png'))
    # 1 -> white
    # 0 -> black

    for path_index in range(len(mypath2)):
        for root, dirs, files in os.walk(mypath1+mypath2[path_index]):
            #print(len(files))
            temp_files=[]
            for i in range(len(files)):
                if files[i]!='.DS_Store':
                    temp_files.append(files[i])
            files=temp_files
            indices_array=np.arange(len(files))
            np.random.shuffle(indices_array)

            up_bound=-1
            if LIMIT_PER_CLASS < len(files)-LIMIT_PER_CLASS_TEST:
                up_bound=LIMIT_PER_CLASS
            else:
                up_bound=len(files)-LIMIT_PER_CLASS_TEST

            for i in range(up_bound):
            #for i in range(len(files)):
                file_index=indices_array[i]
                file_itself=files[file_index]
                path2file= root + '/' + file_itself
                im_data = cv.imread(path2file)
                im_data = cv.cvtColor(im_data, cv.COLOR_BGR2GRAY)
                im_data=torch.squeeze(pil2tensor(im_data),0)
                if path_index <= 12:
                    label=0
                else:
                    label=1
                im_dict={'data':im_data,'class':label}
                train_samples.append(im_dict)

            #
            for k in range(LIMIT_PER_CLASS_TEST):
                 file_index=indices_array[k+up_bound]
                 file_itself=files[file_index]
                 path2file= root + '/' + file_itself
                 im_data = cv.imread(path2file)
                 im_data = cv.cvtColor(im_data, cv.COLOR_BGR2GRAY)
                 im_data = torch.squeeze(pil2tensor(im_data), 0)
                 #im_data=torch.squeeze(pil2tensor(Image.open(path2file)),0)
                 if path_index <= 12:
                     label = 0
                 else:
                     label = 1
                 im_dict={'data':im_data,'class':label}
                 test_samples.append(im_dict)

    training_data = []
    labels=torch.zeros(1,len(train_samples),dtype=torch.long)

    test_data = []
    test_labels = torch.zeros(1, len(test_samples), dtype=torch.long)

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    for j in range(len(train_samples)):
        training_data.append(train_samples[j]['data'])
        labels[0][j]=(train_samples[j]['class'])

    for j in range(len(test_samples)):
        test_data.append(test_samples[j]['data'])
        test_labels[0][j]=(test_samples[j]['class'])

    #return training_data,labels,len(mypath2),test_data,test_labels
    return training_data, labels,2, test_data, test_labels
    #return training_data, labels, len(mypath2)

segmentation_data=[]

def readSegmentationTestData():
    mypath = '/Users/gokberk/Desktop/data/trials4segmentation/'

    for root, dirs, files in os.walk(mypath):
        for i in range(len(files)):
            if files[i]!='.DS_Store':
                print(files[i])
                file_itself = files[i]
                path2file = root + '/' + file_itself
                #im_data = torch.squeeze(pil2tensor(Image.open(path2file)), 0)
                im_data = cv.imread(path2file)
                im_data = cv.cvtColor(im_data, cv.COLOR_BGR2GRAY)
                im_data = torch.squeeze(pil2tensor(im_data), 0)
                segmentation_data.append(im_data)

    return segmentation_data