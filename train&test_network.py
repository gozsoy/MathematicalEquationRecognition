import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_prep
import network
import matplotlib.pyplot as plt
import torchvision
import hog


device=torch.device("cpu")

training_data,labels,total_class_size,test_data,test_labels=data_prep.readData()
#training_data,labels,total_class_size=data_prep.readData()
training_data=torch.stack(training_data)
test_data=torch.stack(test_data)

training_data=training_data.numpy()
test_data=test_data.numpy()



hog_features=np.zeros((len(training_data),64))

for i in range(len(training_data)):
    hog_features[i]=hog.hog(training_data[i])

hog_features=torch.from_numpy(hog_features)

test_hog_features=np.zeros((len(test_data),64))

for i in range(len(test_data)):
    test_hog_features[i]=hog.hog(test_data[i])

test_hog_features=torch.from_numpy(test_hog_features)

segmentation_test_data=data_prep.readSegmentationTestData()
segmentation_test_data=torch.stack(segmentation_test_data)
segmentation_test_data=segmentation_test_data.numpy()

segmentation_hog_features=np.zeros((len(segmentation_test_data),64))

for i in range(len(segmentation_test_data)):
    segmentation_hog_features[i]=hog.hog(segmentation_test_data[i])

segmentation_hog_features=torch.from_numpy(segmentation_hog_features)

model=network.Net2(total_class_size)

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(model.parameters(),lr=5*10e-4)

# Training
mini_batch_size=16
total_batch_no=np.int(np.floor(len(training_data)/mini_batch_size))

loss_array=[]
for epoch in range(100):
    accum=0
    for batch_index in range(total_batch_no):

        mini_batch=hog_features[batch_index*mini_batch_size:(batch_index+1)*mini_batch_size]
        mini_targets=labels[0][batch_index*mini_batch_size:(batch_index+1)*mini_batch_size]

        output=model(mini_batch.float())
        loss=criterion(output,mini_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accum += loss.item()

    print("epoch "+str(epoch)+" : loss "+ str(accum/total_batch_no))
    loss_array.append(accum/total_batch_no)


plt.plot(loss_array)
#plt.show()

#Test
with torch.no_grad():
    test_output=model(test_hog_features.float())
    soft=nn.LogSoftmax(0)
    test_output=soft(test_output)
    test_output_array=[]
    for t in range(len(test_output)):
        test_output_array.append(test_output[t].numpy())

    for t2 in range(len(test_output_array)):
        test_output_array[t2] = np.argmax(test_output_array[t2], 0)


    no_of_correct_labels=0
    for p in range(len(test_output_array)):
        if test_labels[0][p].item()==test_output_array[p]:
            no_of_correct_labels += 1

    test_accuracy=(no_of_correct_labels/len(test_output_array))*100
    print(test_accuracy)



#Segmentation results Test
with torch.no_grad():
     test_output=model(segmentation_hog_features.float())
     soft=nn.LogSoftmax(0)
     test_output=soft(test_output)
     test_output_array=[]
     for t in range(len(test_output)):
         test_output_array.append(test_output[t].numpy())
#
     for t2 in range(len(test_output_array)):
         test_output_array[t2] = np.argmax(test_output_array[t2], 0)
#
     print(test_output_array)

