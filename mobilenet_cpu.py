
# coding: utf-8

# ### Implementing LeNet-5 Architecture On MNIST Dataset (GPU Implementation)

# In[1]:

import torch
torch.multiprocessing.set_start_method("spawn")        # https://github.com/pytorch/pytorch/issues/3491#event-1326332533
import torch.nn   
import torch.optim 
import torch.nn.functional 
import torchvision.datasets   
import torchvision.transforms     

import numpy as np   # this is torch's wrapper for numpy 
#import matplotlib
#matplotlib.use('Agg')       
#get_ipython().magic('matplotlib inline')
#from matplotlib import pyplot    
#from matplotlib.pyplot import subplot     
from sklearn.metrics import accuracy_score
import time 

# In[2]:

# ---------- MNIST data from torch ----------     
# First download the dataset and set aside training and test data. Then perform transformation.  
# 'torchvision.transforms.compose()' creates a series of transformations to be applied on dataset. 
# 'torchvision' reads datasets into PILImage (Python imaging format) which are in [0, 255] range. 
# 'torchvision.transforms.ToTensor()' converts the PIL Image from range [0, 255] to a FloatTensor of 
# shape (C x H x W) with range [0.0, 1.0]
# We then renormalize the input of range [0, 1] to range [-1, 1] using μ = 0.5 and standard deviation = 0.5

# [Refer line 73] http://pytorch.org/docs/0.2.0/_modules/torchvision/datasets/mnist.html
# [Refer 'ToTensor' class] http://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html

transformImg = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)) , torchvision.transforms.ToTensor()])
train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transformImg)
valid = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transformImg)
test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transformImg)  

# create training and validation set indexes (80-20 split)
idx = list(range(len(train)))
np.random.seed(1009)
np.random.shuffle(idx)          
train_idx = idx[ : int(0.8 * len(idx))]       
valid_idx = idx[int(0.8 * len(idx)) : ]


# In[3]:

# sample images
fig1 = train.train_data[0].numpy()  
fig2 = train.train_data[2500].numpy()
fig3 = train.train_data[25000].numpy()  
fig4 = train.train_data[59999].numpy()
#subplot(2,2,1), pyplot.imshow(fig1)  
#subplot(2,2,2), pyplot.imshow(fig2) 
#subplot(2,2,3), pyplot.imshow(fig3)
#subplot(2,2,4), pyplot.imshow(fig4)


# In[4]:

# generate training and validation set samples
train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)    
valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)  

# Load training and validation data based on above samples
# Size of an individual batch during training and validation is 30
# Both training and validation datasets are shuffled at every epoch by 'SubsetRandomSampler()'. Test set is not shuffled.
train_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=train_set, num_workers=0)  
valid_loader = torch.utils.data.DataLoader(train, batch_size=30, sampler=valid_set, num_workers=0)    
test_loader = torch.utils.data.DataLoader(test, num_workers=0)       


# In[5]:

# Defining the network (LeNet-5)  
class LeNet5(torch.nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2,  bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=True, groups=32)

        # 112x112
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1,  bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, bias=True, groups=64)

        #56x56
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1,  bias=True)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=True, groups=128)

        self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1,  bias=True)
        self.conv8 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, bias=True, groups=128)

        #28x28
        self.conv9 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1,  bias=True)
        self.conv10 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True, groups=256)

        self.conv11 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1,  bias=True)
        self.conv12 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, bias=True, groups=256)

        #14x14
        self.conv13 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1,  bias=True)

        #1
        self.conv14 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,  bias=True, groups=512)
        self.conv15 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=True)
        #2
        self.conv16 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,  bias=True, groups=512)
        self.conv17 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=True)

        #3
        self.conv18 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,  bias=True, groups=512)
        self.conv19 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=True)
        #4
        self.conv20 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,  bias=True, groups=512)
        self.conv21 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=True)

        #5
        self.conv22 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,  bias=True, groups=512)
        self.conv23 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, bias=True)

        self.conv24 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2,  bias=True, groups=512)
        self.conv25 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, bias=True)   

        self.conv26 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1,  bias=True, groups=1024)
        self.conv27 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, bias=True)


        self.avg_pool = torch.nn.AvgPool2d(kernel_size=7)

        self.fc1 = torch.nn.Linear(1024, 1000)

        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        start = time.time() 
        x = torch.nn.functional.relu(self.conv1(x)) 
        end = time.time()
        print("elapsed time for layer 1 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv2(x)) 
        end = time.time()
        print("elapsed time for layer 2 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv3(x)) 
        end = time.time()
        print("elapsed time for layer 3 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv4(x)) 
        end = time.time()
        print("elapsed time for layer 4 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv5(x)) 
        end = time.time()
        print("elapsed time for layer 5 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv6(x)) 
        end = time.time()
        print("elapsed time for layer 6 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv7(x)) 
        end = time.time()
        print("elapsed time for layer 7 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv8(x)) 
        end = time.time()
        print("elapsed time for layer 8 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv9(x)) 
        end = time.time()
        print("elapsed time for layer 9 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv10(x)) 
        end = time.time()
        print("elapsed time for layer 10 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv11(x)) 
        end = time.time()
        print("elapsed time for layer 11 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv12(x)) 
        end = time.time()
        print("elapsed time for layer 12 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv13(x)) 
        end = time.time()
        print("elapsed time for layer 13 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv14(x)) 
        end = time.time()
        print("elapsed time for layer 14 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv15(x)) 
        end = time.time()
        print("elapsed time for layer 15 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv16(x)) 
        end = time.time()
        print("elapsed time for layer 16 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv17(x)) 
        end = time.time()
        print("elapsed time for layer 17 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv18(x)) 
        end = time.time()
        print("elapsed time for layer 18 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv19(x)) 
        end = time.time()
        print("elapsed time for layer 19 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv20(x)) 
        end = time.time()
        print("elapsed time for layer 20 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv21(x)) 
        end = time.time()
        print("elapsed time for layer 21 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv22(x)) 
        end = time.time()
        print("elapsed time for layer 22 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv23(x)) 
        end = time.time()
        print("elapsed time for layer 23 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv24(x)) 
        end = time.time()
        print("elapsed time for layer 24 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv25(x)) 
        end = time.time()
        print("elapsed time for layer 25 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv26(x)) 
        end = time.time()
        print("elapsed time for layer 26 : ", end - start)

        start = time.time() 
        x = torch.nn.functional.relu(self.conv27(x)) 
        end = time.time()
        print("elapsed time for layer 27 : ", end - start)

        start = time.time() 
        x = self.avg_pool(x) 
        end = time.time()
        print("elapsed time for layer 28 : ", end - start)

        start = time.time() 
        x = self.fc1(x) 
        end = time.time()
        print("elapsed time for layer 29 : ", end - start)

        # start = time.time() 
        # x = torch.nn.functional.Softmax(x) 
        # end = time.time()
        # print("elapsed time for layer 30 : ", end - start)


        # # convolve, then perform ReLU non-linearity
        # start = time.time()
        # x = torch.nn.functional.relu(self.conv2(x))
        # # max-pooling with 2x2 grid
        # x = self.max_pool_2(x)
        # end = time.time()
        # print("elapsed time for layer 2 : ", end - start)

        # # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # # read through https://stackoverflow.com/a/42482819/7551231
        # x = x.view(-1, 16*5*5)
        
        # start = time.time()
        # # FC-1, then perform ReLU non-linearity
        # x = torch.nn.functional.relu(self.fc1(x))
        # end = time.time()
        # print("elapsed time for layer 3 : ", end - start)

        # # FC-2, then perform ReLU non-linearity
        
        # start = time.time()
        # x = torch.nn.functional.relu(self.fc2(x))
        # end = time.time()
        # print("elapsed time for layer 4 : ", end - start)

        # # FC-3

        # start = time.time()
        # x = self.fc3(x)
        # end = time.time()
        # print("elapsed time for layer 5 : ", end - start)

        return x
     
net = LeNet5()     
net.cuda()


# In[6]:

# set up loss function -- 'SVM Loss' a.k.a 'Cross-Entropy Loss'
loss_func = torch.nn.CrossEntropyLoss()
       
# SGD used for optimization, momentum update used as parameter update  
optimization = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)


# In[7]:

# Let training begin!
numEpochs = 20    
training_accuracy = []     
validation_accuracy = []

for epoch in range(numEpochs):
    
    # training set -- perform model training
    epoch_training_loss = 0.0
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function        
        # split training data into inputs and labels
        inputs, labels = training_batch                              # 'training_batch' is a list               
        # wrap data in 'Variable'
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)        
        # Make gradients zero for parameters 'W', 'b'
        optimization.zero_grad()         
        # forward, backward pass with parameter update
        forward_output = net(inputs)
        loss = loss_func(forward_output, labels)
        loss.backward()   
        optimization.step()     
        # calculating loss 
        epoch_training_loss += loss.data
        num_batches += 1
        
    print("epoch: ", epoch, ", loss: ", epoch_training_loss/num_batches)            
     
    # calculate training set accuracy
    accuracy = 0.0 
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function        
        num_batches += 1
        inputs, actual_val = training_batch
        # perform classification
        predicted_val = net(torch.autograd.Variable(inputs))
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function    
        predicted_val = predicted_val.data.numpy()    # convert cuda() type to cpu(), then convert it to numpy
        predicted_val = np.argmax(predicted_val, axis = 1)  # retrieved max_values along every row    
        # accuracy   
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    training_accuracy.append(accuracy/num_batches)   

    # calculate validation set accuracy 
    accuracy = 0.0 
    num_batches = 0
    for batch_num, validation_batch in enumerate(valid_loader):        # 'enumerate' is a super helpful function        
        num_batches += 1
        inputs, actual_val = validation_batch
        # perform classification
        predicted_val = net(torch.autograd.Variable(inputs))    
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function    
        predicted_val = predicted_val.data.numpy()    # convert cuda() type to cpu(), then convert it to numpy
        predicted_val = np.argmax(predicted_val, axis = 1)  # retrieved max_values along every row    
        # accuracy        
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    validation_accuracy.append(accuracy/num_batches)


# In[8]:

epochs = list(range(numEpochs))

# plotting training and validation accuracies
#fig1 = pyplot.figure()
#pyplot.plot(epochs, training_accuracy, 'r')
#pyplot.plot(epochs, validation_accuracy, 'g')
#pyplot.xlabel("Epochs")
#pyplot.ylabel("Accuracy") 
#pyplot.show(fig1)


# In[9]:

# test the model on test dataset
correct = 0
total = 0 
print("started testing")
for test_data in test_loader:
    total += 1
    inputs, actual_val = test_data 
    # perform classification
    predicted_val = net(torch.autograd.Variable(inputs))   
    # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
    predicted_val = predicted_val.data
    max_score, idx = torch.max(predicted_val, 1)
    # compare it with actual value and estimate accuracy
    correct += (idx == actual_val).sum()
       
print("Classifier Accuracy: ", correct/total * 100)

