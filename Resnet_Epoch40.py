trainDatasetPath = "D:\Personal Projects\ISM_2022\\train\\train"
testDatasetPath = "D:\Personal Projects\ISM_2022\\test\\"
test_noiseDataSetPath = "D:\Personal Projects\ISM_2022\\test_noise\\"
mtec_testDataSetPath = "D:\Personal Projects\ISM_2022\\mtec_test\\"
csvPath = "D:\Personal Projects\ISM_2022\ImageName_with_Class.csv"
train_csvPath = "D:\Personal Projects\ISM_2022\TrainingDataSplit.csv"
val_csvPath = "D:\Personal Projects\ISM_2022\ValidationDataSplit.csv"

import os
import torch
import pandas as pd
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import numpy as np
import random
import torch.optim as optim
from sklearn.model_selection import train_test_split


#Input Image
imageHeight = 256
imageWidth = 256

modelImgSize = 224

numOutputClasses = 4

#Hyperparameters
batchSize = 4
learningRate = 0.01
epochs = 41



#GPU Info
print('Numbers of GPUs: ', torch.cuda.device_count())
print('Type of GPUs: ', torch.cuda.get_device_name(device=None))

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CompleteDataset = pd.read_csv(csvPath)
#print(CompleteDataset)

imageName = CompleteDataset['Image Name']
imageClass = CompleteDataset['Image Class']
#print(imageName)
#print(imageClass)

Xarr = imageName.to_numpy()
Yarr = imageClass.to_numpy()

X_Train, X_Val, Y_Train, Y_Val = train_test_split(Xarr, Yarr, test_size=0.2, shuffle=True, random_state=4)

trainingData = pd.DataFrame(list(zip(X_Train,Y_Train)),columns=['Image Name', 'Image Class'])

validationData = pd.DataFrame(list(zip(X_Val,Y_Val)),columns=['Image Name', 'Image Class'])

print(trainingData)
print(validationData)


trainingData.to_csv('TrainingDataSplit.csv',index=False)
#!cp TrainingDataSplit.csv "drive/My Drive/data/"

validationData.to_csv('ValidationDataSplit.csv',index=False)
#!cp ValidationDataSplit.csv "drive/My Drive/data/"

class CreateDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_transforms = transforms.Compose ([transforms.ToPILImage(),
                                        transforms.Resize((imageHeight,imageWidth)),
                                        transforms.CenterCrop(modelImgSize),
                                        transforms.RandomApply([transforms.RandomRotation(45)], p =0.2),
                                      transforms.RandomHorizontalFlip(p=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

test_transforms =  transforms.Compose ([transforms.ToPILImage(),
                                        transforms.Resize((imageHeight,imageWidth)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

val_transforms = test_transforms

trainDataset = CreateDataset(csv_file = train_csvPath, img_dir = trainDatasetPath, transform = train_transforms)
trainDataloader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True, pin_memory=True)

totalItems = len(trainDataloader)
totalImages = totalItems * batchSize
print("Train DataLoader Items : " + str(totalItems))
print("No. of Training Images (Approx) : " + str(totalImages))

#Expected value of totalItems = Total No. of images in Dataset / batchSize

valDatasetPath = trainDatasetPath

valDataset = CreateDataset(csv_file = val_csvPath, img_dir = valDatasetPath, transform = val_transforms)
valDataloader = torch.utils.data.DataLoader(dataset = valDataset, batch_size = batchSize, shuffle = True, pin_memory=True)

totalItems = len(valDataloader)
totalImages = totalItems * batchSize
print("Validation DataLoader Items : " + str(totalItems))
print("No. of Validation Images (Approx)  : " + str(totalImages))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data and make a grid
dataIter = iter(trainDataloader)
trainInputs, classes = dataIter.next()
trainingGrid = torchvision.utils.make_grid(trainInputs)
print("Sample of Training Images")
#imshow(trainingGrid)


# Get a batch of training data and make a grid
valInputs, classes = next(iter(valDataloader))
validationGrid = torchvision.utils.make_grid(valInputs)
print("Sample of Validation Images")
#imshow(validationGrid)


for images, labels in trainDataloader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    print(labels)
    break


model = torchvision.models.resnet101(pretrained=True)
#When backpropagation is required through all the layers (fine-tuning), keep this code commented
#When used as a feature extractor, uncomment the below code
#for param in model.parameters():
#    param.requires_grad = False


lastLayers = nn.Sequential(#nn.Linear(2208, 1000), 
                           #nn.ReLU(),
                           #nn.Dropout(p=0.5),
                           nn.Linear(1024, 512), 
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(512, 100), 
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(100, numOutputClasses))

model.classifier = lastLayers

model.to(device)

## compute accuracy
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# for epoch in range(epochs):
#     train_running_loss = 0.0
#     train_acc = 0.0

#     model = model.train()

#     ## training step
#     for i, (images, labels) in enumerate(trainDataloader):
        
#         images = images.to(device)
#         labels = labels.to(device)

#         ## forward + backprop + loss
#         logits = model(images)
#         loss = criterion(logits, labels)
#         optimizer.zero_grad()
#         loss.backward()

#         ## update model params
#         optimizer.step()

#         train_running_loss += loss.detach().item()
#         train_acc += get_accuracy(logits, labels, batchSize)
        
#         if i % 20 == 0:
#           print('Epoch: {} | Loss: {} | Train Accuracy: {}'.format(epoch, train_running_loss / (i+1), train_acc/(i+1)))        
    
#     model.eval()
#     print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
#           %(epoch+1, train_running_loss / i, train_acc/i)) 


# def save_checkpoint(state, filename="dense121frozen_40.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
# save_checkpoint(checkpoint)

load_checkpoint(torch.load("D:\Personal Projects\ISM_2022\\dense121frozen_40.pth.tar"), model, optimizer)

test_acc = 0.0

for i, (images, labels) in enumerate(valDataloader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_acc += get_accuracy(outputs, labels, batchSize)
        
print('Validation Accuracy: %.2f'%( test_acc/i))
