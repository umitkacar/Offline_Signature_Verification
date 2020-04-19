# General
from os import path
from random import randrange
from sklearn.model_selection import train_test_split
import pickle

# Pytorch
import torch
from torch.nn import Linear, Conv2d, MaxPool2d, LocalResponseNorm, Dropout
from torch.nn.functional import relu
from torch.nn import Module
from torch.tensor import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Pillow
from PIL import Image
from PIL.ImageOps import invert

# Numpy
import numpy as np

# Custom
from utils import fix_pair_sign, fix_pair_person
from Model import SiameseConvNet, ContrastiveLoss, distance_metric
from Dataset import TrainDataset

PERSON_NUMBER = 79
SIGN_NUMBER_EACH = 12

if __name__ == "__main__":
    
    # version control
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device = " + str(device))    
    
    # dataset path
    img_path = './Data_raw/genuines/NFI-%03d%02d%03d.png'

    # preparing data for training model
    data = []
    n_samples_of_each_class = 10000
    prefix ='./Data/'
    count = 0
    for i in range(n_samples_of_each_class):
        
        # positive data
        anchor_person = randrange(1, PERSON_NUMBER+1)
        anchor_sign = randrange(1, SIGN_NUMBER_EACH+1)
        pos_sign = randrange(1, SIGN_NUMBER_EACH+1)
        anchor_sign, pos_sign = fix_pair_sign(anchor_sign, pos_sign)
        positive = [img_path%(anchor_person, anchor_sign, anchor_person), img_path%(anchor_person, pos_sign, anchor_person), 1]
        
        if path.exists(positive[0]) and path.exists(positive[1]):
            data.append(positive)
        else:
            count +=1
            print(positive)
        
        # Negative data
        neg_person = randrange(1, PERSON_NUMBER+1)
        neg_sign = randrange(1, SIGN_NUMBER_EACH+1)
        anchor_person, neg_person = fix_pair_person(anchor_person, neg_person)
        negative = [img_path%(anchor_person, anchor_sign, anchor_person), img_path%(neg_person, neg_sign, neg_person), 0]
        
        if path.exists(negative[0]) and path.exists(negative[1]):
            data.append(negative)
        else:
            count +=1
            print(negative)
    print(count)
    input()       
    # Split train & test
    train, test = train_test_split(data, test_size=0.05)
    
    with open('./Data/train_index.pkl', 'wb') as train_index_file:
        pickle.dump(train, train_index_file)  
    
    with open('./Data/test_index.pkl', 'wb') as test_index_file:
        pickle.dump(test, test_index_file)
        
    # Hyper parameters
    num_epochs = 5
    batch_size = 8
    learning_rate = 0.001
    
    # model, loss and optimizer
    model = SiameseConvNet().to(device)
    criterion = ContrastiveLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    train_dataset = TrainDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  
    # training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        for index, (images1, images2, labels) in enumerate(train_loader):
            # images.size = 8, 1, 220, 155
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.float().to(device)
            
            # forward and loss
            features1, features2 = model.forward(images1, images2)
            loss = criterion(features1, features2, labels)
            total_loss += loss.item()
            # backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (index+1) % 1 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {index+1}/{n_total_steps}, loss = {loss.item():.4f}', end='\r')
        
        average_loss = total_loss / (len(train_dataset) // batch_size)
        print(f'Average loss {average_loss:.4f}')
    
        # model save - checkpoint        
        model_save_path = "./Models/checkpoint_epoch_%d" % epoch
        with open(model_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)