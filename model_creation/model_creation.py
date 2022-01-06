import pandas as pd
import random
import csv
import numpy as np

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class CustomSKDataset(Dataset):
    def __init__(self, csv_file):  
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = list(data.loc[idx, ["a1", "o1", "c1", "g1", "l1", "b1", "f1", "t1", "j1", "u1", "q1", "a2", "o2", "c2", "g2", "l2", "b2", "f2", "t2", "j2", "u2", "q2","a3", "o3", "c3", "g3", "l3", "b3", "f3", "t3", "j3", "u3", "q3"]])
        y = list(data.loc[idx, ["Basis.txt", "jaw_open.txt", "left_eye_closed.txt", "mouth_open.txt", "right_eye_closed.txt", "smile.txt", "smile_left.txt", "smile_right.txt"]])     
        X = torch.flatten(torch.tensor(X).float()) 
        y = torch.flatten(torch.tensor(y).float())          
        # X et y sont des tenseurs flat
        return X, y

def data_normalization(y_train, y_test):
    values = []
    for i in range(len(y_train)):
        for y in range(len(y_train[i])):
            values.append(y_train[i][y])
    for i in range(len(y_test)):
        for y in range(len(y_test[i])):
            values.append(y_test[i][y])    
    mean = np.mean(values)
    st = np.std(values)
    for i in range(len(y_train)):
        for y in range(len(y_train[i])):
            y_train[i][y] = (y_train[i][y]-mean)/st
    for i in range(len(y_test)):
        for y in range(len(y_test[i])):
            y_test[i][y] = (y_test[i][y]-mean)/st
    return (y_train, y_test)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

if __name__ == "__main__": 
    phonemes = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2021/ALphSistant/data_3_1.csv")
    sk_weights = pd.read_csv("C:/Users/Enzo.Magal/Documents/Enzo2021/AlphSistant/ds_weights.csv")
    
    phonemes['ref'] = phonemes['record'] + '/' + phonemes['frame']
    sk_weights['ref'] = sk_weights['sequence'] + '/' + sk_weights['frame']
    phonemes.drop('frame', axis=1)

    data = pd.merge(phonemes, sk_weights)

    data.to_csv('sk_data.csv')

    sk_dataset = CustomSKDataset('sk_data.csv')

    dataloader = DataLoader(sk_dataset, batch_size=512, shuffle=True, num_workers=0)

    layer1 = nn.Linear(in_features=33, out_features=3000)
    layer2 = nn.Linear(in_features=3000, out_features=3000)
    model = nn.Sequential(
        layer1,
        nn.Tanh(),
        layer2,
        nn.Tanh(),
        layer2,
        nn.Tanh(),
        nn.Linear(3000, 8)
    )

    print("Model structure: ", model, "\n\n")

    learning_rate = 1e-3
    batch_size = 64

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, model, loss_fn, optimizer)
        test_loop(dataloader, model, loss_fn)
    print("Done!")

    torch.save(model, 'C:/Users/Enzo.Magal/Documents/Enzo2021/models/sk_model.pth')