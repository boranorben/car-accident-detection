# -*- coding: utf-8 -*-
"""socar_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mq6GItnioAXBm9pQQGFNDc0beST1ZOex
"""

# mount
from google.colab import drive
drive.mount('/content/drive')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import pandas as pd 
from fnmatch import fnmatch
from tqdm import tqdm
import argparse
import time
import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt

# preprocessing
root = '/content/drive/MyDrive/Colab Notebooks/쏘카/'
pattern = "*-mp4-acc.csv"
label = root + 'task1/data_set_01_labeling_result.csv'

# label
df_label = pd.read_csv(label, encoding='cp949', index_col=False)
df_label = df_label.drop('Unnamed: 2', axis=1)

# acc
acclists = []
for path, subdirs, files in os.walk(root + 'task1/'):
    for name in files:
        if fnmatch(name, pattern):
          acclists.append(os.path.join(path, name))

files, x_list, y_list, z_list = [], [], [], []

for acclist in acclists:
  file = acclist[76 : 105]
  file = file.replace('-', '.')
  files.append(file)

  data_acc = pd.read_csv(acclist, index_col=False)
  x_list.append(data_acc.x.tolist()[1:])
  y_list.append(data_acc.y.tolist()[1:])
  z_list.append(data_acc.z.tolist()[1:])

dict_x = {'file': files, 'x': x_list}
dict_y = {'file': files, 'x': y_list}
dict_z = {'file': files, 'x': z_list}

df_x = pd.DataFrame(dict_x)
df_y = pd.DataFrame(dict_y)
df_z = pd.DataFrame(dict_z)

df_x = pd.merge(df_x, df_label, how='inner', on=['file'])
df_y = pd.merge(df_y, df_label, how='inner', on=['file'])
df_z = pd.merge(df_z, df_label, how='inner', on=['file'])

root_preproc = root + 'preprocessed/'

df_x.to_csv(root_preproc + 'acc_x.csv', index=False)
df_y.to_csv(root_preproc + 'acc_y.csv', index=False) 
df_z.to_csv(root_preproc + 'acc_z.csv', index=False)

def preproc(data_path):
  data_csv = pd.read_csv(data_path, engine='python')
  labels_csv = [int(i) for i in data_csv['accident']]
  proc_data = []
  labels = []

  idx = 0

  for d in data_csv['x']:
    dat = d[1:-1].split(', ')
    dat = [float(x) for x in dat]
    if (len(dat) == 118):
      dat = torch.FloatTensor(dat)
      proc_data.append(dat)
      labels.append(labels_csv[idx])
    idx += 1

  return proc_data, labels

class CarDataloader(Dataset):

  def __init__(self, transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.data_x, _  = preproc(root_preproc + 'acc_x.csv')
    self.data_y, _  = preproc(root_preproc + 'acc_y.csv')
    self.data_z, _  = preproc(root_preproc + 'acc_z.csv')
    _, self.labels = preproc(root_preproc + 'acc_x.csv')
      
  
  def get_num_of_classes(self):
    return 2

  def __len__(self):
    return len(self.data_x)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    acc_x = self.data_x[idx]
    acc_y = self.data_y[idx]
    acc_z = self.data_z[idx]
    label = self.labels[idx]

    data = torch.cat((acc_x, acc_y, acc_z))
    data = torch.reshape(data, (3,118))

    return data, label

n_timesteps, n_features = 3, 118

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 128
num_layers = 2
batch_size = 4
num_epochs = 100
learning_rate = 0.01

input_size = 118
num_classes = 2

car_dataloader = CarDataloader()

torch.manual_seed(0)
torch.manual_seed(torch.initial_seed())
train_set, val_set = torch.utils.data.random_split(car_dataloader,
                                                   [int(0.7 * len(car_dataloader)), int(0.3 * len(car_dataloader))])

train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=batch_size,
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    # Set initial hidden and cell states 
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])
    return out

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(3, 3, 3)
    self.conv2 = nn.Conv1d(3, 6, 5)
    self.fc1 = nn.Linear(672, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))

    x = torch.flatten(x, 1)  # flatten all dimensions except batch

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()

class Net2(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(3, 3, 3)
    self.conv2 = nn.Conv1d(3, 6, 5)
    self.fc1 = nn.Linear(672, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)
    self.PRelu = nn.PReLU()

  def forward(self, x):
    x = self.PRelu(self.conv1(x))
    x = self.PRelu(self.conv2(x))

    x = torch.flatten(x, 1)  # flatten all dimensions except batch

    x = self.PRelu(self.fc1(x))
    x = self.PRelu(self.fc2(x))
    x = self.fc3(x)
    return x

class Net3(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(3, 3, 3)
    self.conv2 = nn.Conv1d(3, 6, 5)
    self.fc1 = nn.Linear(672, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)
    self.acc = nn.Tanh()

  def forward(self, x):
    x = self.acc(self.conv1(x))
    x = self.acc(self.conv2(x))

    x = torch.flatten(x, 1)  # flatten all dimensions except batch

    x = self.acc(self.fc1(x))
    x = self.acc(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net3()

class Net23(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv1d(3, 3, 3)
    self.conv2 = nn.Conv1d(3, 6, 5)
    self.fc1 = nn.Linear(672, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)
    self.acc = nn.Tanh()
    self.acc2 = nn.PReLU()

  def forward(self, input):
    x = self.acc(self.conv1(input))
    y = self.acc2(self.conv1(input))

    x = torch.add(x, y)

    y = self.acc2(self.conv2(x))
    x = self.acc(self.conv2(x))

    x = torch.add(x, y)

    x = torch.flatten(x, 1)  # flatten all dimensions except batch

    y = self.acc2(self.fc1(x))
    x = self.acc(self.fc1(x))

    x = torch.add(x, y)

    y = self.acc(self.fc2(x))
    x = self.acc(self.fc2(x))

    x = torch.add(x, y)
    x = self.fc3(x)

    return x

# Recurrent neural network (many-to-one)
class Combine(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(Combine, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

    self.conv1 = nn.Conv1d(3, 3, 3)
    self.conv2 = nn.Conv1d(3, 6, 5)
    self.fc1 = nn.Linear(672, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)

  def forward(self, x):
    # Set initial hidden and cell states 
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    out = torch.add(out, x)

    return out

# Recurrent neural network (many-to-one)
class Combine2(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(Combine2, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

    self.conv1 = nn.Conv1d(3, 3, 3)
    self.conv2 = nn.Conv1d(3, 6, 5)
    self.fc1 = nn.Linear(672, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)
    self.PRelu = nn.PReLU()

  def forward(self, x):
    # Set initial hidden and cell states 
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

    # Forward propagate LSTM
    out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])

    x = self.PRelu(self.conv1(x))
    x = self.PRelu(self.conv2(x))

    x = torch.flatten(x, 1)  # flatten all dimensions except batch

    x = self.PRelu(self.fc1(x))
    x = self.PRelu(self.fc2(x))
    x = self.fc3(x)

    # out = F.relu(out)
    # x = F.relu(x)

    out = torch.add(out, x)

    return out

# model = RNN(input_size, hidden_size, 4, num_classes).to(device)
# model = Net().to(device)
# model = Net2().to(device)
# model = Net3().to(device)
# model = Net23().to(device)
# model = Combine(input_size, hidden_size, num_layers, num_classes).to(device)
model = Combine2(input_size, hidden_size, num_layers, num_classes).to(device)

# Load model
# root_model = root + 'models/'

# model = torch.load(root_model + 'best_model_9583.pt')
# model = model.to(device)

num_epochs = 100

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_total_step, test_total_step = len(train_dataloader), len(test_loader)
train_values, test_values = [], []
train_loss, test_loss = 0.0, 0.0
for epoch in range(num_epochs):
  for step, (acc_data, labels) in enumerate(train_dataloader):
    acc_data = acc_data.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(acc_data)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % (batch_size) == 0:
      # train_loss = loss.item() * acc_data.size(0)
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, num_epochs, step + 1, train_total_step, loss.item()))

  # for step, (acc_data, labels) in enumerate(test_loader):
  #   acc_data = acc_data.to(device)
  #   labels = labels.to(device)

  #   # Forward pass
  #   outputs = model(acc_data)
  #   loss = criterion(outputs, labels)

  #   # Backward and optimize
  #   optimizer.zero_grad()
  #   loss.backward()
  #   optimizer.step()

  #   if (step + 1) % (batch_size) == 0:
  #     test_loss = loss.item() * acc_data.size(0)

  # train_values.append(train_loss / train_total_step)
  # test_values.append(test_loss / test_total_step)

# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.plot(train_values, label='train')
# plt.plot(test_values, label='test')
# plt.legend()
# plt.show()

# Test the model
model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for acc_data, labels in test_loader:
      acc_data = acc_data.to(device)
      labels = labels.to(device)
      outputs = model(acc_data)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Test Accuracy : {} %'.format(100 * correct / total)) 

# Save the model checkpoint
# torch.save(model.state_dict(), root_model + 'model.ckpt')