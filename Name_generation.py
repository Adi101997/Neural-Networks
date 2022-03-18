#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


# In[2]:


names=[]
#Please change the path for the data file according to your folder structure
file=open("C:/UIC_CS/NN/Homework/names.txt","r")
for line in file:
    names.append(line.strip('\n').lower())


# In[3]:


encode=[]
one_hot_encode_dict={}
alphabets = ['EON','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                 'p','q','r','s','t','u','v','w','x','y','z']
char_dict = dict(enumerate(alphabets))
for char in char_dict.keys():
    encode.append(char)
one_hot=F.one_hot(torch.tensor(encode), num_classes=27)
for alphabet, oh in zip(alphabets,one_hot):
    one_hot_encode_dict[alphabet]=oh


# In[4]:


break_name_list=[]
for name in names:
    break_name_list.append([char for char in name])
for i in range(0, len(break_name_list)):
    if len(break_name_list[i])<11:
        break_name_list[i].extend((11-len(break_name_list[i]))*["EON"])


# In[5]:


def char2int(char_dict):
    char_int_dict={}
    for key, value in char_dict.items():
        char_int_dict[value]=key
    return char_int_dict


# In[6]:


char_int_dict=char2int(char_dict)


# In[7]:


def encoding_data(break_name_list):
    data_x=torch.zeros((11,27))
    single_input_x=[]
    for c in break_name_list:
        single_input_x.append(one_hot_encode_dict[c])
    for i in range(11):
        for j in range(27):
            data_x[i][j]=single_input_x[i][j]
    
    char_int_dict=char2int(char_dict)
    y_data=[]
    single_input_y=[]

    y=break_name_list[1:]
    y.append("EON")
    for c_y in y:
        single_input_y.append(char_int_dict[c_y])
    data_y=torch.zeros((11))
    for m in range(11):
        data_y[m]=single_input_y[m]
    return data_x,data_y


# In[8]:


class custom_dataset_dataloader(torch.utils.data.Dataset):
    def __init__(self,break_name_list):
        self.break_name_list=break_name_list
        
    def __len__(self):
        return len(self.break_name_list)

    def __getitem__(self, idx: int):
        return encoding_data(break_name_list[idx])
    


# In[9]:


train_data = custom_dataset_dataloader(break_name_list)


# In[10]:


train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)


# In[11]:


epochs = 60
epochs_list=np.arange(1,epochs+1)
learning_rate = 0.01
gamma = 0.95
input_size = 27
hidden_size = 108
output_size = 27
number_layers = 2
step_size = 100


# In[12]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


# In[13]:


class Model(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, number_layers):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=number_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        
    def forward(self, X_data, states):
        h_state, c_state = states
        batch_size = X_data.size(0)
        output, (h_state, c_state) = self.lstm1(X_data, (h_state, c_state))
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        return output, (h_state, c_state) 


# In[14]:


model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, number_layers=number_layers)
model = model.to(device)


# In[15]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


# In[16]:


def train(train_loader):
    epoch_loss = 0
    epoch_learning_rate = 0
    
    for i, (X_data, Y_data) in enumerate(train_loader):
        X_data, Y_data = X_data.to(device), Y_data.to(device)
        
        hidden_state = torch.zeros((number_layers, X_data.size(0), hidden_size)).to(device)
        out_state = torch.zeros((number_layers, X_data.size(0), hidden_size)).to(device)

        optimizer.zero_grad()
        y_pred, (hidden_state, out_state) = model(X_data, (hidden_state, out_state))
        y_pred = y_pred.transpose(1, 2)
        loss = criterion(y_pred, Y_data.long())
        loss.backward(retain_graph=True)
        optimizer.step()
        
        epoch_loss = epoch_loss+loss.item()
        epoch_learning_rate = epoch_learning_rate+scheduler.get_last_lr()[0]
        scheduler.step()
    return epoch_loss, epoch_learning_rate


# In[17]:


iter_losses = []
epoch_loss_list = []
iter_learning_rates = []

for epoch in range(1, epochs+1):
    epoch_loss, epoch_learning_rate=train(train_loader) 
    epoch_loss = epoch_loss/len(train_loader)
    epoch_loss_list.append(epoch_loss)
    epoch_learning_rate = epoch_learning_rate/len(train_loader)
    
    print("Epoch:", epoch, " Loss:", epoch_loss, " Learning Rate:", epoch_learning_rate)
    
torch.save(model.state_dict(), "0702-670099560-Chaudhry.pt")


# In[18]:


plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs_list,epoch_loss_list)
plt.show()


# In[ ]:




