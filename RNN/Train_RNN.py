from comet_ml import Experiment
from comet_ml.utils import safe_filename
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable, variable
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
from scipy.special import inv_boxcox
import os 
import random
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Set seeds for all processes
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Hyperparameters
hyper_params={
    "alpha" : 0.005,
    "dropout":0,
    "hidden_size" : 64,
    "n_epochs" : 30, 
    "num_layers" : 1,
    "input_size" : 6,
    "sequence_length" : 92,
    "batch_size" : 1000,
}

#Import data
Raw_data = pd.read_csv("../Data/Raw_data.csv",index_col=0)
Training_df = pd.read_csv("../Data/Train.csv",index_col=0)
Val_df = pd.read_csv("../Data/Val.csv",index_col=0)
Test_df = pd.read_csv("../Data/Test.csv",index_col=0)

#Import data required to undo normalization
#import the list of min_maxs for normalization
min_max= []
with open('../../Data/Min_max.txt', 'r') as file:
    contents = file.readlines()
    for i in contents:
        i.strip("''")
        line = i[1:-3].split(",")
        min_max.append(list(line))
for i in range(len(min_max)):
    for j in range(len(min_max[i])):
        if j >= 1:
            min_max[i][j] = float(min_max[i][j])
        else:
            min_max[i][j] = min_max[i][j].strip("'")
            
#import the list of lambdas
lambda_list = []
with open('../../Data/Lambda.txt', 'r') as file:
    contents = file.readlines()
    for i in contents:
        i.strip("''")
        line = i[1:-3].split(",")
        lambda_list.append(list(line))
for i in range(len(lambda_list)):
    for j in range(len(lambda_list[i])):
        if j >= 1:
            lambda_list[i][j] = float(lambda_list[i][j])
        else:
            lambda_list[i][j] = lambda_list[i][j].strip("'")

#Import Trend Data
Trend = pd.read_csv("../Data/Trend.csv",index_col=0)

#Prepare Validation and Test dataset
#Add the last 92 entries from Training to Validation. 
Val_df2 = Training_df.iloc[-99:,:].append(Val_df)

#Add the last 92 entries from Validation to Test. 
Test_df2 = Val_df.iloc[-99:,:].append(Test_df)

#Prepare training data into data loader
class Sequential_Data(Dataset):
    def __init__(self, data, window):
        self.data = torch.Tensor(data.values)
        self.window = window
        self.shape = self.__getshape__()
        self.size = self.__getsize__()
 
    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.data[index+6+self.window,3]
        return x, y
 
    def __len__(self):
        return len(self.data) -  self.window -7
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())

#Load in data
Training_data = Sequential_Data(Training_df,hyper_params['sequence_length'])
Validation_data = Sequential_Data(Val_df2,hyper_params['sequence_length'])
Test_data = Sequential_Data(Test_df2,hyper_params['sequence_length'])

# Load into a data loader 
trainloader = DataLoader(dataset=Training_data,batch_size=300,shuffle=False,num_workers=0)
valloader = DataLoader(dataset=Validation_data,batch_size=300,shuffle=False,num_workers=0)
testloader = DataLoader(dataset=Test_data,batch_size=len(Test_data),shuffle=False,num_workers=0)    

#Define RNN model
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,drop_p):
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop_p = drop_p
        self.fc = nn.Linear(in_features=hidden_size,out_features=1)
        #input = [batch_size,number_in_seq,num_features]
    
    def forward(self,x):
        #initialise
        h0 = Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device))
        # h_initial = [number_of_layers,bath_size,hidden_size] -> [1,100,128]
        #Proporgate
        out, not_needed = self.rnn(x,h0)
        
        #retrieve last hidden state:
        last_out = out[:,-1,:] #[n,128]
        
        #add non-linearity
        #non_linear = self.sig(last_out)
        non_linear = self.relu(last_out)

        #collapse the output to a prediction
        prediction = self.fc(non_linear)
        #reshape the output 
        prediction = torch.reshape(prediction,(last_out.size()[0],))
        return prediction

#SMAPE 
#Calculate accuracy of using the symmetric mean absolute percentage error
def SMAPE(Predicted, Actual):
    try:
        n= len(Predicted)
    except:
        n= 1
    numer = abs(Predicted-Actual)
    denom = (abs(Actual) + abs(Predicted))*0.5
    return (100/n) * (np.sum(numer/denom))

def to_raw(Results,Data):
    if type(Results) == torch.Tensor:
        x = np.array(Results.cpu().detach())
    else :
        x = np.array(Results.cpu())
    Un_normal = (x*(min_max[3][2]-min_max[3][1])) + min_max[3][1]
    
    if Data=="Val":
        Seasonal = Un_normal + np.array(Trend.iloc[-400:-200,3])
    elif Data == "Test": 
        Seasonal = Un_normal + np.array(Trend.iloc[-200:,3])
    else:
        Seasonal = Un_normal + np.array(Trend.iloc[97:-401,3])
    
    Non_box_cox = inv_boxcox(Seasonal,lambda_list[3][1])
    return Non_box_cox

#Set seeds for all processes
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = RNN(input_size=hyper_params['input_size'],
                                    hidden_size=hyper_params['hidden_size'],
                                    num_layers=hyper_params['num_layers'],
                                    drop_p=hyper_params['dropout']).to(device)

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(),lr=hyper_params['alpha'])

#Train model 
#make sure the model parameters are reset before training
#with experiment.train():
model.train()
print("alpha: ", hyper_params['alpha'], "dropout: ",hyper_params['dropout'],"hidden: ",
      hyper_params['hidden_size'],'layers: ',hyper_params['num_layers'])
for epoch in range(hyper_params['n_epochs']):
    with torch.no_grad():
        for j, (val_sequence, val_forecast) in enumerate(valloader): 
        # Forward pass
            val_sequence=Variable(val_sequence).to(device)
            val_forecast = Variable(val_forecast).to(device)
            val_outputs = model(val_sequence)
            Val_loss = criterion(val_outputs,val_forecast)
    #experiment.log_metric("Validation Loss",Val_loss,step = epoch)

    for i, (train_sequence, train_forecast) in enumerate(trainloader): 
        train_sequence = Variable(train_sequence).to(device)
        train_forecast = Variable(train_forecast).to(device)

        # Forward pass
        opt.zero_grad() #zero gradients
        outputs = model(train_sequence) #input sequence into model
        loss = criterion(outputs,train_forecast) # calculate MSE 
        #experiment.log_metric("loss",loss,step = epoch) #log the results 

        # Backward and optimize
        loss.backward()
        opt.step()
        opt.zero_grad()


    print('Epoch [%d/%d], Loss: %.4f'
        % (epoch + 1, hyper_params['n_epochs'],  loss.item()))

model.eval()
# Test model on data
Results_list =[]
for i, (val_sequence, val_forecast) in enumerate(valloader): 
    val_sequence = Variable(val_sequence).to(device)
    val_forecast = Variable(val_forecast).to(device)

    val_out = model(val_sequence)
    Results_list.append(np.array(val_out.cpu().detach()))

All_results = np.concatenate(Results_list, axis=0)
Predicted = to_raw(Results=val_out, Data="Val")
Actual =np.array(Raw_data.iloc[-400:-200,3])
Val_SMAPE = SMAPE(Actual=Actual,Predicted=Predicted)
#experiment.log_metric("Validation Accuracy",Val_SMAPE)

for j, (test_sequence, test_forecast) in enumerate(testloader): 
    test_sequence = Variable(test_sequence).to(device)
    test_forecast = Variable(test_forecast).to(device)

    Test_out = model(test_sequence)
    Predicted = to_raw(Results=Test_out, Data= "Test")
    Actual =np.array(Raw_data.iloc[-200:,3])
    Test_SMAPE = SMAPE(Actual=Actual,Predicted=Predicted)
    #experiment.log_metric("Test Accuracy",Test_SMAPE)
print("Test Accuracy: ",Test_SMAPE)

#save weights for analysis
torch.save(model.state_dict(), "../Weights/RNN.pt")