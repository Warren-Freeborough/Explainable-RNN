#import libraries
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
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
import random
from captum.attr import IntegratedGradients
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
import itertools
import os
#set device to use gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
#RNN Parameters
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
# Import data
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Import data
Raw_data = pd.read_csv("../Data/Raw_data.csv",index_col=0)
Training_df = pd.read_csv("../Data/Train.csv",index_col=0)
Val_df = pd.read_csv("../Data/Val.csv",index_col=0)
Test_df = pd.read_csv("../Data/Test.csv",index_col=0)

#Import data required to undo normalization
#import the list of min_maxs for normalization
min_max= []
with open('../Data/Min_max.txt', 'r') as file:
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
with open('../Data/Lambda.txt', 'r') as file:
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
Val_df2 = Training_df.iloc[-(hyper_params['sequence_length']+7):,:].append(Val_df)

#Add the last 92 entries from Validation to Test. 
Test_df2 = Val_df.iloc[-(hyper_params['sequence_length']+7):,:].append(Test_df)

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
trainloader = DataLoader(dataset=Training_data,batch_size=hyper_params['batch_size'],shuffle=False,num_workers=0)
valloader = DataLoader(dataset=Validation_data,batch_size=hyper_params['batch_size'],shuffle=False,num_workers=0)

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
        Seasonal = Un_normal + np.array(Trend.iloc[hyper_params['sequence_length']+5:-401,3])
    
    Non_box_cox = inv_boxcox(Seasonal,lambda_list[3][1])
    return Non_box_cox


#Define RNN model
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
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


model = RNN(input_size=hyper_params['input_size'],
                                    hidden_size=hyper_params['hidden_size'],
                                    num_layers=hyper_params['num_layers']).to(device)
criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(),lr=hyper_params['alpha'])

#Load model
model.load_state_dict(torch.load("../Weights/RNN.pt"))
model.eval()
#Determine accuracy for each dataset
trainloader = DataLoader(dataset=Training_data,batch_size=len(Training_data),shuffle=False,num_workers=0)
valloader = DataLoader(dataset=Validation_data,batch_size=len(Validation_data),shuffle=False,num_workers=0)
testloader = DataLoader(dataset=Test_data,batch_size=len(Test_data),shuffle=False,num_workers=0)

#Make predictions for test dataset. This will be used as the benchmark
for i, (test_sequence, test_forecast) in enumerate(testloader): 
    test_sequence = Variable(test_sequence).to(device)
    test_forecast = Variable(test_forecast).to(device)

    Test_out = model(test_sequence)
    Predicted = to_raw(Results=Test_out, Data= "Test")

#____________________________________________________________________________________
#Ablation
#Prepare training data into data loader
def Ablate(DF,Feature,Start,Window):
    Copy = DF.copy()
    for i in range(Window):
        x= np.mean(DF.iloc[Start-1+i-92:Start-1+i,Feature])
        Copy.iloc[Start+i,Feature] = x
    
    return np.array(Copy.iloc[:,Feature])

#Prepare training data into data loader
class Ablated_Data(Dataset):
    def __init__(self, Original_Data, Ablated_data, window):
        self.Adata = torch.Tensor(Ablated_data.values)
        self.Odata= torch.Tensor(Original_Data.values)
        self.window = window
        self.shape = self.__getshape__()
        self.size = self.__getsize__()
 
    def __getitem__(self, index):
        x = self.Adata[index:index+self.window]
        y = self.Odata[index+6+self.window,3]
        return x, y
 
    def __len__(self):
        return len(self.Adata) -  self.window -7
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())

#Create error array
Ablation_Error_array = np.zeros((115,6,92))
Start = 93
Window = 1
for i in range(6):
    for j in range(115):
            Start = 93+i
            Complete_ablated = Test_df2.copy()
            ablate_column = Ablate(Test_df2,i,Start,Window)
            Complete_ablated.iloc[:,i] = ablate_column.copy()
            #create loader
            Ablate_Dataset = Ablated_Data(Original_Data=Test_df2, Ablated_data= Complete_ablated,window =hyper_params['sequence_length'])
            Ab_loader = DataLoader(dataset=Ablate_Dataset,batch_size=len(Ablate_Dataset),shuffle=False,num_workers=0)
            
            for k, (test_sequence, test_forecast) in enumerate(Ab_loader): 
                test_sequence = Variable(test_sequence).to(device)
                test_forecast = Variable(test_forecast).to(device)

                Ab_out = model(test_sequence)
                Predicted_Ab = to_raw(Results=Ab_out, Data= "Test")

            #Error
            Feature_errors = np.zeros(92)
            Errors = abs(Predicted_Ab-Predicted)/Predicted
            point= Start-91
            if point +92 <= 299:
                to_array = np.flip(Errors[point:point+92],-1)
            else:
                to_array = np.flip(Errors[point:],-1)
            Feature_errors = np.append(np.zeros(92-len(to_array)),to_array)

            #Append to the empty array
            Ablation_Error_array[j][i]= Feature_errors
#average the errors over the 0 axis and plot
Avg_Ablation_Error = np.mean(Ablation_Error_array,axis=0)
fig, ax = plt.subplots(figsize=(20,5)) 
ax = sns.heatmap(np.log(Avg_Ablation_Error),vmin=-14,vmax=0,cmap="cubehelix",yticklabels=Test_df.columns)
fig.savefig('Results\RNN_Ablation_log.pdf')

fig2, ax2 = plt.subplots(figsize=(20,5)) 
ax2 = sns.heatmap(Avg_Ablation_Error,cmap="Blues",vmin=0,vmax=0.015,yticklabels=Test_df.columns)
fig2.savefig('Results\RNN_Ablation.pdf')
print("Finished Ablation")

#_________________________________________________________________________________________________________________
#Noise 
#Intoduces noise into a specific cell in the neural network
class Noise_Window_Data(Dataset):
    def __init__(self, data, window,cell):
        self.data = torch.Tensor(data.values)
        self.window = window
        self.cell = cell
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

 
    def __getitem__(self, index):
        Copy = self.data.clone()
        x = Copy[index:index+self.window]
        x[self.cell] = x[self.cell] + x[self.cell]*0.01
        y = self.data[index+6+self.window,3]
        return x, y
 
    def __len__(self):
        return len(self.data) -  self.window -7
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())

Cell_SMAPE = np.zeros(92)
for cell in range(92):
    Noise_1_data= Noise_Window_Data(Test_df2,92,cell)
    Noise1loader = DataLoader(dataset=Noise_1_data,batch_size=len(Noise_1_data),shuffle=False,num_workers=0)

    for j, (test_sequence, test_forecast) in enumerate(Noise1loader): 
        test_sequence = Variable(test_sequence).to(device)
        test_forecast = Variable(test_forecast).to(device)

        Test_out = model(test_sequence)
        Predicted_Noise_window = to_raw(Results=Test_out, Data= "Test")
        Total_SMAPE = SMAPE(Actual=Predicted,Predicted=Predicted_Noise_window)
    Cell_SMAPE[cell]= Total_SMAPE

#Average the errors 
fig3, ax = plt.subplots(figsize=(20,5)) 
ax = sns.heatmap(np.reshape(Cell_SMAPE,(1,92)),vmin=0,vmax=0.45,cmap="Blues")
plt.xlabel("Day") 
fig3.savefig('Results\\RNN_Noise.pdf')

fig4, ax = plt.subplots(figsize=(20,5)) 
ax = sns.heatmap(np.log(np.reshape(Cell_SMAPE,(1,92))),vmax=0,cmap="cubehelix",yticklabels="")
plt.xlabel("Day") 
fig4.savefig('Results\\RNN_Noise_log.pdf')
print("Finished Noise")
#_________________________________________________________________________________________________________________
#Permutation
#Define function to permute datasets
def Permute(DF,Feature,Start=1,End=298):
    COPY = DF.copy()
    for i in np.arange(End,Start,-1):
        rs =random.sample(range(i),1)
        if Feature == 6:
            COPY.iloc[i,:] = DF.iloc[rs,:]
        else:
            COPY.iloc[i,Feature] = DF.iloc[rs,Feature]
    
    return COPY   
#
class Permuted_Data(Dataset):
    def __init__(self, Original_Data, Permuted_data, window):
        self.Pdata = torch.Tensor(Permuted_data.values)
        self.Odata= torch.Tensor(Original_Data.values)
        self.window = window
        self.shape = self.__getshape__()
        self.size = self.__getsize__()
 
    def __getitem__(self, index):
        x = self.Pdata[index:index+self.window]
        y = self.Odata[index+6+self.window,3]
        return x, y
 
    def __len__(self):
        return len(self.Pdata) -  self.window -7
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())
#
combinations =[]
for i in np.arange(1,7):
    x = itertools.combinations([0,1,2,3,4,5],i)
    for j in x:
        combinations.append(list(j))

epochs = 100
X = []
Y = []
Combo_Error_all =[]
for combo in combinations:
    Avg_Error_for_combo =[]
    for i in range(epochs):
        #print("combo: ",combo)
        Combined = Test_df2.copy()
        Permuted_DF = Permute(Test_df2,6)
        #for each feature in list, change the features to the permuted
        for feature in combo:
            Combined.iloc[:,feature] = Permuted_DF.iloc[:,feature]
        #create a dataset
        Permute_Dataset = Permuted_Data(Original_Data=Test_df2, Permuted_data= Combined,window =hyper_params['sequence_length'])
        P_loader = DataLoader(dataset=Permute_Dataset,batch_size=len(Permute_Dataset),shuffle=False,num_workers=0)
        #test the permuted accuracy against the unpermuted 
        for k, (test_sequence, test_forecast) in enumerate(P_loader): 
                test_sequence = Variable(test_sequence).to(device)
                test_forecast = Variable(test_forecast).to(device)

                Permute_out = model(test_sequence)
                Predicted_Per = to_raw(Results=Permute_out, Data= "Test")
                Per_SMAPE = SMAPE(Actual=Predicted,Predicted=Predicted_Per)
        #create OHE dataset for which columns are permuted
        OHE = [0,0,0,0,0,0]
        for feature in combo:
            OHE[feature]=1
        X.append(OHE)
        Y.append(Per_SMAPE)

#create DF
Permuted_df2 = pd.concat([pd.DataFrame(X),pd.DataFrame(Y)],axis=1, join="inner")
Permuted_df2.columns = ["High","Low","Open","Close","Volume","Adj_closed","Error"]
y2, x2 = dmatrices('Error ~ High + Low + Open + Close + Volume + Adj_closed -1' , data=Permuted_df2, return_type='dataframe')

#fit a linear regression to one hot encoded data
mod = sm.OLS(y2, x2)    
res = mod.fit()       
importance = res.params
#Plot feature importance
# summarize feature importance
# plot feature importance
plt.rcdefaults()
fig5, ax = plt.subplots()
features = Permuted_df2.columns
ax.barh([x for x in range(len(importance))], 100*(importance/np.sum(importance)))
y = np.arange(len(features))
ax.set_yticks(y)
ax.set_yticklabels(features)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Percent of error contributed')
ax.set_title('Percent of error contributed by permuting each feature')
fig5.savefig('Results\RNN_Permutation.pdf')
print("Finished Permutation")

#________________________________________________________________________________________________________________
#Intergrated 
#Create mean baseline 
def Base_mean(DF):
    COPY = DF.copy()
    for i in np.arange(298,93,-1):
        COPY.iloc[i,:] = np.mean(np.array(Test_df2.iloc[i-92:i-1,:]),axis = 0)
    return COPY    

model.train()
Mean_df = Base_mean(Test_df2)
Mean_data = Sequential_Data(Mean_df,hyper_params['sequence_length'])
Mean_loader = DataLoader(dataset=Mean_data,batch_size=len(Mean_data),shuffle=False,num_workers=0)
Mean_baseline = next(iter(Mean_loader))[0]
Mean_baseline = Mean_baseline[1:].to(device)
Mean_input = next(iter(testloader))[0][1:].to(device)

Mean_loader = DataLoader(dataset=Mean_data,batch_size=len(Mean_data),shuffle=False,num_workers=0)
Mean_baseline = next(iter(Mean_loader))[0]
Mean_baseline = Mean_baseline[1:].to(device)
Mean_input = next(iter(testloader))[0][1:].to(device)

#Set up attributions 
ig = IntegratedGradients(model,multiply_by_inputs=True)
at_mean, mean_error = ig.attribute(Mean_input,
                                         baselines=Mean_baseline,
                                         method='gausslegendre',
                                         return_convergence_delta=True)

at_mean = np.mean(np.array(at_mean.cpu()),axis=0)

fig6 = plt.figure(figsize=(20,3))
ax = sns.heatmap(np.log(at_mean.T),cmap="cubehelix",vmax=0,vmin=-35,yticklabels=Test_df2.columns)
fig6.savefig('Results\RNN_Attributions_log+.pdf')

fig7 = plt.figure(figsize=(20,3))
ax = sns.heatmap(np.log(-1*at_mean.T),cmap="cubehelix",vmax=0,vmin=-35,yticklabels=Test_df2.columns)
fig7.savefig('Results\RNN_Attributions_log-.pdf')

fig8 = plt.figure(figsize=(20,3))
ax = sns.heatmap(at_mean.T,cmap="Blues",yticklabels=Test_df2.columns)
fig8.savefig('Results\RNN_Attributions.pdf')

fig9 = plt.figure(figsize=(20,3))
ax = sns.heatmap(np.log(np.abs(at_mean.T)),cmap="cubehelix", vmin=-35,yticklabels=Test_df2.columns)
fig9.savefig('Results\RNN_Attributions_log_abs.pdf')
print("Finished Attributions")