import numpy as np
import pandas as pd
import torch.utils.data
from functools import partial
import matplotlib.pyplot as plt
import os
from network import *
#import res
import loggging
import config1 as c
import plotly
import pickle
import random
import copy
import scipy.io
from CustomLayers import *

# %% Load Data: 1. Train Test Split
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

Train_type =1

# %% Load Data

## 1. Train Test Splitting
if Train_type == 1:
    test_size = 0.1
    mat_file = loadmat('Data/NEWDataset.mat')
    X = mat_file['X']
    #PP = mat_file['PP_spl']
    #X = np.concatenate((X,PP), axis=1)
    Y = mat_file['Y']
    trainset_Raw_X, testset_Raw_X, trainset_Raw_Y, testset_Raw_Y = train_test_split(X, Y, test_size=test_size, random_state=1)
    trainset_Raw = np.concatenate((trainset_Raw_X, trainset_Raw_Y), axis=1)
    testset_Raw = np.concatenate((testset_Raw_X, testset_Raw_Y), axis=1)

else:
## 2. Extrapolation Spliting: quadrant-wise
    if Train_type == 2:
        quad = 1    # 1: one quadrant train remaining test; 2: two quadrants training remaining test
        if quad == 1:
            mat_file = loadmat('Data/Dataset13Train_NewPartialPhys_1quad.mat')
            test_size = "Quadrant 1 split"
        elif quad == 2:
            mat_file = loadmat('Data/Dataset13Train_NewPartialPhys.mat')
            test_size = "Quadrant 2 split"

## 3. Extrapolation Spliting: Radial split    
    elif Train_type == 3:
        mat_file = loadmat('Data/Dataset_radial.mat')
        test_size = "Radial 50% split"
    trainset_Raw_X = mat_file['train_X']
    #trainset_Raw_X_PP = mat_file['train_PP_spl']
    trainset_Raw_Y = mat_file['train_Y']
    trainset_Raw = np.concatenate((trainset_Raw_X, trainset_Raw_Y), axis=1)
    
    testset_Raw_X = mat_file['test_X']
    #testset_Raw_X_PP = mat_file['test_PP_spl']
    testset_Raw_Y = mat_file['test_Y']
    testset_Raw = np.concatenate((testset_Raw_X, testset_Raw_Y), axis=1)


# %% Normalization
dataset = np.concatenate((trainset_Raw,testset_Raw), axis=0)
lb_Out=np.min(dataset[:,3])
ub_Out=np.max(dataset[:,3])
mean_Out = np.mean(dataset[:,3])
trainset = copy.deepcopy(trainset_Raw)
trainset[:,3] = (((trainset_Raw[:,3] - lb_Out) * (1 - 0)) / (ub_Out - lb_Out)) + 0
testset = copy.deepcopy(testset_Raw)
testset[:,3] = (((testset_Raw[:,3] - lb_Out) * (1 - 0)) / (ub_Out - lb_Out)) + 0

# %% Creating Validation Data
train_perc = 0.9    ## Percentage used for training
val_perc = 1-train_perc  ## Percentage used for validation
idx_val = int(train_perc * trainset.shape[0])
trainset_val = trainset[0:idx_val,:]
validationset_val = trainset[idx_val:,:]
# if 0.1*
batch_size_validation = 1

# %% PyTorch Variables
x_test = torch.Tensor(testset[:,:c.D_in])
y_test = torch.Tensor(testset[:,c.D_in:])
x_val = torch.Tensor(validationset_val[:,:c.D_in])
y_val = torch.Tensor(validationset_val[:,c.D_in:])
x_train = torch.Tensor(trainset_val[:,:c.D_in])
y_train = torch.Tensor(trainset_val[:,c.D_in:])

x_train = torch.Tensor(x_train).to(c.device)
y_train = torch.Tensor(y_train).to(c.device)
x_val= torch.Tensor(x_val).to(c.device)
y_val = torch.Tensor(y_val).to(c.device)
x_test= torch.Tensor(x_test).to(c.device)
y_test = torch.Tensor(y_test).to(c.device)
# test_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(x_test, y_test),
#     batch_size=c.batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_val, y_val),
    batch_size=batch_size_validation, shuffle=True, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, drop_last=True)

cd = {
    'network_size' : c.Num_layers,
    'dropout': c.dropout,
    'hidden_layer_size':c.Hidden_layer_size
}


# %% Initalize model and optimizer
model = Fully_connected(c.D_in,c.D_out,cd)
print(c.D_in,c.D_out,c.Num_layers,c.Hidden_layer_size)
model.to(c.device)
optimizer = torch.optim.Adam(model.parameters(),c.lr,weight_decay=c.weight_decay)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1, patience=100, threshold=0.00001, threshold_mode='rel', cooldown=150, min_lr=1e-6, eps=1e-08, verbose=False)

# %% Actual Training.........
train_step = make_train_step(model,optimizer)
training_loss = []
test_loss =[]
ConvHistLoss_train=torch.rand((c.epochs,1))
ConvHistLoss_test=torch.rand((c.epochs,1))
for epoch in range(c.epochs):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        loss = train_step(x_batch, y_batch)
        batch_losses.append(loss)
    training_loss.append(np.mean(batch_losses))
    batch_losses =[]
    for x_batch,y_batch in val_loader:  #test_loader:
        loss = train_step(x_batch, y_batch,test=True)
        batch_losses.append(loss)
    test_loss.append(np.mean(batch_losses))
    ConvHistLoss_train[epoch,0]=training_loss[-1]
    ConvHistLoss_test[epoch,0]=test_loss[-1]
    print('epoch',epoch,'training loss',training_loss[-1], 'test loss',test_loss[-1])

#%% Saving data
U = model(x_test); #pos_vec1 = model_input;
Y_pred_norm = U.cpu()
Y_pred_norm = Y_pred_norm.detach().numpy()
Y_pred = (Y_pred_norm*(ub_Out -lb_Out)) + lb_Out 
with open('output/Testset_Pred', 'wb') as f:
    pickle.dump(Y_pred, f)
    
## RAE
rae=copy.deepcopy(Y_pred);re=copy.deepcopy(Y_pred)
rae[:,0] = np.absolute(Y_pred[:,0]-testset_Raw[:,3])/mean_Out *100
re[:,0] = (Y_pred[:,0]-testset_Raw[:,3])/mean_Out *100
mean_rae = np.mean(rae)

## RMSE
predictions = Y_pred_norm[:,0]
targets = testset[:,3]
mse = ((predictions - targets) ** 2).mean()
rmse = np.sqrt(mse)




lala = ConvHistLoss_train.cpu()
ConvHistTrain = lala.detach().numpy()
with open('output/ConvHistTrain', 'wb') as f:
    pickle.dump(ConvHistTrain, f)
    
lala = ConvHistLoss_test.cpu()
ConvHistTest = lala.detach().numpy()
with open('output/ConvHistTest', 'wb') as f:
    pickle.dump(ConvHistTest, f)

torch.save(model.state_dict(), 'output/trained_model.pt')

#%% Retieve Transfer model output
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

cd = {
    'network_size' : c.Num_layers,
    'dropout': c.dropout,
    'hidden_layer_size':c.Hidden_layer_size
}
#Initalize model and optimizer
model = Fully_connected(c.D_in,c.D_out,cd)
model.to(c.device)
model.load_state_dict(torch.load('output/trained_model.pt'))
model.eval()
# model = MyModel()
model.linear_out.register_forward_hook(get_activation('linear_out'))
#x = torch.randn(1, 25)
out = model(x_test)
x_act = activation['linear_out']

U_np=x_act.cpu()
U_np=U_np.detach().numpy()
Amp_U = {"U_np": U_np}
scipy.io.savemat("output/AmplitudeValues.mat", Amp_U)
#print(activation['linear_out'])

mdic = {"Y_pred": Y_pred, "re": re, "rae": rae, "mean_rae": mean_rae, "Normalized_RMSE": rmse, "testset_Raw": testset_Raw, "ConvHistTrain": ConvHistTrain, "ConvHistTest": ConvHistTest, "TransferModelActivation": Amp_U}
scipy.io.savemat("output/matlab_matrix.mat", mdic)

#%% Rechecking if activation is correct
p_uav1 = torch.zeros(x_act.shape[0],1,dtype=torch.cfloat).to(c.device)
for n in range (0,4):
    r = torch.sqrt( torch.pow( x_test[:,0] - c.mono_loc[0,n], 2) + torch.pow( x_test[:,1] - c.mono_loc[1,n], 2) + torch.pow( x_test[:,2] - c.mono_loc[2,n], 2))
    p_uav1[:,0] = p_uav1[:,0] + (x_act[:,n]* torch.exp(c.comp_1i*( - c.kappa[n]*r + c.phi[0,n])))/r

temp = c.T0*p_uav1/c.P_ref
re= torch.sqrt( torch.pow(temp.real,2) + torch.pow(temp.imag,2))
spl_mic_main1 = 20* torch.log10( torch.abs(re))
FinalOut = Normalize(spl_mic_main1)
FinalOutNP=FinalOut.cpu()
FinalOutNP=FinalOutNP.detach().numpy()

tempnp=out.cpu()
outputNP=tempnp.detach().numpy()

check = FinalOutNP - outputNP

# %% Writing to an excel sheet
import csv
from csv import writer  
List = [Train_type, test_size, mse, rmse, mean_rae, c.Num_layers, c.Hidden_layer_size, c.dropout, c.lr, c.batch_size, c.epochs, c.weight_decay, ConvHistTrain[-1], ConvHistTest[-1], "TRUE"]
  
with open('Output/OPTMA_NET_resultsVALIDATION.csv', 'a', newline='') as f_object:
    writer_object = writer(f_object)  
    writer_object.writerow(List)
    f_object.close()
    
# %% Contour Generation
######### X=-1 ###########
# resol = 300
# size_x = resol
# size_y =resol
# size_z = resol
# x = np.linspace(-1,-1,size_x)
# y = np.linspace(-2,2,size_y)
# z = np.linspace(-2,2,size_z)


# contour_out = torch.zeros(size_y,size_z)
# contour_out = contour_out.to(c.device)
# for i in range(y.shape[0]):
#     for j in range(z.shape[0]):
#         in_np = [[-1, y[i], z[j]]]
#         in_pt = torch.Tensor(in_np)
#         in_pt = torch.Tensor(in_pt).to(c.device)
#         contour_out[i][j] = model(in_pt)

# temp1 = contour_out.cpu()
# contour_vals_norm = temp1.detach().numpy()
# contour_vals = (contour_vals_norm*(ub_Out -lb_Out)) + lb_Out 

# mdic_contour = {"ContourVal": contour_vals}
# scipy.io.savemat("output/matlab_matrix_contour1.mat", mdic_contour)



# ######### Z=-1 ###########
# resol = 300
# size_x = resol
# size_y =resol
# size_z = resol

# x = np.linspace(-2,2,size_x)
# y = np.linspace(-2,2,size_y)
# z = np.linspace(-1,-1,size_z)

# contour_out_2 = torch.zeros(size_x,size_y)
# contour_out_2 = contour_out_2.to(c.device)
# for i in range(x.shape[0]):
#     for j in range(y.shape[0]):
#         in_np = [[x[i], y[j], -1]]
#         in_pt = torch.Tensor(in_np)
#         in_pt = torch.Tensor(in_pt).to(c.device)
#         contour_out_2[i][j] = model(in_pt)

# temp2 = contour_out_2.cpu()
# contour_vals_norm_2 = temp2.detach().numpy()
# contour_vals_2 = (contour_vals_norm_2*(ub_Out -lb_Out)) + lb_Out 

# mdic_contour_2 = {"ContourVal": contour_vals_2}
# scipy.io.savemat("output/matlab_matrix_contour2.mat", mdic_contour_2)


# x_train_full = torch.Tensor(trainset[:,:c.D_in])
# x_train_full = torch.Tensor(x_train_full).to(c.device)
# U = model(x_train_full); #pos_vec1 = model_input;
# Y_pred_norm_train = U.cpu()
# Y_pred_norm_train = Y_pred_norm_train.detach().numpy()
# Y_pred_train = (Y_pred_norm_train*(ub_Out -lb_Out)) + lb_Out 
# mdic_Ypred_traintest = {"Y_pred_train": Y_pred_train, "Y_pred": Y_pred}
# scipy.io.savemat("output/mdic_Ypred_traintest.mat", mdic_Ypred_traintest)