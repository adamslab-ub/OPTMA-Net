import torch
import torch.nn
import config1 as c
import numpy as np
#from Call_Partial import Custom_Loss
#from Call_Partial_Numeric import Partial_Phys1
from CustomLayers import *




# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.sparse_(m.weight,sparsity=0.9)
        m.bias.data.fill_(1)
class Fully_connected(torch.nn.Module):
    def __init__(self, D_in, D_out,config):
        super(Fully_connected, self).__init__()
        self.layers = torch.nn.ModuleList()
        H = config['hidden_layer_size']
        #self.drop = torch.nn.ModuleList()
        self.norm = torch.nn.BatchNorm1d(D_in)
        self.linear_in = torch.nn.Linear(D_in, H)
        self.dropoutp = config['dropout']

        for i in range(c.Num_layers):
            self.layers.append(torch.nn.Linear(H,H))
        self.drop = torch.nn.Dropout(p=self.dropoutp)
        self.linear_out = torch.nn.Linear(H, D_out)
        #self.nl1 = torch.nn.ReLU()
        #self.nl1 = torch.nn.LeakyReLU(negative_slope=1)
        self.nl1 = torch.nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        out = self.linear_in(self.norm(x))
        for i in range(len(self.layers)):
            net = self.layers[i]
            out = self.nl1(self.drop(net(out)))
        out = self.linear_out(out)
        #out = torch.nn.ReLU(out)# + 0.5

        p_uav1 = torch.zeros(out.shape[0],1,dtype=torch.cfloat).to(c.device)
        for n in range (0,4):
            r = torch.sqrt( torch.pow( x[:,0] - c.mono_loc[0,n], 2) + torch.pow( x[:,1] - c.mono_loc[1,n], 2) + torch.pow( x[:,2] - c.mono_loc[2,n], 2))
            p_uav1[:,0] = p_uav1[:,0] + (out[:,n]* torch.exp(c.comp_1i*( - c.kappa[n]*r + c.phi[0,n])))/r
    
        temp = c.T0*p_uav1/c.P_ref
        re= torch.sqrt( torch.pow(temp.real,2) + torch.pow(temp.imag,2))
        spl_mic_main1 = 20* torch.log10( torch.abs(re))
        
        spl_mic_main1 = (((spl_mic_main1 - c.lb_norm) * (1 - 0)) / (c.ub_norm - c.lb_norm)) + 0
        return (spl_mic_main1)

def l2_loss(input, target):    
     loss = torch.nn.MSELoss()
     return loss(input,target)
def make_train_step(model,optimizer,scheduler=None):
    # Builds function that performs a step in the train loop
    def train_step(x, y,test=False):
        a=model
        if not test:
            yhat = a(x)
            #print(yhat.requires_grad)
            #loss = l2_loss(yhat, y, x)
            loss = l2_loss(yhat, y)
            #print(yhat)
            optimizer.zero_grad()
            loss.backward()
            #print(model.layers[0].weight.grad[0,:])
            # torch.autograd.gradcheck()
            optimizer.step()
        else:
            a = model.eval()
            with torch.no_grad():
                yhat = a(x)
                #loss = l2_loss(yhat, y, x)
                loss = l2_loss(yhat, y)
            if scheduler:
                scheduler.step(loss)
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step
