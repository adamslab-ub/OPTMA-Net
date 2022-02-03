import torch
device = torch.device('cuda:0')
D_in = 3
D_invin = 80
D_invout = 30
D_out = 4
lr = 1e-4
dropout = 0.1
batch_size = 25
epochs = 100
Hidden_layer_size = 50
Num_layers = 3
weight_decay = 0#1e-5


# mono_loc=torch.cuda.FloatTensor([[0.642718009121920,	-0.769149253898193,	-1.99925545455873,	1.99747800231622], 
#                             [-0.632032920724389,	-1.98522154843943,	-1.99955394174514,	-1.99525104388291],
#                             [-0.531431436623549,	-1.99485413030846,	-1.99924280318578,	1.99496038586142]])
mono_loc=torch.cuda.FloatTensor([[0.176,  -0.176,  -0.176,  0.176], 
                            [0.176,  0.176,  -0.176,  -0.176],
                            [0,  0,  0,  0]])

comp_1i = torch.tensor([[0.0 + 1j]]).to(device)
# phi = torch.cuda.FloatTensor([[45.0016347524273,	0.997894925987488,	45.3771593274238,	0.000439932865296745]])
phi = torch.cuda.FloatTensor([[45,  45,  45,  45]])
T0=torch.cuda.FloatTensor([[1]])
P_ref = torch.cuda.FloatTensor([[20e-6]])


#freq=torch.cuda.FloatTensor([[94.0512457240110,	0.000776299875521865,	134.229782863088,	0.000888841271693431]])
freq=torch.cuda.FloatTensor([[175,	175,	175,	175]])
pi = torch.acos(torch.zeros(1)).item() * 2
ang_freq = 2*pi*freq[0,:]
kappa = ang_freq/343

lb_norm= torch.cuda.FloatTensor([[75.75212099867892]])
ub_norm= torch.cuda.FloatTensor([[83.27268669594696]])
