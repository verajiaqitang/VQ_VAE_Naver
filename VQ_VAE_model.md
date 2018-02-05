

```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data

# set up the VQ class 
class VQ(torch.autograd.Function):
    
    def forward(ctx, x, emb):
    '''
    x: batch size, dimension
    emb: embedding number, dimemsion
    '''
        dist = E_distance(x, emb)
        indices = torch.min(dist, -1)[1]
        ctx.indices = indices
        ctx.emb_num = emb.size(0)
        ctx.b_size = x.size(0)
        return torch.index_select(emb, 0, indices)

   
    def backward(ctx, grad_output):
        indices = ctx.indices.view(-1,1)
        b_size = ctx.b_size
        emb_num = ctx.emb_num

        # get a one hot index
        one_hot_ind = torch.zeros(b_size, emb_num)
        one_hot_ind.scatter_(1, indices, 1)
        one_hot_ind = Variable(one_hot_ind, requires_grad=False)
        grad_emb = torch.mm(one_hot_ind.t(), grad_output)
        return grad_output, grad_emb


class Embediing_layer(nn.Module):
    def __init__(self, D, K):
        super(Embediing_layer, self).__init__()
        self.emb = nn.Embedding(K, D)
        self.K = K
        self.D = D

    def forward(self, x):
     
        return VQ.apply(x, self.emb.weight)
def E_distance(x1, x2):
  
    a = x1.size(0)
    b = x2.size(0)
    d1 = torch.stack([x1]*b).transpose(0,1)
    d2 = torch.stack([x2]*a)
    distance=torch.sum((d1-d2)**2, 2).squeeze()
    return distance

class VQVAE(nn.Module):
    def __init__(self, emb_dim, emb_num):
        super(VQVAE, self).__init__()
       
        self.layer = Embediing_layer(D=emb_dim,K=emb_num)

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, emb_dim)
        self.fc3 = nn.Linear(emb_dim, 400)
        self.fc4 = nn.Linear(400, 784)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def encoder(self, x):
        h1 =   self.relu(self.fc1(x))
        h2=self.fc2(h1)
        return h2
    def decoder(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    def forward(self, x):
        self.z_e = self.encoder(x)
        self.z_q = self.layer(self.z_e)
        self.x_reconst = self.decoder(self.z_q)
        return self.x_reconst
 
    
    def sample_from_modes(self):
       
        zq = self.layer.emb.weight
        samples = self.decoder(zq)
        return samples
    
    


```
