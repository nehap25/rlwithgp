from torch_rbf import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
rbf_theta=0.05

class FancyKernel(nn.Module):

  def __init__(self,sigma_w,hidden_dim):
    super(FancyKernel, self).__init__()
    self.K_theta=RBF(hidden_dim,1,gaussian)
    self.sigma_w=sigma_w
    self.M=nn.Linear(2,hidden_dim)
  def transform_inputs(self,x):
    " x.dim = (n,2) M.dim = (2,hidden_dim)"
    return self.M(x).float()
  def calc_likelihood(self,x,y):
    K_val=self.K_theta(x) + self.sigma_w * torch.eye(x.size()[0])
    
    part1= 0.5 * torch.matmul(torch.matmul(y.T,torch.inverse(K_val)),y)
    part2= 0.5 * torch.log(torch.abs(K_val))
    return part1 + part2
  def forward(self,data):
    '''
    data is [x,y]
    '''
    x=self.transform_inputs(data[0]) 
    y=data[1]
    return self.calc_likelihood(x,y)

def train(model,optimizer,data):
  for i in range(10):
    loss=model.forward(data) #MAKE THIS LIST OF TENSORS
    print(loss)
    loss.backward()
    optimizer.step()
    print(loss.item())
X_data= torch.from_numpy(np.array([[1.1,2.2],[2.2,3.3],[3.3,4.4],[4.4,5.5],[5.5,6.6]]))
Y_data=torch.from_numpy(np.array([[1.1],[2.2],[3.3],[4.4],[5.5]]))

data=[X_data,Y_data]
model=FancyKernel(0.1,10).double()
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train(model,optimizer,data)
