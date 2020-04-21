from torch_rbf import *
import torch.nn as nn
import torch.nn.functional as F

rbf_theta=0.05

class FancyKernel(nn.Module):

  def __init__(self,sigma_w,hidden_dim):
    self.K_theta=RBF(hidden_dim,1,gaussian(1/(2*(rbf_theta**2))))
    self.sigma_w=sigma_w
    self.M=nn.Linear(hidden_dim,2)
    self.optimizer= optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  def transform_inputs(x):
    ''''
    x.dim = (n,2)
    M.dim = (2,hidden_dim)
    ''''
    return nn.Linear(x).data.numpy
  def calc_likelihood(self,x,y):
    K_val=self.K_theta(x) + self.sigma_w * torch.eye(x.size()[0])
    part1= 0.5 * torch.matmul(torch.matmmul(y,torch.inverse(K_val),y) 
    part2= 0.5 * torch.log(torch.abs(K_val))
    return part1 + part2
  def forward(self,data):
    '''
    data is [x,y]
    '''
    x=transform_inputs(data[0]) 
    y=data[1]
    return self.calc_likelihood(x,y)

  def train(self,data):
    for i in range(10):
      loss=self.forward(data) #MAKE THIS LIST OF TENSORS
      print(loss.data)
      loss.backward()
      optimizer.step()
