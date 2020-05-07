import torch
import torch.nn as nn

# RBF Layer

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self):
        super(RBF, self).__init__()
        #self.in_features = in_features
        #self.out_features = out_features
        #self.centres= nn.Parameter(torch.Tensor(out_features, in_features))
        #self.sigmas = nn.Parameter(torch.Tensor(out_features))
        

    def ignore_forward(self, input1,input2,gamma):
        print(input1.size(),input2.size())
        input1=nn.Parameter(input1)
        input2=nn.Parameter(input2.T)
        size1=input1.size(0)
        size2=input2.size(1)
        x = input1.repeat(size2,1)
        c = input2.repeat(1,size1)
        distances = (x - c).pow(2)*torch.tensor(gamma)
        return torch.exp(distances)
    def forward(self,input1,input2,gamma):
        size1=input1.size(0)
        size2=input2.size(0)
        array=torch.ones((size1,size2))
        
        for i in range(size1):
          for j in range(size2):
             #print(torch.exp(input1[i,:]-input2[j,:]).pow(2)*torch.tensor(gamma))
             array[i,j]=torch.exp(torch.sum(input1[i,:]-input2[j,:]).pow(2))*torch.tensor(gamma)
        #print(array.data.numpy())
        #x=input1.repeat(size2,1)
        #c=input2.repeat(1,size1)
        #distances=(x-c).pow(2)*torch.tensor(gamma)
        #print(x.size(),c.size(),"SIZES")
        return array


# RBFs

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
#kernel=RBF()
#print(kernel(torch.Tensor([[1.0,2.0],[3.0,4.0]]),torch.Tensor([[2.0,3.0],[4.0,5.0]]),0.5))
