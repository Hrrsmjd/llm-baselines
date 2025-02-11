import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReluSquared(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 2)

class ReluCubed(nn.Module):
    def forward(self, x):
        return torch.pow(F.relu(x), 3)
    
class MRePU(nn.Module):
    def __init__(self, c=-1, p=2):
        super(MRePU, self).__init__()
        self.c = -c
        self.p = p
    def forward(self, x):
        return x * torch.pow(F.relu(x + self.c), self.p)
    
class MRePU_learnable(nn.Module):
    ## MRePU with learnable p
    def __init__(self, c=-1,p=2):
        super(MRePU_learnable, self).__init__()
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.c = -c
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
    def forward(self, x):
        return x * torch.pow(F.relu(x + self.c), F.softplus(self.p))
    
class MRePU_learnable2(nn.Module):
    ## MRePU with learnable p
    def __init__(self, c=-1, p=2):
        super(MRePU_learnable2, self).__init__()
        inverse_softplus_c = math.log(math.exp(-c) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.c = nn.Parameter(torch.tensor(inverse_softplus_c, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
    def forward(self, x):
        return x * torch.pow(F.relu(x + F.softplus(self.c)), F.softplus(self.p))

class xReLU(nn.Module):
    '''
    Idea2: 
    x * (alpha * ReLU(x)^p)
    alpha is learnable
    p is learnable
    alpha has to be positive
    p has to be positive
    '''
    def __init__(self, alpha=1.0, p=1, a=0.0, k=0.0):
        super(xReLU, self).__init__()
        inverse_softplus_alpha = math.log(math.exp(alpha) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.alpha = nn.Parameter(torch.tensor(inverse_softplus_alpha, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.a = a
        self.k = k
    def forward(self, x):
        return x * F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), F.softplus(self.p)) + self.k
    
class xReLU2(nn.Module):
    '''
    Idea2: 
    x * (alpha * ReLU(x - a)^p + k)
    alpha is learnable
    p is learnable
    a is learnable
    k is learnable
    alpha has to be positive
    p has to be positive
    '''
    def __init__(self, alpha=1.0, p=1, a=0.0, k=0.0):
        super(xReLU2, self).__init__()
        inverse_softplus_alpha = math.log(math.exp(alpha) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.alpha = nn.Parameter(torch.tensor(inverse_softplus_alpha, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
    def forward(self, x):
        return x * F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), F.softplus(self.p)) + self.k
    
class xReLU3(nn.Module):
    '''
    Idea2: 
    x * (alpha * ReLU(x - a)^p + k)
    alpha is learnable
    p is learnable
    a is learnable
    k is learnable
    alpha has to be positive
    p has to be 1 or greater
    '''
    def __init__(self, alpha=1.0, p=1e-7, a=0.0, k=0.0):
        super(xReLU3, self).__init__()
        inverse_softplus_alpha = math.log(math.exp(alpha) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.alpha = nn.Parameter(torch.tensor(inverse_softplus_alpha, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
    def forward(self, x):
        # Add 1 to F.softplus(self.p) to ensure p >= 1
        return x * F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), 1.0 + F.softplus(self.p)) + self.k
    
class mReLU(nn.Module):
    '''
    Idea2: 
    (alpha * ReLU(x - a)^p + k)
    alpha is learnable
    p is learnable
    a is learnable
    k is learnable
    alpha has to be positive
    p has to be 1 or greater
    '''
    def __init__(self, alpha=1.0, p=1.0, a=0.0, k=0.0):
        super(mReLU, self).__init__()
        inverse_softplus_alpha = math.log(math.exp(alpha) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.alpha = nn.Parameter(torch.tensor(inverse_softplus_alpha, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
    def forward(self, x):
        # Add 1 to F.softplus(self.p) to ensure p >= 1
        return F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), 1.0 + F.softplus(self.p)) + self.k

# Dictionary mapping activation names to their classes
ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'relu2': ReluSquared,
    'relu3': ReluCubed,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'mrepu': MRePU,
    'mrepu_learnable': MRePU_learnable,
    'mrepu_learnable2': MRePU_learnable2,
    'xrelu': xReLU,
    'xrelu2': xReLU2,
    'xrelu3': xReLU3,
    'mrelu': mReLU,
}

def get_activation(activation_name, config=None):
    """
    Returns an instance of the activation function
    
    Args:
        activation_name (str): Name of the activation function
        config: Configuration object containing activation parameters
        
    Returns:
        nn.Module: Instantiated activation function
    """
    if activation_name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Activation {activation_name} not found. Available activations: {list(ACTIVATION_REGISTRY.keys())}")
    
    activation_class = ACTIVATION_REGISTRY[activation_name]
    
    # Handle parameterized activations
    if activation_name == 'mrepu':
        return activation_class(c=config.activation_c, p=config.activation_p)
    elif activation_name == 'mrepu_learnable':
        return activation_class(c=config.activation_c, p=config.activation_p)
    elif activation_name == 'mrepu_learnable2':
        return activation_class(c=config.activation_c, p=config.activation_p)
    elif activation_name == 'xrelu':
        return activation_class(alpha=config.activation_alpha, p=config.activation_p, a=config.activation_a, k=config.activation_k)
    elif activation_name == 'xrelu2':
        return activation_class(alpha=config.activation_alpha, p=config.activation_p, a=config.activation_a, k=config.activation_k)
    elif activation_name == 'xrelu3':
        return activation_class(alpha=config.activation_alpha, p=config.activation_p, a=config.activation_a, k=config.activation_k)
    elif activation_name == 'mrelu':
        return activation_class(alpha=config.activation_alpha, p=config.activation_p, a=config.activation_a, k=config.activation_k)
    return activation_class() 