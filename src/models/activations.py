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
        return x * (F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), F.softplus(self.p)) + self.k)
    
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
        return x * (F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), F.softplus(self.p)) + self.k)
    
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
        return x * (F.softplus(self.alpha) * torch.pow(F.relu(x - self.a), 1.0 + F.softplus(self.p)) + self.k)
    
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
    
class mExp(nn.Module):
    '''
    (a * exp(x - s)^p + b)
    a is learnable
    p is learnable
    s is learnable
    b is learnable
    a has to be positive
    p has to be positive
    '''
    def __init__(self, a=1.0, p=1.0, s=0.0, b=-1.0):
        super(mExp, self).__init__()
        inverse_softplus_a = math.log(math.exp(a) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.a = nn.Parameter(torch.tensor(inverse_softplus_a, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
    def forward(self, x):
        return (F.softplus(self.a) * torch.pow(torch.exp(x - self.s), F.softplus(self.p)) + self.b)
    
class mSiLU(nn.Module):
    '''
    SiLU(x) = x * sigmoid(x)
    mSiLU(x) = (a * abs(SiLU(x - s))^p * sign(SiLU(x - s)) + b)
    a is learnable
    p is learnable
    s is learnable
    b is learnable
    a has to be positive
    p has to be 1 or greater
    '''
    def __init__(self, a=1.0, p=1e-7, s=0.0, b=0.0):
        super(mSiLU, self).__init__()
        inverse_softplus_a = math.log(math.exp(a) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.a = nn.Parameter(torch.tensor(inverse_softplus_a, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.activation = nn.SiLU()
    def forward(self, x):
        silu = self.activation(x - self.s)
        return (F.softplus(self.a) * torch.pow(torch.abs(silu), 1.0 + F.softplus(self.p)) * torch.sign(silu) + self.b)
    
class mGELU(nn.Module):
    '''
    GELU(x) = 0.5 * (1 + erf(x / sqrt(2)))
    mGELU(x) = (a * abs(GELU(x - s))^p * sign(GELU(x - s)) + b)
    a is learnable
    p is learnable
    s is learnable
    b is learnable
    a has to be positive
    p has to be 1 or greater
    '''
    def __init__(self, a=1.0, p=1e-7, s=0.0, b=0.0):
        super(mGELU, self).__init__()
        inverse_softplus_a = math.log(math.exp(a) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.a = nn.Parameter(torch.tensor(inverse_softplus_a, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.activation = nn.GELU()
    def forward(self, x):
        gelu = self.activation(x - self.s)
        return (F.softplus(self.a) * torch.pow(torch.abs(gelu), 1.0 + F.softplus(self.p)) * torch.sign(gelu) + self.b)
    
class mELU(nn.Module):
    '''
    ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
    mELU(x) = (a * abs(ELU(x - s))^p * sign(ELU(x - s)) + b)
    a is learnable
    p is learnable
    s is learnable
    b is learnable
    a has to be positive
    p has to be 1 or greater
    '''
    def __init__(self, a=1.0, p=1e-7, s=0.0, b=0.0):
        super(mELU, self).__init__()
        inverse_softplus_a = math.log(math.exp(a) - 1.0)
        inverse_softplus_p = math.log(math.exp(p) - 1.0)
        self.a = nn.Parameter(torch.tensor(inverse_softplus_a, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(inverse_softplus_p, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.activation = nn.ELU(alpha=1.0)
    def forward(self, x):
        elu = self.activation(x - self.s)
        return (F.softplus(self.a) * torch.pow(torch.abs(elu), 1.0 + F.softplus(self.p)) * torch.sign(elu) + self.b)

class m2ReLU(nn.Module):
    '''
    m2ReLU(x):
        x<=s: a_n * ReLU(-(x-s)) + b
        x>s: a_p * ReLU(x-s)^p_p + b
    a_n is learnable
    a_p is learnable
    p_p is learnable
    s is learnable
    b is learnable
    a_p has to be positive
    p_p has to be 1 or greater
    '''
    def __init__(self, a_n=-0.1, a_p=1.0, p_p=1.0, s=0.0, b=0.0):
        super(m2ReLU, self).__init__()
        self.a_n = nn.Parameter(torch.tensor(a_n, dtype=torch.float32))
        self.a_p = nn.Parameter(torch.tensor(math.log(math.exp(a_p) - 1.0), dtype=torch.float32))
        self.p_p = nn.Parameter(torch.tensor(math.log(math.exp(p_p) - 1.0), dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
    def forward(self, x):
        return self.a_n * F.relu(-(x-self.s)) + F.softplus(self.a_p) * torch.pow(F.relu(x-self.s), 1.0 + F.softplus(self.p_p)) + self.b

class m3ReLU(nn.Module):
    '''
    m3ReLU(x):
        x<=s: a_n * ReLU(-(x-s))^p_n + b
        x>s: a_p * ReLU(x-s)^p_p + b
    a_n is learnable
    a_p is learnable
    p_n is learnable
    p_p is learnable
    s is learnable
    b is learnable
    a_n has to be positive
    a_p has to be positive
    p_n has to be 1 or greater
    p_p has to be 1 or greater
    '''
    def __init__(self, a_n=1e-7, a_p=1.0, p_n=1e-7, p_p=1.0, s=0.0, b=0.0):
        super(m3ReLU, self).__init__()
        self.a_n = nn.Parameter(torch.tensor(math.log(math.exp(a_n) - 1.0), dtype=torch.float32))
        self.a_p = nn.Parameter(torch.tensor(math.log(math.exp(a_p) - 1.0), dtype=torch.float32))
        self.p_n = nn.Parameter(torch.tensor(math.log(math.exp(p_n) - 1.0), dtype=torch.float32))
        self.p_p = nn.Parameter(torch.tensor(math.log(math.exp(p_p) - 1.0), dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
    def forward(self, x):
        return self.a_n * torch.pow(F.relu(-(x-self.s)), 1.0 + F.softplus(self.p_n)) + F.softplus(self.a_p) * torch.pow(F.relu(x-self.s), 1.0 + F.softplus(self.p_p)) + self.b

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
    'mexp': mExp,
    ## NEW
    'silu': nn.SiLU,
    'msilu': mSiLU,
    'gelu': nn.GELU,
    'mgelu': mGELU,
    'elu': nn.ELU,
    'melu': mELU,
    ## NEW
    'm2relu': m2ReLU,
    'm3relu': m3ReLU,
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
    elif activation_name == 'mexp':
        return activation_class(a=config.activation_a, p=config.activation_p, s=config.activation_s, b=config.activation_b)
    elif activation_name == 'msilu':
        return activation_class(a=config.activation_a, p=config.activation_p, s=config.activation_s, b=config.activation_b)
    elif activation_name == 'mgelu':
        return activation_class(a=config.activation_a, p=config.activation_p, s=config.activation_s, b=config.activation_b)
    elif activation_name == 'melu':
        return activation_class(a=config.activation_a, p=config.activation_p, s=config.activation_s, b=config.activation_b)
    elif activation_name == 'm2relu':
        return activation_class(a_n=config.activation_a_n, a_p=config.activation_a_p, p_p=config.activation_p_p, s=config.activation_s, b=config.activation_b)
    elif activation_name == 'm3relu':
        return activation_class(a_n=config.activation_a_n, a_p=config.activation_a_p, p_n=config.activation_p_n, p_p=config.activation_p_p, s=config.activation_s, b=config.activation_b)
    return activation_class() 