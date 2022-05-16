
from torch.autograd import Variable
import random, pickle
import numpy as np, torch, os



def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)

