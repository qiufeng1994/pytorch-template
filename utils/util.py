import os
import torch
from torch.autograd.variable import Variable


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.zeros(len(labels), 2)
    one_hot[torch.arange(len(labels)), labels] =1
    target = Variable(one_hot)
    
    return target