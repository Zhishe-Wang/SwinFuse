import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda"if torch.cuda.is_available()else"cpu")

EPSILON = 1e-5


def attention_fusion_weight(tensor1, tensor2, p_type):

    f_row_vector = row_vector_fusion(tensor1, tensor2, p_type)
    f_column_vector = column_vector_fusion(tensor1, tensor2, p_type)

    tensor_f = (f_row_vector + f_column_vector)

    return tensor_f


def row_vector_fusion(tensor1, tensor2, p_type):
    shape = tensor1.size()
    # calculate row vector attention
    row_vector_p1 = row_vector_attention(tensor1, p_type)
    row_vector_p2 = row_vector_attention(tensor2, p_type)

    # get weight map
    row_vector_p_w1 = torch.exp(row_vector_p1) / (torch.exp(row_vector_p1) + torch.exp(row_vector_p2) + EPSILON)
    row_vector_p_w2 = torch.exp(row_vector_p2) / (torch.exp(row_vector_p1) + torch.exp(row_vector_p2) + EPSILON)

    row_vector_p_w1 = row_vector_p_w1.repeat(1, 1, shape[2], shape[3])
    row_vector_p_w1 = row_vector_p_w1.to(device)
    row_vector_p_w2 = row_vector_p_w2.repeat(1, 1, shape[2], shape[3])
    row_vector_p_w2 = row_vector_p_w2.to(device)

    tensor_f = row_vector_p_w1 * tensor1 + row_vector_p_w2 * tensor2

    return tensor_f


def column_vector_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate column vector attention
    column_vector_1 = column_vector_attention(tensor1, spatial_type)
    column_vector_2 = column_vector_attention(tensor2, spatial_type)

    column_vector_w1 = torch.exp(column_vector_1) / (torch.exp(column_vector_1) + torch.exp(column_vector_2) + EPSILON)
    column_vector_w2 = torch.exp(column_vector_2) / (torch.exp(column_vector_1) + torch.exp(column_vector_2) + EPSILON)

    column_vector_w1 = column_vector_w1.repeat(1, shape[1], 1, 1)
    column_vector_w1 = column_vector_w1.to(device)
    column_vector_w2 = column_vector_w2.repeat(1, shape[1], 1, 1)
    column_vector_w2 = column_vector_w2.to(device)

    tensor_f = column_vector_w1 * tensor1 + column_vector_w2 * tensor2

    return tensor_f


# row vector_attention
def row_vector_attention(tensor, type="l1_mean"):
    shape = tensor.size()

    c = shape[1]
    h = shape[2]
    w = shape[3]
    row_vector = torch.zeros(1, c, 1, 1)
    if type is"l1_mean":
        row_vector = torch.norm(tensor, p=1, dim=[2, 3], keepdim=True) / (h * w)
    elif type is"l2_mean":
        row_vector = torch.norm(tensor, p=2, dim=[2, 3], keepdim=True) / (h * w)
    elif type is "linf":
            for i in range(c):
                tensor_1 = tensor[0,i,:,:]
                row_vector[0,i,0,0] = torch.max(tensor_1)
            ndarray = tensor.cpu().numpy()
            max = np.amax(ndarray,axis=(2,3))
            tensor = torch.from_numpy(max)
            row_vector = tensor.reshape(1,c,1,1)
            row_vector = row_vector.to(device)
    return row_vector


# # column vector attention
def column_vector_attention(tensor, type='l1_mean'):

    shape = tensor.size()
    c = shape[1]
    h = shape[2]
    w = shape[3]
    column_vector = torch.zeros(1, 1, 1, 1)
    if type is 'l1_mean':
        column_vector = torch.norm(tensor, p=1, dim=[1], keepdim=True) / c
    elif type is"l2_mean":
        column_vector = torch.norm(tensor, p=2, dim=[1], keepdim=True) / c
    elif type is "linf":
        column_vector, indices = tensor.max(dim=1, keepdim=True)
        column_vector = column_vector / c
        column_vector = column_vector.to(device)
    return column_vector

