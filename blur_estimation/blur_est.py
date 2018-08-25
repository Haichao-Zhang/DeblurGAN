"""
This file is for estimating the blur kernel
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable
from qpth.qp import QPFunction


def shift2d(arr, di, dj, fill_value=0):
    # arr is Tensor
    if di == 0 and dj == 0:
        return arr

    result = torch.zeros_like(arr)
    if di == 0:
        if dj > 0:
            result[:, dj:] = arr[:, :-dj]
        elif dj <= 0:
            result[:, :dj] = arr[:, -dj:]
    elif dj == 0:
        if di > 0:
            result[di:, :] = arr[:-di, :]
        elif di < 0:
            result[:di, :] = arr[-di:, :]
    elif di > 0 and dj > 0:
        result[di:, dj:] = arr[:-di, :-dj]
    elif di > 0 and dj < 0:
        result[di:, :dj] = arr[:-di, -dj:]
    elif di < 0 and dj > 0:
        result[:di, dj:] = arr[-di:, :-dj]
    elif di < 0 and dj < 0:
        result[:di, :dj] = arr[-di:, -dj:]
    return result




def get_H(img, k_sz):
    # image is a tensor (e.g. predicted sharp image)
    assert k_sz % 2 == 1, "kernel size should be odd number"
    h_sz = (k_sz - 1) / 2
    img_sz = np.prod(img.shape)
    H = torch.zeros([img_sz, k_sz * k_sz])
    cnt = 0
    for i in range(-h_sz, h_sz + 1):
        for j in range(-h_sz, h_sz + 1): # should + 1
            tmp = shift2d(img, i, j)
            H[:, cnt] = tmp.view(-1)
            cnt += 1
    return H


def get_K(img, b_img, k_sz):
    # img: single channel image
    # image Tensor
    img = img.data.cpu().numpy()
    b_img = b_img.data.cpu().numpy()


    #img = img.numpy()
    #b_img = img.numpy()
    img = img.astype(float)
    b_img = b_img.astype(float)

    img = torch.Tensor(img)
    H = get_H(img, k_sz)
    img = img.numpy()
    A = np.matmul(H.transpose(1, 0), H)
    b = np.matmul(H.transpose(1, 0), b_img.reshape(-1, 1))
    
    reg = np.eye(A.shape[0])
    # A = A + 0.00001 * reg
    b = -b

    A = torch.Tensor(A)
    b = torch.Tensor(b)
    A = A.unsqueeze(0)
    b = b.unsqueeze(0).squeeze(-1)

    G = - torch.eye(A.size(1))
    h = torch.zeros(A.size(1))

    J = torch.ones((1, A.size(1)))
    o = torch.ones(1)
    
    e = Variable(torch.Tensor())
    k = QPFunction(verbose=False)(A, b, G, h, J, o)
    k = k.numpy()
    k = k.reshape(k_sz, k_sz)
    k = k.astype(float)
    k[k < 0] = 0
    # normalize
    k = k / k.sum()
    return torch.Tensor(k)


def test_get_K():
    folder = '/media/DATA/data/blurred_sharp_org/blurred_sharp'
    img = cv2.imread(folder + '/mine/sharp/1.png')
    img = img[:,:,0].astype(float)
    # print img

    b_img = cv2.imread(folder + '/mine/blurry/1.png')
    b_img = b_img[:,:,0].astype(float)
    k_sz = 21
    img = torch.Tensor(img)
    H = get_H(img, k_sz)
    img = img.numpy()
    A = np.matmul(H.transpose(1, 0), H)
    b = np.matmul(H.transpose(1, 0), b_img.reshape(-1, 1))
    
    reg = np.eye(A.shape[0])
    # A = A + 0.00001 * reg
    b = -b

    A = torch.Tensor(A)
    b = torch.Tensor(b)
    A = A.unsqueeze(0)
    b = b.unsqueeze(0).squeeze(-1)

    G = - torch.eye(A.size(1))
    h = torch.zeros(A.size(1))

    J = torch.ones((1, A.size(1)))
    o = torch.ones(1)

    print A.shape
    print b.shape
    
    #k = torch.gels(b, A)
    e = Variable(torch.Tensor())
    k = QPFunction(verbose=False)(A, b, G, h, J, o)
    print(k.view(-1))
    k = k.numpy()
    
    #k = np.linalg.solve(A, b)
    # k, _ = np.linalg.lstsq(A, b)

    print(k)
    k = k.reshape(21, 21)
    # k = -k
    k = k.astype(float)
    k[k < 0] = 0
    k = k / max(k.reshape(-1)) * 255 
    print(k)
    cv2.imwrite(folder + '/ker.png', k)


def test():
    k = torch.zeros(5, 5)
    k[0, 2] = 1
    k[3, 4] = 2
    k[2, 1] = 4
    print(k)

    ks = shift2d(k, 1, 1)
    print(ks)
    #ks = shift2d(k, 1, -1)
    #print(ks)
    #ks = shift2d(k, 1, 1)
    #print(ks)
    #ks = shift2d(k, -1, 1)
    #print(ks)


if __name__ == "__main__":
    test_get_K()
