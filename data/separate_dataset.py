import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class SeparateDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print self.root
        # dataset structure
        # -- root
        #    + sharp
        #    + blurry
        #    + psf
        self.dir_sharp = os.path.join(opt.dataroot, 'sharp')
        self.dir_blurry = os.path.join(opt.dataroot, 'blurry')
        self.dir_psf = os.path.join(opt.dataroot, 'psf')
        self.blurry_paths = sorted(make_dataset(self.dir_blurry))
        self.sharp_paths = sorted(make_dataset(self.dir_sharp))
        self.psf_paths = sorted(make_dataset(self.dir_psf))

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        transform_list_ker = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)
        self.transform_ker = transforms.Compose(transform_list_ker)

    def __getitem__(self, index):
        A_path = self.blurry_paths[index]
        A = Image.open(A_path).convert('RGB')
        A = self.transform(A)
        B_path = self.sharp_paths[index]
        B = Image.open(B_path).convert('RGB')
        B = self.transform(B)

        #A = A[0, :, :].unsqueeze(0)
        #B = B[0, :, :].unsqueeze(0)

        w = A.size(2)
        h = A.size(1)
        
        # perform cropping
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        K_path = self.psf_paths[index]
        K = Image.open(K_path)
        K = self.transform_ker(K)
        K_arr = K.numpy()

        K_arr = K_arr / sum(K_arr.reshape(-1))

        K = torch.Tensor(K_arr)
        # avoid resizing image only
        # AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        # A = self.transform(A)


        return {'A': A, 'B': B, 'K': K,
                'A_paths': A_path, 'B_paths': B_path, 'K_path': K_path}

    def __len__(self):
        return len(self.blurry_paths)

    def name(self):
        return 'AlignedDataset'
