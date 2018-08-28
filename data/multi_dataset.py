import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_joint_dataset
from PIL import Image

# Dataset for Multiple Observations
class MultiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print self.root
        # dataset structure
        # -- root
        #    + image_i folder
        #    + sharp img i
        #    + blurry_ij
        #    + psf_ij
        self.dir_img_folders = opt.dataroot # the root for all image folders

        self.sharp_paths, self.blurry_paths, self.psf_paths = 
        make_joint_dataset(self.dir_img_folders)

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        transform_list_ker = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)
        self.transform_ker = transforms.Compose(transform_list_ker)

    def __getitem__(self, index):
        # get the sharp image, and the set of blurry-kernel pairs

        sharp_path = self.sharp_paths[index]
        sharp = Image.open(sharp_path).convert('RGB')
        sharp = self.transform(sharp)        

        w = sharp.size(2)
        h = sharp.size(1)
        # perform cropping
        sharp = sharp[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]


        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        y_paths = self.blurry_paths[index]
        k_paths = self.psf_paths[index]
        
        blurry_set = []
        kernel_set = []
        for yp, kp in zip(y_paths, k_paths):
            y = Image.open(yp).convert('RGB')
            y = self.transform(y)
            y = y[:, h_offset:h_offset + self.opt.fineSize,
                  w_offset:w_offset + self.opt.fineSize]
            blurry_set.append(y)
            k = Image.open(kp).convert('RGB')
            k = self.transform_ker(k)
            K_arr = K.numpy()
            K_arr = K_arr / sum(K_arr.reshape(-1))
            #print("K_arr numpy %s " % K_arr)
            K = torch.Tensor(K_arr)
            kernel_set.append(k)

        return {'sharp': sharp,
                'blurry_set': blurry_set, 
                'kernel_set': kernel_set,
                }

    def __len__(self):
        return len(self.sharp_paths)

    def name(self):
        return 'MultiDataset'
