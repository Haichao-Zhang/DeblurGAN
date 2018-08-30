import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_joint_dataset
from PIL import Image

# Dataset for Multiple Observations
class MultiDatasetTest(BaseDataset):
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

        # if there is no ground-truth, the sharp_paths and/or psf_paths
        # would be empty
        self.sharp_paths, self.blurry_paths, self.psf_paths = \
        make_joint_dataset(self.dir_img_folders)
                
        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))
                         ]

        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
        # get the sharp image, and the set of blurry-kernel pairs
        
        sharp = None
        if len(self.sharp_paths) != 0:
            sharp_path = self.sharp_paths[index]
            sharp = Image.open(sharp_path).convert('RGB')
            sharp = self.transform(sharp)


        y_paths = self.blurry_paths[index]
        k_paths = self.psf_paths[index]
        
        blurry_set = []
        kernel_set = []
        w_offset = -1
        h_offset = -1
        for yp in y_paths:
            y = Image.open(yp).convert('RGB')
            y = self.transform(y)
            w = sharp.size(2)
            h = sharp.size(1)
            # perform cropping
            if w_offset < 0:
                w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            if h_offset < 0:
                h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            

            y = y[:, h_offset:h_offset + self.opt.fineSize,
                  w_offset:w_offset + self.opt.fineSize]
            
            blurry_set.append(y)

        print(blurry_set)
        return {'sharp': sharp,
                'blurry_set': blurry_set, 
                'kernel_set': kernel_set,
                }

    def __len__(self):
        return len(self.sharp_paths)

    def name(self):
        return 'MultiDatasetTest'
