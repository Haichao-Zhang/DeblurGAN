###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

## make a joint dataset based on a specific folder structure
def make_joint_dataset(dir_img_folders):
    x_paths = []
    y_paths = []
    k_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for folder_name in sorted(dir):
        # there should be only one sharp image
        for fname in glob.glob(folder_name + "*sharp*"):
            if is_image_file(fname):
                x_paths.append(fname)

        yp = []
        for fname in sorted(glob.glob(folder_name + "*blurry*")):
            if is_image_file(fname):
                yp.append(fname)
        y_paths.append(yp)

        kp = []
        for fname in sorted(glob.glob(folder_name + "*ker*")):
            if is_image_file(fname):
                kp.append(fname)
        k_paths.append(kp)

        assert len(y_paths) == len(k_paths)
        assert len(y_paths) == len(x_paths)
        

    return x_paths, y_paths, k_paths


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
