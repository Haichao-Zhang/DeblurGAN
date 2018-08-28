import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory


class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None,
                 path_to_save=None):
        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path_to_save_img: folder to save blurry image results.
        :param path_to_save_psf: folder to save psfs
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = misc.imread(self.image_path)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                raise Exception('We support only RGB images yet.')
            elif self.shape[0] != self.shape[1]:
                raise Exception('We support only square images yet.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path_to_save

        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def blur_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        # self.original in [0, 255]
        self.original = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.original = cv2.cvtColor(self.original, cv2.COLOR_RGB2BGR)
        if len(psf) > 1:
            for p in psf:
                tmp = np.pad(p, delta // 2, 'constant')
                tmp = tmp.astype(float)
                tmp = tmp / tmp.sum()
                blurred = np.zeros_like(self.original)
                blurred[:, :, 0] = np.array(signal.fftconvolve(self.original[:, :, 0], tmp, 'same'))
                blurred[:, :, 1] = np.array(signal.fftconvolve(self.original[:, :, 1], tmp, 'same'))
                blurred[:, :, 2] = np.array(signal.fftconvolve(self.original[:, :, 2], tmp, 'same'))
                #result.append(np.abs(blurred))
                result.append(blurred.copy())
        else:
            print("new----------")
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            tmp = tmp.astype(float)
            tmp = tmp / tmp.sum()
            blurred = np.zeros_like(self.original)
            #cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blurred[:, :, 0] = np.array(signal.fftconvolve(self.original[:, :, 0], tmp, 'same'))
            blurred[:, :, 1] = np.array(signal.fftconvolve(self.original[:, :, 1], tmp, 'same'))
            blurred[:, :, 2] = np.array(signal.fftconvolve(self.original[:, :, 2], tmp, 'same'))
            # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            result.append(blurred)
        self.result = result
        self.psf = psf
        if show or save:
            self.__plot_canvas(show, save)

    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                        axes[i].imshow(self.result[i])
            else:
                plt.axis('off')

                plt.imshow(self.result[0])
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                file_name = self.image_path.split('/')[-1]
                split_name = file_name.split('.')
                main_name = split_name[0]
                ext_name = split_name[1]
                path_out = os.path.join(self.path_to_save, main_name)
                if not os.path.exists(path_out):
                    try:
                        os.makedirs(path_out)
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                print(os.path.join(path_out, 'sharp', '.', ext_name))
                cv2.imwrite(os.path.join(path_out, 'sharp'+'.'+ext_name), self.original * 255)
                cnt = 0
                for i, (y, k) in enumerate(zip(self.result, self.PSFs)):
                    cv2.imwrite(os.path.join(path_out, 'blurry_'+str(i)+'.'+ext_name), y * 255)
                    cv2.imwrite(os.path.join(path_out, 'kernel_'+str(i)+'.'+ext_name), k / k.max() * 255)

            elif show:
                plt.show()


if __name__ == '__main__':
    folder_src = '/media/DATA/data/blurred_sharp_org/blurred_sharp/sharp'
    # this folder will include sharp image, kernels, and blurry images
    folder_dst = '/media/DATA/data/blurred_sharp_org/blurred_sharp/mine/multi' 

    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    num_obs = 3
    for path in os.listdir(folder_src):
        print(path)
        psf_set = []
        for i in range(num_obs):
            trajectory = Trajectory(canvas=25, max_len=21, expl=np.random.choice(params)).fit()
            psf = PSF(canvas=25, trajectory=trajectory).fit()
            psf_set.append(psf[-1]) # get the last one

        BlurImage(os.path.join(folder_src, path), PSFs=psf_set,
                  path_to_save=folder_dst,
                  part=None).\
            blur_image(save=True)
