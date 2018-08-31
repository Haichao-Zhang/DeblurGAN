import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from PIL import Image
import cv2
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.no_clip = True
opt.resize_or_crop = 'scale_width'
opt.dataset_mode = 'multi_test'
opt.model = 'test'

print(opt.phase)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % ('db', opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break
	counter = i
	model.set_input(data)
	model.test()
	visuals = model.get_current_visuals()
	#avgPSNR += PSNR(visuals['fake_B'],visuals['real_B'])
	#pilFake = Image.fromarray(visuals['fake_B'])
	#pilReal = Image.fromarray(visuals['real_B'])
	#avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
	#img_path = model.get_image_paths()
	#print('process image... %s' % img_path)
        # later can write results into a webpage
	#visualizer.save_images(webpage, visuals, img_path)
        img = np.zeros((visuals['Restored_Train'].shape[0], visuals['Restored_Train'].shape[1], 3))
        #img[:,:,0] = visuals['Restored_Train']
        #img[:,:,1] = visuals['Restored_Train']
        #img[:,:,2] = visuals['Restored_Train']
        #cv2.imwrite('./db.bmp', img)
        print(visuals['Restored_Train'].shape)
        cv2.imwrite('./db.bmp', cv2.cvtColor(visuals['Restored_Train'], cv2.COLOR_RGB2BGR))
        cv2.imwrite('./b1.bmp', cv2.cvtColor(visuals['blurry1'], cv2.COLOR_RGB2BGR))

#avgPSNR /= counter
#avgSSIM /= counter
#print('PSNR = %f, SSIM = %f' %
#				  (avgPSNR, avgSSIM))

webpage.save()
