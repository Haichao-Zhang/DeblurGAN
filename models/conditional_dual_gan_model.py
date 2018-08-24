import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalDualGAN(BaseModel):
	def name(self):
		return 'ConditionalGANModelObs'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
								   opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
								   opt.fineSize, opt.fineSize)

                # blur kernels (single channel)
		self.input_K = self.Tensor(opt.batchSize, 1, opt.kerSize, opt.kerSize) # generalize later

		# load/define networks
		#Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		use_parallel = not opt.gan_type == 'wgan-gp'
		self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
									  opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)
		if self.isTrain:
			use_sigmoid = opt.gan_type == 'gan'
			self.netD = networks.define_D(opt.output_nc, opt.ndf,
										  opt.which_model_netD,
										  opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)

                        # E is a generator for blur kernel
			self.netE = networks.define_E(1, opt.output_nc * 2, opt.ndf,
                                                      opt.which_model_netD,
                                                      opt.norm,
                                                      not opt.no_dropout,
                                                      self.gpu_ids, use_parallel, opt.learn_residual)
			self.netB = networks.define_B(opt.input_nc, opt.ndf,
										  opt.which_model_netD,
										  opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)

                        # discriminator for psf (single channel input)
			self.netD_psf = networks.define_D(1, opt.ndf,
										  opt.which_model_netD,
										  opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)


		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.fake_AB_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))

			self.optimizer_E = torch.optim.Adam(self.netE.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))

			self.optimizer_B = torch.optim.Adam(self.netB.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			
                        self.optimizer_D_psf = torch.optim.Adam(self.netD_psf.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))

			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1

			# define loss functions
			self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		if self.isTrain:
			networks.print_network(self.netD)
		print('-----------------------------------------------')

	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		input_A = input['A' if AtoB else 'B']
		input_B = input['B' if AtoB else 'A']
		self.input_A.resize_(input_A.size()).copy_(input_A)
		self.input_B.resize_(input_B.size()).copy_(input_B)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B)
		self.real_K = Variable(self.input_K)

                # blur est and re-blur
                # self.blur_est = self.netE.forward(self.real_A)
                # input_data = torch.cat((self.real_A, self.fake_B), 1)
                ## perform convolutional data interaction rather than concat
                conv2_t = torch.nn.functional.conv2d 
                input_data = conv2_t(self.real_A, self.fake_B, padding=self.real_A.size(2) / 2)

                self.blur_est = self.netE.forward(input_data)
                #self.reblur_A = self.netB.forward(self.fake_B, self.blur_est)
                self.reblur_A = self.netB.forward(self.fake_B.detach(), self.blur_est)

	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
		self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B)

		self.loss_D.backward(retain_graph=True)

	def backward_G(self):
		self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B)
		# Second, G(A) = B
		self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A

		self.loss_G = self.loss_G_GAN + self.loss_G_Content

		self.loss_G.backward(retain_graph=True)

        # B induces a loss while E is not
	def backward_B(self):
                # L = nn.MSELoss()
 		# self.loss_obs = L(self.reblur_A, self.real_A.detach()) * self.opt.lambda_C
                self.loss_obs = self.contentLoss.get_loss(self.reblur_A, self.real_A.detach()) * self.opt.lambda_C

		self.loss_obs.backward(retain_graph=True)


	def backward_D_psf(self):
		self.loss_D_psf = self.discLoss.get_loss(self.netD_psf, None, self.blur_est.unsqueeze(0), self.real_K)

		self.loss_D_psf.backward(retain_graph=True)

	def backward_G_psf(self):
		self.loss_G_GAN_psf = self.discLoss.get_g_loss(self.netD_psf, None,  self.blur_est.unsqueeze(0))
		# Second, G(A) = B
                # L = nn.MSELoss()
		#self.loss_G_Content_psf = L(self.blur_est.unsqueeze(0), self.real_K) * self.opt.lambda_K
                k_est = self.blur_est.unsqueeze(0)
                k_est3 = torch.cat((k_est, k_est, k_est), 1)
                k_real3 = torch.cat((self.real_K, self.real_K, self.real_K), 1)

		self.loss_G_Content_psf = self.contentLoss.get_loss(
                        k_est3, k_real3) * self.opt.lambda_K

		self.loss_G_psf = self.loss_G_GAN_psf + self.loss_G_Content_psf

		self.loss_G_psf.backward(retain_graph=True)


	def optimize_parameters(self):
		self.forward()

		for iter_d in xrange(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()
                        self.optimizer_D_psf.zero_grad()
			self.backward_D_psf()
			self.optimizer_D_psf.step()

		self.optimizer_G.zero_grad()
                self.optimizer_E.zero_grad()
                self.optimizer_B.zero_grad()
                ## obs cost
                self.backward_B()
		self.backward_G()
                self.backward_G_psf()
		self.optimizer_G.step()
                self.optimizer_E.step()
                self.optimizer_B.step()

	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
							('G_L1', self.loss_G_Content.data[0]),
							('D_real+fake', self.loss_D.data[0]),
							('G_pdf', self.loss_G_psf.data[0]),
							('D_real+fake_pdf', self.loss_D_psf.data[0]),
							('reblur_err', self.loss_obs.data[0])
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		fake_B = util.tensor2im(self.fake_B.data)
		real_B = util.tensor2im(self.real_B.data)
                kernel = util.tensor2psf(self.blur_est.data)
                reblur_A = util.tensor2im(self.reblur_A.data)
		return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B), ('Est_ker', kernel), ('reblur', reblur_A)])

	def save(self, label):
		self.save_network(self.netG, 'G', label, self.gpu_ids)
		self.save_network(self.netD, 'D', label, self.gpu_ids)

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer_D.param_groups:
			param_group['lr'] = lr
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
