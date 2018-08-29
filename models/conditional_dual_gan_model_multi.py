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
from blur_estimation.blur_est import get_K

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3

class ConditionalDualGANMulti(BaseModel):
	def name(self):
		return 'ConditionalDualGANMulti'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		# define tensors
		self.input_x = self.Tensor(opt.batchSize, opt.input_nc,
                                           opt.fineSize, opt.fineSize)
		self.input_y = self.Tensor(opt.batchSize, opt.output_nc,
                                           opt.fineSize, opt.fineSize)

                # blur kernels (single channel)
		self.input_k = self.Tensor(opt.batchSize, 1, opt.kerSize, opt.kerSize) # generalize later

		self.init_state = self.Tensor(opt.batchSize, opt.output_nc,
                                           opt.fineSize, opt.fineSize)

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

			self.netB = networks.define_B(opt.input_nc, opt.ndf,
                                                      opt.which_model_netD,
                                                      opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)

                        # define fusion network
                        self.netFusion = networks.define_fusion(opt.input_nc * 1, opt.output_nc, opt.ngf,
                                              opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)

		if not self.isTrain or opt.continue_train:
			self.load_network(self.netG, 'G', opt.which_epoch)
			if self.isTrain:
				self.load_network(self.netD, 'D', opt.which_epoch)

		if self.isTrain:
			self.old_lr = opt.lr

			# initialize optimizers
			self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                            lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                            lr=opt.lr, betas=(opt.beta1, 0.999))

			self.optimizer_fusion = torch.optim.Adam(self.netFusion.parameters(),
                                                                 lr=opt.lr, betas=(opt.beta1, 0.999))


			self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1

			# define loss functions
			self.discLoss, self.contentLoss = init_loss(opt, self.Tensor)

		print('---------- Networks initialized -------------')
		networks.print_network(self.netG)
		if self.isTrain:
			networks.print_network(self.netD)

	def set_input(self, input):
                def to_cuda(cpu_data):
                        return [d.cuda() for d in cpu_data]
                        
                # Already tensors
		self.in_y = to_cuda(input['blurry_set']) # blurry
		self.real_x = input['sharp'].cuda()
                self.in_k = to_cuda(input['kernel_set']) # kernel

                # why need copy??
		#self.input_A.resize_(input_A.size()).copy_(input_A)
		#self.input_B.resize_(input_B.size()).copy_(input_B)
		#self.input_K.resize_(input_K.size()).copy_(input_K)

                ## initial state as zero, size the same size as image (will change later)
		self.init_state = torch.zeros_like(self.real_x)
                self.obs_num = len(self.in_y)

	def forward(self):
                ## why need this?
		#self.real_A = Variable(self.input_A)
		#self.real_B = Variable(self.input_B)
		#self.real_K = Variable(self.input_K) # x, {y_, k_i} -> {out_i} for cost computation


                state = self.init_state

                out_x = []
                out_y = []
                # recurrent forwarding
                #for i in range(self.obs_num)):
                for i, (yi, ki) in enumerate(zip(self.in_y, self.in_k)):
                        h_x = self.netG.forward(yi) # hidden state for x
                        # fusion function
                        # state = self.netFusion(h_x, state)
                        #in_cat = torch.cat((h_x, state), 1)
                        if i == 0:
                                in_cat = h_x
                        else:
                                in_cat = (h_x + state) / 2.0
                        state = self.netFusion(in_cat)
                        #state = (h_x + state) / 2 # simple average fusion
                        fusion_x = state # currently an identity function
                        # now using the true kernel for constructing the observation process
                        reblur_A = self.netB.forward(fusion_x, ki.unsqueeze(0))
                        out_x.append(fusion_x)
                        out_y.append(reblur_A)
                
                self.out_x = out_x
                self.out_y = out_y


	# no backprop gradients
	def test(self):
		self.real_A = Variable(self.input_A, volatile=True)
		self.fake_B = self.netG.forward(self.real_A)
		self.real_B = Variable(self.input_B, volatile=True)

	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D(self):
                # compute loss for all the observations
                self.loss_D = 0
                for xi, yi in zip(self.out_x, self.out_y):
                        # real_B is the real sharp image
                        self.loss_D += self.discLoss.get_loss(self.netD, None, xi.detach(), self.real_x)
		self.loss_D.backward(retain_graph=True)

	def backward_G(self):
                self.loss_G_GAN = 0
                self.loss_G_Content = 0
                for xi, yi in zip(self.out_x, self.out_y):
                        self.loss_G_GAN += self.discLoss.get_g_loss(self.netD, None, xi)
                        # Second, G(A) = B
                        self.loss_G_Content += self.contentLoss.get_loss(xi, self.real_x) * self.opt.lambda_A
		self.loss_G = self.loss_G_GAN + self.loss_G_Content
		self.loss_G.backward(retain_graph=True)

        # B induces a loss while E is not
	def backward_reblur(self):
                L = nn.MSELoss()
                self.loss_obs = 0
                for yi_reblur, yi_true in zip(self.out_y, self.in_y):
                        self.loss_obs += L(yi_reblur, yi_true) * self.opt.lambda_C
                        #self.loss_obs = self.contentLoss.get_loss(self.reblur_A, self.real_A.detach()) * self.opt.lambda_C
		self.loss_obs.backward(retain_graph=True)

	def optimize_parameters(self):
                # saved everything
		self.forward()

		for iter_d in xrange(self.criticUpdates):
			self.optimizer_D.zero_grad()
			self.backward_D()
			self.optimizer_D.step()

		self.optimizer_G.zero_grad()
		self.optimizer_fusion.zero_grad()

                ## obs cost
                self.backward_reblur()
		self.backward_G()
                self.optimizer_G.step()
                self.optimizer_fusion.step()



        # for visualization, model saving
	def get_current_errors(self):
		return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                    ('G_L1', self.loss_G_Content.data[0]),
                                    ('D_real+fake', self.loss_D.data[0]),
                                    #('G_psf', self.loss_G_psf.data[0]),
                                    #('D_real+fake_psf', self.loss_D_psf.data[0]),
                                    ('reblur_err', self.loss_obs.data[0]),
							])

	def get_current_visuals(self):
                vis = OrderedDict()
		sharp_real = util.tensor2im(self.real_x.data)
                vis['Sharp_Train'] = sharp_real
		sharp_est = util.tensor2im(self.out_x[-1].data) # the last estimate

                vis['Restored_Train'] = sharp_est
		reblur = util.tensor2im(self.out_y[-1].data) # the last estimate
                vis['reblur'] = reblur
                # in_y
                # in_k
                for i, (yi, ki) in enumerate(zip(self.in_y, self.in_k)):
                        blurry = util.tensor2im(yi.data)
                        kernel = util.tensor2psf(ki.squeeze(0).data) # remove the singlton batch dim
                        vis['blurry'+str(i)] = blurry
                        vis['kernel'+str(i)] = kernel

                return vis
                #reblur_A = util.tensor2im(self.reblur_A.data)
                """
		return OrderedDict([('Blurred_Train', real_A), 
                                    ('Restored_Train', fake_B), 
                                    ('Sharp_Train', real_B), 
                                    ('Real_ker', kernel),
                                    #('Est_ker', kernel_est), 
                                    ('reblur', reblur_A)])
                """

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
