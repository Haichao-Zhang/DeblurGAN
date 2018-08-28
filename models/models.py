from .conditional_gan_model import ConditionalGAN
from .conditional_gan_obs_model import ConditionalGANObs
from .conditional_dual_gan_model import ConditionalDualGAN

def create_model(opt):
	model = None
        print("888888888888 %s " % opt.model)
	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		from .test_model import TestModel
		model = TestModel()
	else:
		#model = ConditionalGANObs()
		#model = ConditionalGAN()
		model = ConditionalDualGAN()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
