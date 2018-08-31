from .conditional_gan_model import ConditionalGAN
from .conditional_gan_obs_model import ConditionalGANObs
from .conditional_dual_gan_model import ConditionalDualGAN
from .conditional_dual_gan_model_multi import ConditionalDualGANMulti


def create_model(opt):
	model = None
        print("888888888888 %s " % opt.model)
	if opt.model == 'test':
                if opt.dataset_mode == 'single':
                        from .test_model import TestModel
                        model = TestModel()
                elif opt.dataset_mode == 'multi_test':
                        print("multi-test-----")
                        #from .test_model_multi import TestModelMulti
                        #model = TestModelMulti()
                        model = ConditionalDualGANMulti()
        elif opt.dataset_mode == 'multi':
                print("888888888888 %s " % 'Multi')
		model = ConditionalDualGANMulti()
	else:
		#model = ConditionalGANObs()
		#model = ConditionalGAN()
		model = ConditionalDualGAN()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
