from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)

        # define fusion network
        self.netFusion = networks.define_fusion(opt.input_nc * 2, opt.output_nc, opt.ngf,
                                                opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)


        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)
        self.load_network(self.net_fusion, 'fusion', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.net_fusion)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_paths']

        def to_cuda(cpu_data):
            return [d.cuda() for d in cpu_data]
                        
        # Already tensors
        self.in_y = to_cuda(input['blurry_set']) # blurry
        # self.real_x = input['sharp'].cuda()
        #self.in_k = to_cuda(input['kernel_set']) # kernel
        self.init_state = torch.zeros_like(self.in_y[0])
        self.obs_num = len(self.in_y)

    def test(self):
        self.deblur = self.forward(self.in_y)

    def forward(self):
        state = self.init_state

        out_x = []
        out_y = []
        # recurrent forwarding
        #for i in range(self.obs_num)):
        for i, (yi, ki) in enumerate(zip(self.in_y, self.in_k)):
            h_x = self.netG.forward(yi) # hidden state for x
            # fusion function
            # state = self.netFusion(h_x, state)
            if i == 0:
                in_cat = torch.cat((h_x, h_x), 1)
            else:
                in_cat = torch.cat((h_x, state), 1)
            state = self.netFusion(in_cat)
            fusion_x = state # currently an identity function
            out_x.append(fusion_x)
            out_y.append(reblur_A)
        self.out_x = out_x # keep the last estimation
        return out_x[-1]

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
    
        vis = OrderedDict()

        sharp_est = util.tensor2im(self.out_x[-1].data) # the last estimate

        vis['Restored_Train'] = sharp_est
 
        for i, yi in enumerate(self.in_y):
            blurry = util.tensor2im(yi.data)
            #kernel = util.tensor2psf(ki.squeeze(0).data) # remove the singlton batch dim
            vis['blurry'+str(i)] = blurry
            #vis['kernel'+str(i)] = kernel

        return vis
