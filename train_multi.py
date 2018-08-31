import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
import cv2

def train(opt, data_loader, model, visualizer):
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
                        #print("data %s " % data['K'])
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'],results['Sharp_Train'])
				print('PSNR on Train = %f' %
					  (psnrMetric))
				visualizer.display_current_results(results,epoch)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				if opt.display_id > 0:
					visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				model.save('latest')
                                
                        if opt.phase == 'test':
                                visuals = model.get_current_visuals()
                                print("dir  %s" % opt.results_dir)
                                cv2.imwrite(opt.results_dir + '/db.bmp', cv2.cvtColor(visuals['Restored_Train'], cv2.COLOR_RGB2BGR))
                                cv2.imwrite(opt.results_dir + '/b1.bmp', cv2.cvtColor(visuals['blurry1'], cv2.COLOR_RGB2BGR))

                if opt.phase == 'test':
                        break

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save('latest')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()
			
opt = TrainOptions().parse()
opt.batchSize = 1
# opt.model = 'multi'
opt.dataset_mode = 'multi'
data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, data_loader, model, visualizer)
