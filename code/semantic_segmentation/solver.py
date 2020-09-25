import os, datetime
import csv

import torch
import torchvision
import torch.nn.functional as F

from utils.config import *
from utils.evaluation import *
from utils.networks import U_Net, R2U_Net, AttU_Net, R2AttU_Net


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader  = test_loader

		self.dataset = config.dataset

		# Models
		self.unet = None
		self.arma = config.use_arma
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.model_name = config.model_name +'.pkl'
		self.mode = config.mode

		use_cuda  = config.use_cuda and torch.cuda.is_available()
		self.device = torch.device('cuda' if use_cuda else 'cpu')
		print("Device    : ", self.device)

		self.model_type = config.model_type
		self.steps = config.steps
		self.tensorboard = config.tensorboard
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type == "U_Net":
			self.unet = U_Net(in_channels = dataset_channel[self.dataset], out_channels = 1, factor = dataset_factor[self.dataset], arma = self.arma, arma2 = True if self.arma and self.dataset == "ISIC" else False )
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(in_channels = dataset_channel[self.dataset], out_channels = 1, arma = self.arma, steps = self.steps)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(in_channels = dataset_channel[self.dataset], out_channels = 1, factor = dataset_factor[self.dataset], arma = self.arma, arma2 = True if self.arma and self.dataset == "ISIC" else False )
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(in_channels = dataset_channel[self.dataset], out_channels = 1, arma = self.arma, steps = self.steps)

		self.optimizer = torch.optim.Adam(self.unet.parameters(), self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		#print(model)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, self.model_name)

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			# self.unet.load_state_dict(torch.load(unet_path))
			print('Resume: Loaded %s from %s'%(self.model_type, unet_path))
			self.test()
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			
			for epoch in range(self.num_epochs):
				self.tensorboard.add_scalar('lr', lr, epoch + 1)

				self.unet.train(True)
				epoch_loss = 0
				valid_loss = 0
				test_loss  = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth
					batch_size = images.size(0)
					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = torch.sigmoid(self.unet(images))
					# SR_probs = torch.sigmoid(SR)
					SR_flat = SR.view(SR.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					for j in range(batch_size):
						acc += get_accuracy(SR[j],GT[j])
						SE += get_sensitivity(SR[j],GT[j])
						SP += get_specificity(SR[j],GT[j])
						PC += get_precision(SR[j],GT[j])
						F1 += get_F1(SR[j],GT[j])
						JS += get_JS(SR[j],GT[j])
						DC += get_DC(SR[j],GT[j])
					length += images.size(0)
					
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length

				# Print the log info
				
				self.tensorboard.add_scalar('train/acc', acc,        epoch+1)
				self.tensorboard.add_scalar('train/nll', epoch_loss, epoch+1)
				self.tensorboard.add_scalar('train/SE', SE, epoch+1)
				self.tensorboard.add_scalar('train/SP', SP, epoch+1)
				self.tensorboard.add_scalar('train/PC', PC, epoch+1)
				self.tensorboard.add_scalar('train/F1', F1, epoch+1)
				self.tensorboard.add_scalar('train/JS', JS, epoch+1)
				self.tensorboard.add_scalar('train/DC', DC, epoch+1)
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))

				if self.dataset == 'ISIC':
					lr = lr * 0.98
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				else:
					if (epoch+1) % 10 == 0:
						lr = lr * 0.98
						for param_group in self.optimizer.param_groups:
							param_group['lr'] = lr
						print ('Decay learning rate to lr: {}.'.format(lr))

				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT) in enumerate(self.valid_loader):
					batch_size = images.size(0)

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = torch.sigmoid(self.unet(images))

					SR_flat = SR.view(SR.size(0),-1)
					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					valid_loss += loss.item()

					for j in range(batch_size):
						acc += get_accuracy(SR[j],GT[j])
						SE += get_sensitivity(SR[j],GT[j])
						SP += get_specificity(SR[j],GT[j])
						PC += get_precision(SR[j],GT[j])
						F1 += get_F1(SR[j],GT[j])
						JS += get_JS(SR[j],GT[j])
						DC += get_DC(SR[j],GT[j])
						
					length += images.size(0)

					if length % dataset_save_length1[self.dataset] == 0 and epoch % dataset_save_epoch1[self.dataset] == 0:
						idx = str(length // dataset_save_length1[self.dataset])
						SR_e = SR.expand(images.size())
						GT_e = GT.expand(images.size())
						images = torch.unsqueeze(images[0], 0)
						SR_e = torch.unsqueeze(SR_e[0], 0)
						GT_e = torch.unsqueeze(GT_e[0], 0)
						img = torchvision.utils.make_grid(torch.cat([images, SR_e, GT_e], 0))
						self.tensorboard.add_image('valid/%s'%(idx), img, epoch)
						valid_image_folder = os.path.join(self.result_path, 'valid/%s'%(idx))
						if not os.path.exists(valid_image_folder): os.makedirs(valid_image_folder)
						torchvision.utils.save_image(img.data.cpu(), os.path.join(valid_image_folder, '%s_valid_idx%s_epoch%d.png'%(self.model_type, idx, epoch)))

						
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				unet_score = JS + DC


				
				self.tensorboard.add_scalar('valid/acc', acc, epoch+1)
				self.tensorboard.add_scalar('valid/nll', valid_loss, epoch+1)
				self.tensorboard.add_scalar('valid/SE', SE, epoch+1)
				self.tensorboard.add_scalar('valid/SP', SP, epoch+1)
				self.tensorboard.add_scalar('valid/PC', PC, epoch+1)
				self.tensorboard.add_scalar('valid/F1', F1, epoch+1)
				self.tensorboard.add_scalar('valid/JS', JS, epoch+1)
				self.tensorboard.add_scalar('valid/DC', DC, epoch+1)
				self.tensorboard.add_scalar('valid/unet_score', unet_score, epoch+1)
				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				
				

				# Save Best U-Net model
				if unet_score > best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)

					#==========================================
					#test:
					self.unet.train(False)
					self.unet.eval()

					test_acc = 0.	# Accuracy
					test_SE = 0.		# Sensitivity (Recall)
					test_SP = 0.		# Specificity
					test_PC = 0. 	# Precision
					test_F1 = 0.		# F1 Score
					test_JS = 0.		# Jaccard Similarity
					test_DC = 0.		# Dice Coefficient
					test_length=0

					for i, (test_images, test_GT) in enumerate(self.test_loader):
						test_batch_size = test_images.size(0)

						test_images = test_images.to(self.device)
						test_GT = test_GT.to(self.device)
						test_SR = torch.sigmoid(self.unet(test_images))

						test_SR_flat = test_SR.view(test_SR.size(0),-1)
						test_GT_flat = test_GT.view(test_GT.size(0),-1)
						test_loss = self.criterion(test_SR_flat,test_GT_flat)
						test_loss += test_loss.item()

						for j in range(test_batch_size):
							test_acc += get_accuracy(test_SR[j],test_GT[j])
							test_SE += get_sensitivity(test_SR[j],test_GT[j])
							test_SP += get_specificity(test_SR[j],test_GT[j])
							test_PC += get_precision(test_SR[j],test_GT[j])
							test_F1 += get_F1(test_SR[j],test_GT[j])
							test_JS += get_JS(test_SR[j],test_GT[j])
							test_DC += get_DC(test_SR[j],test_GT[j])
								
						test_length += test_images.size(0)

						if test_length % dataset_save_length2[self.dataset] == 0 and epoch % dataset_save_epoch2[self.dataset] == 0:
							idx = str(test_length // dataset_save_length2[self.dataset])
							SR_e = test_SR.expand(test_images.size())
							GT_e = test_GT.expand(test_images.size())
							test_images = torch.unsqueeze(test_images[0], 0)
							SR_e = torch.unsqueeze(SR_e[0], 0)
							GT_e = torch.unsqueeze(GT_e[0], 0)
							img = torchvision.utils.make_grid(torch.cat([test_images, SR_e, GT_e], 0))
							self.tensorboard.add_image('test_for_best/%s'%(idx), img, epoch)
							test_image_folder = os.path.join(self.result_path, 'test_for_best/%s'%(idx))
							if not os.path.exists(test_image_folder): os.makedirs(test_image_folder)
							torchvision.utils.save_image(img.data.cpu(), os.path.join(test_image_folder, '%s_test_idx%s_epoch%d.png'%(self.model_type, idx, epoch)))

							
					test_acc = test_acc/test_length
					test_SE  = test_SE /test_length
					test_SP  = test_SP /test_length
					test_PC  = test_PC /test_length
					test_F1  = test_F1 /test_length
					test_JS  = test_JS /test_length
					test_DC  = test_DC /test_length
					test_unet_score = test_JS + test_DC


					self.tensorboard.add_scalar('test/acc', test_acc, epoch+1)
					self.tensorboard.add_scalar('test/nll', test_loss, epoch+1)
					self.tensorboard.add_scalar('test/SE', test_SE, epoch+1)
					self.tensorboard.add_scalar('test/SP', test_SP, epoch+1)
					self.tensorboard.add_scalar('test/PC', test_PC, epoch+1)
					self.tensorboard.add_scalar('test/F1', test_F1, epoch+1)
					self.tensorboard.add_scalar('test/JS', test_JS, epoch+1)
					self.tensorboard.add_scalar('test/DC', test_DC, epoch+1)
					self.tensorboard.add_scalar('test/unet_score', test_unet_score, epoch+1)
					print('[Test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(test_acc,test_SE,test_SP,test_PC,test_F1,test_JS,test_DC))
					#==========================================

					
			#===================================== Test ====================================#
			del self.unet
			del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0

			for i, (images, GT) in enumerate(self.test_loader):
				batch_size = images.size(0)

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = torch.sigmoid(self.unet(images))

				for j in range(batch_size):
					acc += get_accuracy(SR[j],GT[j])
					SE += get_sensitivity(SR[j],GT[j])
					SP += get_specificity(SR[j],GT[j])
					PC += get_precision(SR[j],GT[j])
					F1 += get_F1(SR[j],GT[j])
					JS += get_JS(SR[j],GT[j])
					DC += get_DC(SR[j],GT[j])
						
				length += images.size(0)
				if length % dataset_save_length3[self.dataset] == 0:
					idx = str(length)
					SR_e = SR.expand(images.size())
					GT_e = GT.expand(images.size())

					images = torch.unsqueeze(images[0], 0)
					SR_e = torch.unsqueeze(SR_e[0], 0)
					GT_e = torch.unsqueeze(GT_e[0], 0)
					images = torchvision.utils.make_grid(torch.cat([images, SR_e, GT_e], 0))
					self.tensorboard.add_image('final_test/%s'%(idx), images, length)
					image_folder = os.path.join(self.result_path, 'final_test/%s'%(idx))
					if not os.path.exists(image_folder): os.makedirs(image_folder)
					torchvision.utils.save_image(images.data.cpu(), os.path.join(image_folder, '%s_final_test_idx%s.png'%(self.model_type, idx)))

					
			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length
			unet_score = JS + DC

			f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
			f.close()



	def test(self):
		unet_path = os.path.join(self.model_path, self.model_name)
		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))
		self.unet.train(False)
		self.unet.eval()

		acc = 0.	# Accuracy
		SE = 0.		# Sensitivity (Recall)
		SP = 0.		# Specificity
		PC = 0. 	# Precision
		F1 = 0.		# F1 Score
		JS = 0.		# Jaccard Similarity
		DC = 0.		# Dice Coefficient
		length=0

		for i, (images, GT) in enumerate(self.test_loader):
			batch_size = images.size(0)

			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = torch.sigmoid(self.unet(images))

			for j in range(batch_size):
				acc += get_accuracy(SR[j],GT[j])
				SE += get_sensitivity(SR[j],GT[j])
				SP += get_specificity(SR[j],GT[j])
				PC += get_precision(SR[j],GT[j])
				F1 += get_F1(SR[j],GT[j])
				JS += get_JS(SR[j],GT[j])
				DC += get_DC(SR[j],GT[j])
					
			length += images.size(0)
			if length % dataset_save_length3[self.dataset] == 0:
				idx = str(length)
				SR_e = SR.expand(images.size())
				GT_e = GT.expand(images.size())

				images = torch.unsqueeze(images[0], 0)
				SR_e = torch.unsqueeze(SR_e[0], 0)
				GT_e = torch.unsqueeze(GT_e[0], 0)
				images = torchvision.utils.make_grid(torch.cat([images, SR_e, GT_e], 0))
				self.tensorboard.add_image('final_test/%s'%(idx), images, length)
				image_folder = os.path.join(self.result_path, 'final_test/%s'%(idx))
				if not os.path.exists(image_folder): os.makedirs(image_folder)
				torchvision.utils.save_image(images.data.cpu(), os.path.join(image_folder, '%s_final_test_idx%s.png'%(self.model_type, idx)))

				
		acc = acc/length
		SE = SE/length
		SP = SP/length
		PC = PC/length
		F1 = F1/length
		JS = JS/length
		DC = DC/length
		unet_score = JS + DC

		f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
		wr = csv.writer(f)
		wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
		f.close()
			
