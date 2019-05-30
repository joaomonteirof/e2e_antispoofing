import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm

class TrainLoop_mcc(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15, 100, 180], gamma=0.1)
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0

		if self.valid_loader is not None:
			self.history = {'train_loss': [], 'train_loss_batch': [], 'valid_loss': []}
		else:
			self.history = {'train_loss': [], 'train_loss_batch': []}

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			np.random.seed()
			train_iter = tqdm(enumerate(self.train_loader))
			train_loss_epoch=0.0

			for t, batch in train_iter:
				train_loss = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				train_loss_epoch+=train_loss
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))

			print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			if self.valid_loader is not None:

				tot_correct = 0.
				total = 0.
				
				for t, batch in enumerate(self.valid_loader):
					correct, total_ = self.valid(batch)
					tot_correct += correct
					total += total_

				self.history['valid_loss'].append(1.-tot_correct/total)

				print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			self.scheduler.step()

			print('Current LR: {}'.format(self.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1]):
				self.checkpointing()

		print('Training done!')

		if self.valid_loader is not None:
			print('Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			return np.min(self.history['valid_loss'])

	def train_step(self, batch):

		self.model.train()
		self.optimizer.zero_grad()

		utterances_a, utterances_b, y_a, y_b = batch

		utterances = torch.cat([utterances_a, utterances_b],0)
		y = torch.cat([y_a, y_b],0).squeeze()

		ridx = np.random.randint(utterances.size(3)//2, utterances.size(3))
		utterances = utterances[:,:,:,:ridx]

		if self.cuda_mode:
			utterances, y = utterances.cuda(), y.cuda()

		pred = self.model.forward(utterances)

		loss = torch.nn.CrossEntropyLoss()(pred, y)

		loss.backward()
		self.optimizer.step()
		return loss.item()

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			utterances_a, utterances_b, y_a, y_b = batch

			utterances = torch.cat([utterances_a, utterances_b],0)
			y = torch.cat([y_a, y_b],0).squeeze()

			if self.cuda_mode:
				utterances, y = utterances.cuda(), y.cuda()

			pred = F.softmax(self.model.forward(utterances), dim=1).max(1)[1].long()
			correct = pred.eq(y)

		return correct.detach().sum().item(), correct.size(0)

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'scheduler_state': self.scheduler.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model = self.model.cuda()

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))
