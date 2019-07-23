import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm

from utils import compute_eer

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, patience, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logger=None):
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
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-7)
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.device = next(self.model.parameters()).device
		self.logger = logger

		if self.valid_loader is not None:
			self.history = {'train_loss': [], 'train_loss_batch': [], 'valid_loss': []}
		else:
			self.history = {'train_loss': [], 'train_loss_batch': []}

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader))
			train_loss_epoch=0.0

			for t, batch in train_iter:
				train_loss = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				train_loss_epoch+=train_loss
				if self.logger:
					self.logger.add_scalar('Train Loss', train_loss, self.total_iters)
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))

			print('Total train loss: {:0.4f}'.format(self.history['train_loss'][-1]))

			if self.valid_loader is not None:

				scores, labels = None, None

				for t, batch in enumerate(self.valid_loader):
					scores_batch, labels_batch = self.valid(batch)

					try:
						scores = np.concatenate([scores, scores_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						scores, labels = scores_batch, labels_batch

				self.history['valid_loss'].append(compute_eer(labels, scores))

				if self.logger:
					self.logger.add_scalar('Valid EER', self.history['valid_loss'][-1], np.min(self.history['valid_loss']), self.total_iters)
					self.logger.add_scalar('Best valid EER', np.min(self.history['valid_loss']), self.total_iters)
					self.logger.add_pr_curve('Valid. ROC', labels=labels, predictions=scores, global_step=self.total_iters)

				print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			self.scheduler.step(self.history['valid_loss'][-1])

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

		utterances_clean, utterances_attack, y_clean, y_attack = batch

		utterances = torch.cat([utterances_clean, utterances_attack],0)
		y = torch.cat([y_clean, y_attack],0)

		ridx = np.random.randint(utterances.size(3)//2, utterances.size(3))
		utterances = utterances[:,:,:,:ridx]

		if self.cuda_mode:
			utterances, y = utterances.to(self.device), y.to(self.device)

		pred = self.model.forward(utterances)

		loss = torch.nn.BCEWithLogitsLoss()(pred, y)

		loss.backward()
		self.optimizer.step()
		return loss.item()

	def valid(self, batch):

		self.model.eval()

		with torch.no_grad():

			utterances_clean, utterances_attack, y_clean, y_attack = batch

			utterances = torch.cat([utterances_clean, utterances_attack],0)
			y = torch.cat([y_clean, y_attack],0)

			if self.cuda_mode:
				utterances, y = utterances.to(self.device), y.to(self.device)

			pred = self.model.forward(utterances)

		return torch.sigmoid(pred).detach().cpu().numpy().squeeze(), y.cpu().numpy().squeeze()

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
				self.model = self.model.to(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))
