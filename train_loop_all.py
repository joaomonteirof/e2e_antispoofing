import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm

from utils import compute_eer

class TrainLoop(object):

	def __init__(self, model_la, model_pa, model_mix, optimizer_la, optimizer_pa, optimizer_mix, train_loader, valid_loader, patience, device, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model_la = model_la
		self.model_pa = model_pa
		self.model_mix = model_mix
		self.optimizer_la = optimizer_la
		self.optimizer_pa = optimizer_pa
		self.optimizer_mix = optimizer_mix
		self.scheduler_la = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_la, factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-7)
		self.scheduler_pa = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_pa, factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-7)
		self.scheduler_mix = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_mix, factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-7)
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.device = device

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

				print('Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_loss'][-1], np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			self.scheduler_la.step(self.history['valid_loss'][-1])
			self.scheduler_pa.step(self.history['valid_loss'][-1])
			self.scheduler_mix.step(self.history['valid_loss'][-1])

			print('Current LRs (LA, PA, Mixture): {}, {}, {}'.format(self.optimizer_la.param_groups[0]['lr'], self.optimizer_pa.param_groups[0]['lr'], self.optimizer_mix.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0 or self.history['valid_loss'][-1] < np.min([np.inf]+self.history['valid_loss'][:-1]):
				self.checkpointing()

		print('Training done!')

		if self.valid_loader is not None:
			print('Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_loss']), 1+np.argmin(self.history['valid_loss'])))

			return np.min(self.history['valid_loss'])

	def train_step(self, batch):

		self.model_la.train()
		self.model_pa.train()
		self.model_mix.train()
		self.optimizer_la.zero_grad()
		self.optimizer_pa.zero_grad()
		self.optimizer_mix.zero_grad()

		utterances_clean_la, utterances_clean_pa, utterances_clean_mix, utterances_attack_la, utterances_attack_pa, utterances_attack_mix, y_clean, y_attack = batch

		utterances_la = torch.cat([utterances_clean_la, utterances_attack_la],0)
		utterances_pa = torch.cat([utterances_clean_pa, utterances_attack_pa],0)
		utterances_mix = torch.cat([utterances_clean_mix, utterances_attack_mix],0)
		y = torch.cat([y_clean, y_attack],0).squeeze()

		ridx = np.random.randint(utterances_la.size(3)//2, utterances_la.size(3))
		utterances_la = utterances_la[:,:,:,:ridx]
		utterances_pa = utterances_pa[:,:,:,:ridx]
		utterances_mix = utterances_mix[:,:,:,:ridx]

		if self.cuda_mode:
			utterances_la, utterances_pa, utterances_mix, y = utterances_la.to(self.device), utterances_pa.to(self.device), utterances_mix.to(self.device), y.to(self.device)

		pred_la = self.model_la.forward(utterances_la).squeeze()
		pred_pa = self.model_pa.forward(utterances_pa).squeeze()
		mixture_coef = torch.sigmoid(self.model_mix.forward(utterances_mix)).squeeze()

		pred = mixture_coef*pred_la + (1.-mixture_coef)*pred_pa

		loss = torch.nn.BCEWithLogitsLoss()(pred, y)

		loss.backward()
		self.optimizer_la.step()
		self.optimizer_pa.step()
		self.optimizer_mix.step()
		return loss.item()

	def valid(self, batch):

		self.model_la.eval()
		self.model_pa.eval()
		self.model_mix.eval()

		with torch.no_grad():

			utterances_clean_la, utterances_clean_pa, utterances_clean_mix, utterances_attack_la, utterances_attack_pa, utterances_attack_mix, y_clean, y_attack = batch

			utterances_la = torch.cat([utterances_clean_la, utterances_attack_la],0)
			utterances_pa = torch.cat([utterances_clean_pa, utterances_attack_pa],0)
			utterances_mix = torch.cat([utterances_clean_mix, utterances_attack_mix],0)
			y = torch.cat([y_clean, y_attack],0).squeeze()

			ridx = np.random.randint(utterances_la.size(3)//2, utterances_la.size(3))
			utterances_la = utterances_la[:,:,:,:ridx]
			utterances_pa = utterances_pa[:,:,:,:ridx]
			utterances_mix = utterances_mix[:,:,:,:ridx]

			if self.cuda_mode:
				utterances_la, utterances_pa, utterances_mix, y = utterances_la.to(self.device), utterances_pa.to(self.device), utterances_mix.to(self.device), y.to(self.device)

			pred_la = self.model_la.forward(utterances_la).squeeze()
			pred_pa = self.model_pa.forward(utterances_pa).squeeze()
			mixture_coef = torch.sigmoid(self.model_mix.forward(utterances_mix)).squeeze()

			pred = mixture_coef*pred_la + (1.-mixture_coef)*pred_pa

		return torch.sigmoid(pred).detach().cpu().numpy().squeeze(), y.cpu().numpy().squeeze()

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_la_state': self.model_la.state_dict(),
		'model_pa_state': self.model_pa.state_dict(),
		'model_mix_state': self.model_mix.state_dict(),
		'optimizer_la_state': self.optimizer_la.state_dict(),
		'optimizer_pa_state': self.optimizer_pa.state_dict(),
		'optimizer_mix_state': self.optimizer_mix.state_dict(),
		'scheduler_la_state': self.scheduler_la.state_dict(),
		'scheduler_pa_state': self.scheduler_pa.state_dict(),
		'scheduler_mix_state': self.scheduler_mix.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)
			# Load models states
			self.model_la.load_state_dict(ckpt['model_la_state'])
			self.model_pa.load_state_dict(ckpt['model_pa_state'])
			self.model_mix.load_state_dict(ckpt['model_mix_state'])
			# Load optimizers states
			self.optimizer_la.load_state_dict(ckpt['optimizer_la_state'])
			self.optimizer_pa.load_state_dict(ckpt['optimizer_pa_state'])
			self.optimizer_mix.load_state_dict(ckpt['optimizer_mix_state'])
			# Load schedulers states
			self.scheduler_la.load_state_dict(ckpt['scheduler_la_state'])
			self.scheduler_pa.load_state_dict(ckpt['scheduler_pa_state'])
			self.scheduler_mix.load_state_dict(ckpt['scheduler_mix_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model_la = self.model_la.cuda()
				self.model_pa = self.model_pa.cuda()
				self.model_mix = self.model_mix.cuda()

		else:
			print('No checkpoint found at: {}'.format(ckpt))
