import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm

from utils import compute_eer

class TrainLoop(object):

	def __init__(self, model_la, model_pa, model_mix, optimizer_la, optimizer_pa, optimizer_mix, train_loader, valid_loader, patience, train_mode, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
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
		self.device = next(self.model_la.parameters()).device
		self.train_mode = train_mode

		self.history = {'train_loss': [], 'train_loss_batch': [], 'train_all': [], 'train_all_batch': [], 'train_la': [], 'train_la_batch': [], 'train_pa': [], 'train_pa_batch': [], 'train_mix': [], 'train_mix_batch': [], 'valid_la': [], 'valid_pa': [], 'valid_mix': []}

		if self.train_mode=='mix':
			self.history['valid_all']=[]

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1, save_every=1):

		while (self.cur_epoch < n_epochs):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader))
			train_loss_epoch=0.0
			train_all_epoch=0.0
			train_la_epoch=0.0
			train_pa_epoch=0.0
			train_mix_epoch=0.0

			for t, batch in train_iter:
				train_loss, train_all, train_la, train_pa, train_mix = self.train_step(batch)
				self.history['train_loss_batch'].append(train_loss)
				self.history['train_all_batch'].append(train_all)
				self.history['train_la_batch'].append(train_la)
				self.history['train_pa_batch'].append(train_pa)
				self.history['train_mix_batch'].append(train_mix)
				train_loss_epoch+=train_loss
				train_all_epoch+=train_all
				train_la_epoch+=train_la
				train_pa_epoch+=train_pa
				train_mix_epoch+=train_mix
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))
			self.history['train_all'].append(train_all_epoch/(t+1))
			self.history['train_la'].append(train_la_epoch/(t+1))
			self.history['train_pa'].append(train_pa_epoch/(t+1))
			self.history['train_mix'].append(train_mix_epoch/(t+1))

			print('Total train loss, loss_all,  loss_la,  loss_pa,  loss_mix: {:0.4f}'.format(self.history['train_loss'][-1]))

			scores_all, scores_la, scores_pa, scores_mix, labels = None, None

			if self.train_mode=='mix':

				for t, batch in enumerate(self.valid_loader):

						scores_all_batch, scores_la_batch, scores_pa_batch, scores_mix_batch, labels_batch = self.valid(batch)
					try:
						scores_all = np.concatenate([scores_all, scores_all_batch], 0)
						scores_la = np.concatenate([scores_la, scores_la_batch], 0)
						scores_pa = np.concatenate([scores_pa, scores_pa_batch], 0)
						scores_mix = np.concatenate([scores_mix, scores_mix_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						scores_all, scores_la, scores_pa, scores_mix, labels = scores_batch_all, scores_batch_la, scores_batch_pa, scores_batch_mix, labels_batch

				self.history['valid_all'].append(compute_eer(labels, scores_all))
				self.history['valid_la'].append(compute_eer(labels, scores_la))
				self.history['valid_pa'].append(compute_eer(labels, scores_pa))
				self.history['valid_mix'].append(compute_eer(labels, scores_mix))

				print('ALL: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_all'][-1], np.min(self.history['valid_all']), 1+np.argmin(self.history['valid_all'])))

			else:

				for t, batch in enumerate(self.valid_loader):

						scores_la_batch, scores_pa_batch, scores_mix_batch, labels_batch = self.valid(batch)
					try:
						scores_la = np.concatenate([scores_la, scores_la_batch], 0)
						scores_pa = np.concatenate([scores_pa, scores_pa_batch], 0)
						scores_mix = np.concatenate([scores_mix, scores_mix_batch], 0)
						labels = np.concatenate([labels, labels_batch], 0)
					except:
						scores_la, scores_pa, scores_mix, labels = scores_batch_all, scores_batch_la, scores_batch_pa, scores_batch_mix, labels_batch

				self.history['valid_la'].append(compute_eer(labels, scores_la))
				self.history['valid_pa'].append(compute_eer(labels, scores_pa))
				self.history['valid_mix'].append(compute_eer(labels, scores_mix))

			print('LA: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_la'][-1], np.min(self.history['valid_la']), 1+np.argmin(self.history['valid_la'])))
			print('PA: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_pa'][-1], np.min(self.history['valid_pa']), 1+np.argmin(self.history['valid_pa'])))
			print('MIX: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_mix'][-1], np.min(self.history['valid_mix']), 1+np.argmin(self.history['valid_mix'])))

			self.scheduler_la.step(self.history['valid_la'][-1])
			self.scheduler_pa.step(self.history['valid_pa'][-1])
			self.scheduler_mix.step(self.history['valid_mix'][-1])

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

		utterances_clean_la, utterances_clean_pa, utterances_clean_mix, utterances_attack_la, utterances_attack_pa, utterances_attack_mix, y_clean, y_attack, y_lapa_clean, y_lapa_attack = batch

		utterances_la = torch.cat([utterances_clean_la, utterances_attack_la],0)
		utterances_pa = torch.cat([utterances_clean_pa, utterances_attack_pa],0)
		utterances_mix = torch.cat([utterances_clean_mix, utterances_attack_mix],0)
		y = torch.cat([y_clean, y_attack],0).squeeze()
		y_lapa = torch.cat([y_lapa_clean, y_lapa_attack],0).squeeze()

		ridx = np.random.randint(utterances_la.size(3)//2, utterances_la.size(3))
		utterances_la = utterances_la[:,:,:,:ridx]
		utterances_pa = utterances_pa[:,:,:,:ridx]
		utterances_mix = utterances_mix[:,:,:,:ridx]

		if self.cuda_mode:
			utterances_la, utterances_pa, utterances_mix, y, y_lapa = utterances_la.to(self.device), utterances_pa.to(self.device), utterances_mix.to(self.device), y.to(self.device), y_lapa.to(self.device)

		pred_la = self.model_la.forward(utterances_la).squeeze()
		pred_pa = self.model_pa.forward(utterances_pa).squeeze()
		pred_mix = self.model_mix.forward(utterances_mix).squeeze()

		if self.train_mode == 'mix':
			mixture_coef = torch.sigmoid(pred_mix)
			pred = mixture_coef*pred_la + (1.-mixture_coef)*pred_pa
			loss_full_mix = torch.nn.BCEWithLogitsLoss()(pred, y)
			loss_la = torch.nn.BCEWithLogitsLoss()(pred_la, y)
			loss_pa = torch.nn.BCEWithLogitsLoss()(pred_pa, y)
			loss_mix = torch.nn.BCELoss()(mixture_coef, y_lapa)

		elif self.train_mode == 'lapa':
			loss_full_mix = torch.zeros(1)
			loss_la = torch.nn.BCELoss()(pred_la, y_lapa)
			loss_pa = torch.nn.BCELoss()(pred_pa, y_lapa)
			loss_mix = torch.nn.BCELoss()(pred_mix, y_lapa)

		elif self.train_mode == 'independent':
			loss_full_mix = torch.zeros(1)
			loss_la = torch.nn.BCEWithLogitsLoss()(pred_la, y)
			loss_pa = torch.nn.BCEWithLogitsLoss()(pred_pa, y)
			loss_mix = torch.nn.BCEWithLogitsLoss()(pred_mix, y)

		else:
			raise NotImplementedError

		loss = loss_full_mix + loss_la + loss_pa + loss_mix

		loss.backward()
		self.optimizer_la.step()
		self.optimizer_pa.step()
		self.optimizer_mix.step()
		return loss.item(), loss_full_mix.item(), loss_la.item(), loss_pa.item(), loss_mix.item()

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
			pred_mix = self.model_mix.forward(utterances_mix).squeeze()

			if self.train_mode == 'mix':

				mixture_coef = torch.sigmoid(pred_mix)
				pred = mixture_coef*pred_la + (1.-mixture_coef)*pred_pa
				score_all = 1.-torch.sigmoid(mixture_coef*pred_la + (1.-mixture_coef)*pred_pa).cpu().numpy().squeeze()
				score_la = 1.-torch.sigmoid(pred_la).cpu().numpy().squeeze()
				score_pa = 1.-torch.sigmoid(pred_pa).cpu().numpy().squeeze()
				score_mix = 1.-2*abs(mixture_coef-0.5).cpu().numpy().squeeze()
				return score_all, score_la, score_pa, score_mix, y.cpu().numpy().squeeze()

			elif self.train_mode == 'lapa':
				score_la = 1.-2*abs(torch.sigmoid(pred_la)-0.5).cpu().numpy().squeeze()
				score_pa = 1.-2*abs(torch.sigmoid(pred_pa)-0.5).cpu().numpy().squeeze()
				score_mix = 1.-2*abs(torch.sigmoid(pred_mix)-0.5).cpu().numpy().squeeze()
				return score_la, score_pa, score_mix, y.cpu().numpy().squeeze()

			elif self.train_mode == 'independent':
				score_la = 1.-torch.sigmoid(pred_la).cpu().numpy().squeeze()
				score_pa = 1.-torch.sigmoid(pred_pa).cpu().numpy().squeeze()
				score_mix = 1.-torch.sigmoid(pred_mix).cpu().numpy().squeeze()
				return score_la, score_pa, score_mix, y.cpu().numpy().squeeze()

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
