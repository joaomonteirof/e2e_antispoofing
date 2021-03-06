import torch
import torch.nn.functional as F

import numpy as np

import os
from tqdm import tqdm

from utils import compute_eer

class TrainLoop(object):

	def __init__(self, model_la, model_pa, model_mix, optimizer_la, optimizer_pa, optimizer_mix, train_loader, valid_loader, train_mode, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logger=None):
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
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.total_iters = 0
		self.cur_epoch = 0
		self.device = next(self.model_la.parameters()).device
		self.train_mode = train_mode
		self.logger = logger

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
				if self.logger:
					self.logger.add_scalar('Train/Total train loss', train_loss, self.total_iters)
					self.logger.add_scalar('Train/Train Loss mixture', train_all, self.total_iters)
					self.logger.add_scalar('Train/Train Loss LA', train_la, self.total_iters)
					self.logger.add_scalar('Train/Train Loss PA', train_pa, self.total_iters)
					self.logger.add_scalar('Train/Train Loss MIX', train_mix, self.total_iters)
					self.logger.add_scalar('Info/LR_LA', self.optimizer_la.optimizer.param_groups[0]['lr'], self.total_iters)
					self.logger.add_scalar('Info/LR_PA', self.optimizer_pa.optimizer.param_groups[0]['lr'], self.total_iters)
					self.logger.add_scalar('Info/LR_MIX', self.optimizer_mix.optimizer.param_groups[0]['lr'], self.total_iters)
				self.total_iters += 1

			self.history['train_loss'].append(train_loss_epoch/(t+1))
			self.history['train_all'].append(train_all_epoch/(t+1))
			self.history['train_la'].append(train_la_epoch/(t+1))
			self.history['train_pa'].append(train_pa_epoch/(t+1))
			self.history['train_mix'].append(train_mix_epoch/(t+1))

			print('Total train loss, loss_all,  loss_la,  loss_pa,  loss_mix: {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}'.format(self.history['train_loss'][-1], self.history['train_all'][-1], self.history['train_la'][-1], self.history['train_pa'][-1], self.history['train_mix'][-1]))

			scores_all, scores_la, scores_pa, scores_mix, labels = None, None, None, None, None

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
						scores_all, scores_la, scores_pa, scores_mix, labels = scores_all_batch, scores_la_batch, scores_pa_batch, scores_mix_batch, labels_batch

				self.history['valid_all'].append(compute_eer(labels, scores_all))
				self.history['valid_la'].append(compute_eer(labels, scores_la))
				self.history['valid_pa'].append(compute_eer(labels, scores_pa))
				self.history['valid_mix'].append(compute_eer(labels, scores_mix))

				if self.logger:
					self.logger.add_scalar('Valid/Valid EER mixture', self.history['valid_all'][-1], self.total_iters-1)
					self.logger.add_scalar('Valid/Best valid EER mixture', np.min(self.history['valid_all']), self.total_iters-1)
					self.logger.add_pr_curve('Valid. ROC mixture', labels=labels, predictions=scores_all, global_step=self.total_iters-1)
					self.logger.add_histogram('Valid/Scores_Mixture', values=scores_all, global_step=self.total_iters-1)

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
						scores_la, scores_pa, scores_mix, labels = scores_la_batch, scores_pa_batch, scores_mix_batch, labels_batch

				self.history['valid_la'].append(compute_eer(labels, scores_la))
				self.history['valid_pa'].append(compute_eer(labels, scores_pa))
				self.history['valid_mix'].append(compute_eer(labels, scores_mix))

			if self.logger:
				self.logger.add_scalar('Valid/Valid EER LA', self.history['valid_la'][-1], self.total_iters-1)
				self.logger.add_scalar('Valid/Best valid EER LA', np.min(self.history['valid_la']), self.total_iters-1)
				self.logger.add_pr_curve('Valid. ROC LA', labels=labels, predictions=scores_la, global_step=self.total_iters-1)
				self.logger.add_histogram('Valid/Scores_LA', values=scores_la, global_step=self.total_iters-1)
				self.logger.add_scalar('Valid/Valid EER PA', self.history['valid_pa'][-1], self.total_iters-1)
				self.logger.add_scalar('Valid/Best valid EER PA', np.min(self.history['valid_pa']), self.total_iters-1)
				self.logger.add_pr_curve('Valid. ROC PA', labels=labels, predictions=scores_pa, global_step=self.total_iters-1)
				self.logger.add_histogram('Valid/Scores_PA', values=scores_pa, global_step=self.total_iters-1)
				self.logger.add_scalar('Valid/Valid EER MIX', self.history['valid_mix'][-1], self.total_iters-1)
				self.logger.add_scalar('Valid/Best valid EER MIX', np.min(self.history['valid_mix']), self.total_iters-1)
				self.logger.add_pr_curve('Valid. ROC MIX', labels=labels, predictions=scores_mix, global_step=self.total_iters-1)
				self.logger.add_histogram('Valid/Scores_MIX', values=scores_mix, global_step=self.total_iters-1)
				self.logger.add_histogram('Valid/Labels', values=labels, global_step=self.total_iters-1)




			print('LA: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_la'][-1], np.min(self.history['valid_la']), 1+np.argmin(self.history['valid_la'])))
			print('PA: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_pa'][-1], np.min(self.history['valid_pa']), 1+np.argmin(self.history['valid_pa'])))
			print('MIX: Current validation loss, best validation loss, and epoch: {:0.4f}, {:0.4f}, {}'.format(self.history['valid_mix'][-1], np.min(self.history['valid_mix']), 1+np.argmin(self.history['valid_mix'])))

			print('Current LRs (LA, PA, Mixture): {}, {}, {}'.format(self.optimizer_la.optimizer.param_groups[0]['lr'], self.optimizer_pa.optimizer.param_groups[0]['lr'], self.optimizer_mix.optimizer.param_groups[0]['lr']))

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0 or self.history['valid_la'][-1] < np.min([np.inf]+self.history['valid_la'][:-1]) or self.history['valid_pa'][-1] < np.min([np.inf]+self.history['valid_pa'][:-1]) or self.history['valid_mix'][-1] < np.min([np.inf]+self.history['valid_mix'][:-1]):
				self.checkpointing()

		print('Training done!')

		if self.valid_loader is not None:
			print('\nLA: Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_la']), 1+np.argmin(self.history['valid_la'])))
			print('PA: Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_pa']), 1+np.argmin(self.history['valid_pa'])))
			print('MIX: Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_mix']), 1+np.argmin(self.history['valid_mix'])))
			print('ALL: Best validation loss and corresponding epoch: {:0.4f}, {}'.format(np.min(self.history['valid_all']), 1+np.argmin(self.history['valid_all'])))

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
			loss_full_mix = torch.zeros(1).to(self.device)
			loss_la = torch.nn.BCEWithLogitsLoss()(pred_la, y_lapa)
			loss_pa = torch.nn.BCEWithLogitsLoss()(pred_pa, y_lapa)
			loss_mix = torch.nn.BCEWithLogitsLoss()(pred_mix, y_lapa)

		elif self.train_mode == 'independent':
			loss_full_mix = torch.zeros(1).to(self.device)
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
				score_all = torch.sigmoid(mixture_coef*pred_la + (1.-mixture_coef)*pred_pa).cpu().numpy().squeeze()
				score_la = torch.sigmoid(pred_la).cpu().numpy().squeeze()
				score_pa = torch.sigmoid(pred_pa).cpu().numpy().squeeze()
				score_mix = 2*abs(mixture_coef-0.5).cpu().numpy().squeeze()
				return score_all, score_la, score_pa, score_mix, y.cpu().numpy().squeeze()

			elif self.train_mode == 'lapa':
				score_la = 2*abs(torch.sigmoid(pred_la)-0.5).cpu().numpy().squeeze()
				score_pa = 2*abs(torch.sigmoid(pred_pa)-0.5).cpu().numpy().squeeze()
				score_mix = 2*abs(torch.sigmoid(pred_mix)-0.5).cpu().numpy().squeeze()
				return score_la, score_pa, score_mix, y.cpu().numpy().squeeze()

			elif self.train_mode == 'independent':
				score_la = torch.sigmoid(pred_la).cpu().numpy().squeeze()
				score_pa = torch.sigmoid(pred_pa).cpu().numpy().squeeze()
				score_mix = torch.sigmoid(pred_mix).cpu().numpy().squeeze()
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
			self.optimizer_la.step_num = ckpt['total_iters']
			self.optimizer_pa.step_num = ckpt['total_iters']
			self.optimizer_mix.step_num = ckpt['total_iters']

			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			if self.cuda_mode:
				self.model_la = self.model_la.to(self.device)
				self.model_pa = self.model_pa.to(self.device)
				self.model_mix = self.model_mix.to(self.device)

		else:
			print('No checkpoint found at: {}'.format(ckpt))
