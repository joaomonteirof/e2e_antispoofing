import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os
import random

def augment_spec(example):

	with torch.no_grad():

		if random.random()>0.5:
			example = freq_mask(example, F=50, dim=1)
		if random.random()>0.5:
			example = freq_mask(example, F=50, dim=2)
		if random.random()>0.5:
			example += torch.randn_like(example)*random.choice([1e-1, 1e-2, 1e-3])

	return example

def freq_mask(spec, F=100, num_masks=1, replace_with_zero=False, dim=1):
	"""Frequency masking

	adapted from https://espnet.github.io/espnet/_modules/espnet/utils/spec_augment.html

	:param torch.Tensor spec: input tensor with shape (T, dim)
	:param int F: maximum width of each mask
	:param int num_masks: number of masks
	:param bool replace_with_zero: if True, masked parts will be filled with 0,
		if False, filled with mean
	:param int dim: 1 or 2 indicating to which axis the mask corresponds
	"""

	assert dim==1 or dim==2, 'Only 1 or 2 are valid values for dim!'

	F = min(F, spec.size(dim))

	with torch.no_grad():

		cloned = spec.clone()
		num_bins = cloned.shape[dim]

		for i in range(0, num_masks):
			f = random.randrange(0, F)
			f_zero = random.randrange(0, num_bins - f)

			# avoids randrange error if values are equal and range is empty
			if f_zero == f_zero + f:
				return cloned

			mask_end = random.randrange(f_zero, f_zero + f)
			if replace_with_zero:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = 0.0
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = 0.0
			else:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = cloned.mean()
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = cloned.mean()

	return cloned

class Loader(Dataset):

	def __init__(self, hdf5_clean, hdf5_attack, max_nb_frames, n_cycles=1, augment=False, label_smoothing=0.0):
		super(Loader, self).__init__()
		self.hdf5_1 = hdf5_clean
		self.hdf5_2 = hdf5_attack
		self.n_cycles = n_cycles
		self.max_nb_frames = max_nb_frames

		file_1 = h5py.File(self.hdf5_1, 'r')
		self.idxlist_1 = list(file_1.keys())
		self.len_1 = len(self.idxlist_1)
		file_1.close()

		file_2 = h5py.File(self.hdf5_2, 'r')
		self.idxlist_2 = list(file_2.keys())
		self.len_2 = len(self.idxlist_2)
		file_2.close()

		self.open_file_1 = None
		self.open_file_2 = None

		self.augment = augment

		self.label_smoothing = label_smoothing>0.0
		self.label_dif = label_smoothing

		print('Number of genuine and spoofing recordings: {}, {}'.format(self.len_1, self.len_2))

	def __getitem__(self, index):

		if not self.open_file_1: self.open_file_1 = h5py.File(self.hdf5_1, 'r')
		if not self.open_file_2: self.open_file_2 = h5py.File(self.hdf5_2, 'r')

		index_1 = index % self.len_1
		utt_clean = self.prep_utterance( self.open_file_1[self.idxlist_1[index_1]][0] )

		index_2 = index % self.len_2
		utt_attack = self.prep_utterance( self.open_file_2[self.idxlist_2[index_2]][0] )

		return utt_clean, utt_attack, torch.zeros(1), torch.ones(1)

		if self.label_smoothing:
			return utt_clean, utt_attack, torch.rand(1)*self.label_dif, torch.rand(1)*self.label_dif+(1.-self.label_dif)
		else:
			return utt_clean, utt_attack, torch.zeros(1), torch.ones(1)
	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def prep_utterance(self, data):

		data = np.expand_dims(data, 0)

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		if self.augment:
			data_ = augment_spec(data_)

		return data_

class Loader_all(Dataset):

	def __init__(self, hdf5_la_clean, hdf5_la_attack, hdf5_pa, hdf5_mix, max_nb_frames, label_smoothing=0.0, n_cycles=1):
		super(Loader_all, self).__init__()
		self.hdf5_la_clean = hdf5_la_clean
		self.hdf5_la_attack = hdf5_la_attack
		self.hdf5_pa = hdf5_pa
		self.hdf5_mix = hdf5_mix
		self.n_cycles = n_cycles
		self.max_nb_frames = max_nb_frames

		file_1 = h5py.File(self.hdf5_la_clean, 'r')
		self.idxlist_1 = list(file_1.keys())
		self.len_1 = len(self.idxlist_1)
		file_1.close()

		file_2 = h5py.File(self.hdf5_la_attack, 'r')
		self.idxlist_2 = list(file_2.keys())
		self.len_2 = len(self.idxlist_2)
		file_2.close()

		self.open_file_la_clean = None
		self.open_file_la_attack = None
		self.open_file_pa = None
		self.open_file_mix = None

		self.label_smoothing = label_smoothing>0.0
		self.label_dif = label_smoothing

		print('Number of genuine, spoofing, and total recordings: {}, {}, {}'.format(self.len_1, self.len_2, self.len_1+self.len_2))

	def __getitem__(self, index):

		if not self.open_file_la_clean: self.open_file_la_clean = h5py.File(self.hdf5_la_clean, 'r')
		if not self.open_file_la_attack: self.open_file_la_attack = h5py.File(self.hdf5_la_attack, 'r')
		if not self.open_file_pa: self.open_file_pa = h5py.File(self.hdf5_pa, 'r')
		if not self.open_file_mix: self.open_file_mix = h5py.File(self.hdf5_mix, 'r')

		index_1 = index % self.len_1
		utt_clean = self.idxlist_1[index_1]

		utt_clean_la = self.prep_utterance( self.open_file_la_clean[utt_clean][0] )
		utt_clean_pa = self.prep_utterance( self.open_file_pa[utt_clean][0] )
		utt_clean_mix = self.prep_utterance( self.open_file_mix[utt_clean][0] )

		index_2 = index % self.len_2
		utt_attack = self.idxlist_2[index_2]

		utt_attack_la = self.prep_utterance( self.open_file_la_attack[utt_attack][0] )
		utt_attack_pa = self.prep_utterance( self.open_file_pa[utt_attack][0] )
		utt_attack_mix = self.prep_utterance( self.open_file_mix[utt_attack][0] )

		if self.label_smoothing:
			return utt_clean_la, utt_clean_pa, utt_clean_mix, utt_attack_la, utt_attack_pa, utt_attack_mix, torch.rand(1)*self.label_dif, torch.rand(1)*self.label_dif+(1.-self.label_dif), self.get_label(utt_clean), self.get_label(utt_attack)
		else:
			return utt_clean_la, utt_clean_pa, utt_clean_mix, utt_attack_la, utt_attack_pa, utt_attack_mix, torch.zeros(1), torch.ones(1), self.get_label(utt_clean), self.get_label(utt_attack)

	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def prep_utterance(self, data):

		data = np.expand_dims(data, 0)

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()
		data_ = augment_spec(data_)

		return data_

	def get_label(self, utt):
		prefix = utt.split('-_-')[0]

		assert (prefix=='LA' or prefix=='PA' or prefix=='CLEAN')

		if prefix=='LA':
			if self.label_smoothing:
				return torch.rand(1)*self.label_dif+(1.-self.label_dif)
			else:
				return torch.ones(1)
		elif prefix=='PA':
			if self.label_smoothing:
				return torch.rand(1)*self.label_dif
			else:
				return torch.zeros(1)
		elif prefix=='CLEAN':
			if self.label_smoothing:
				return 0.5*torch.ones(1) + torch.rand(1)*self.label_dif-self.label_dif*0.5
			else:
				return 0.5*torch.ones(1)

class Loader_all_valid(Dataset):

	def __init__(self, hdf5_la_clean, hdf5_la_attack, hdf5_pa, hdf5_mix, max_nb_frames, n_cycles=1):
		super(Loader_all_valid, self).__init__()
		self.hdf5_la_clean = hdf5_la_clean
		self.hdf5_la_attack = hdf5_la_attack
		self.hdf5_pa = hdf5_pa
		self.hdf5_mix = hdf5_mix
		self.n_cycles = n_cycles
		self.max_nb_frames = max_nb_frames

		file_1 = h5py.File(self.hdf5_la_clean, 'r')
		self.idxlist_1 = list(file_1.keys())
		self.len_1 = len(self.idxlist_1)
		file_1.close()

		file_2 = h5py.File(self.hdf5_la_attack, 'r')
		self.idxlist_2 = list(file_2.keys())
		self.len_2 = len(self.idxlist_2)
		file_2.close()

		self.open_file_la_clean = None
		self.open_file_la_attack = None
		self.open_file_pa = None
		self.open_file_mix = None

		print('Number of genuine, spoofing, and total recordings: {}, {}, {}'.format(self.len_1, self.len_2, self.len_1+self.len_2))

	def __getitem__(self, index):

		if not self.open_file_la_clean: self.open_file_la_clean = h5py.File(self.hdf5_la_clean, 'r')
		if not self.open_file_la_attack: self.open_file_la_attack = h5py.File(self.hdf5_la_attack, 'r')
		if not self.open_file_pa: self.open_file_pa = h5py.File(self.hdf5_pa, 'r')
		if not self.open_file_mix: self.open_file_mix = h5py.File(self.hdf5_mix, 'r')

		index_1 = index % self.len_1
		utt_clean = self.idxlist_1[index_1]

		utt_clean_la = self.prep_utterance( self.open_file_la_clean[utt_clean][0] )
		utt_clean_pa = self.prep_utterance( self.open_file_pa[utt_clean][0] )
		utt_clean_mix = self.prep_utterance( self.open_file_mix[utt_clean][0] )

		index_2 = index % self.len_2
		utt_attack = self.idxlist_2[index_2]

		utt_attack_la = self.prep_utterance( self.open_file_la_attack[utt_attack][0] )
		utt_attack_pa = self.prep_utterance( self.open_file_pa[utt_attack][0] )
		utt_attack_mix = self.prep_utterance( self.open_file_mix[utt_attack][0] )

		return utt_clean_la, utt_clean_pa, utt_clean_mix, utt_attack_la, utt_attack_pa, utt_attack_mix, torch.zeros(1), torch.ones(1)

	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def prep_utterance(self, data):

		data = np.expand_dims(data, 0)

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		return data_

class Loader_mcc(Dataset):

	def __init__(self, hdf5_clean, hdf5_attack, max_nb_frames, file_lists_path, n_cycles=1):
		super(Loader_mcc, self).__init__()

		self.labels_dict = {'AA':1, 'AB':2, 'AC':3, 'BA':4, 'BB':5, 'BC':6, 'CA':7, 'CB':8, 'CC':9}

		self.utt2att = self.read_files_lists(file_lists_path)

		self.hdf5_1 = hdf5_clean
		self.hdf5_2 = hdf5_attack
		self.n_cycles = n_cycles
		self.max_nb_frames = max_nb_frames

		file_1 = h5py.File(self.hdf5_1, 'r')
		self.idxlist_1 = list(file_1.keys())
		self.len_1 = len(self.idxlist_1)
		file_1.close()

		file_2 = h5py.File(self.hdf5_2, 'r')
		self.idxlist_2 = list(file_2.keys())
		self.len_2 = len(self.idxlist_2)
		file_2.close()

		self.open_file_1 = None
		self.open_file_2 = None

		print('Number of genuine and spoofing recordings: {}, {}'.format(self.len_1, self.len_2))

	def __getitem__(self, index):

		if not self.open_file_1: self.open_file_1 = h5py.File(self.hdf5_1, 'r')
		if not self.open_file_2: self.open_file_2 = h5py.File(self.hdf5_2, 'r')

		index_1 = index % self.len_1
		utt_clean = self.prep_utterance( self.open_file_1[self.idxlist_1[index_1]][0] )

		index_2 = index % self.len_2
		utt_attack = self.prep_utterance( self.open_file_2[self.idxlist_2[index_2]][0] )

		if np.random.rand() > 0.5:
			return utt_clean, utt_attack, torch.zeros(1).long(), (torch.ones(1)*self.labels_dict[self.utt2att[self.idxlist_2[index_2]]]).long()
		else:
			return utt_attack, utt_clean, (torch.ones(1)*self.labels_dict[self.utt2att[self.idxlist_2[index_2]]]).long(), torch.zeros(1).long()

	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def prep_utterance(self, data):

		data = np.expand_dims(data, 0)

		if data.shape[-1]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		data_ = torch.from_numpy(data_).float().contiguous()

		return data_

	def read_files_lists(self, files_path):

		files_list = glob.glob(files_path + '*.lst')

		utt2att = {}

		for file_ in files_list:

			attack_type = file_.split('_')[-1].split('.')[0]
			utts_list = self.read_utts(file_)

			for utt in utts_list:
				utt2att[utt] = attack_type

		return utt2att
			
	def read_utts(self, file_):
		with open(file_, 'r') as file:
			utt_attacks = file.readlines()

		utt_list = []

		for line in utt_attacks:
			utt = line.split('/')[-1].split('.')[0]
			utt_list.append(utt)

		return utt_list
