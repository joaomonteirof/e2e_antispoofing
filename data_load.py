import h5py
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os

class Loader(Dataset):

	def __init__(self, hdf5_clean, hdf5_attack, max_nb_frames, n_cycles=1):
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

		print('Number of genuine and spoofing recordings: {}, {}'.format(self.len_1, self.len_2))

	def __getitem__(self, index):

		if not self.open_file_1: self.open_file_1 = h5py.File(self.hdf5_1, 'r')
		if not self.open_file_2: self.open_file_2 = h5py.File(self.hdf5_2, 'r')

		index_1 = index % self.len_1
		utt_clean = self.prep_utterance( self.open_file_1[self.idxlist_1[index_1]][0] )

		index_2 = index % self.len_2
		utt_attack = self.prep_utterance( self.open_file_2[self.idxlist_2[index_2]][0] )

		return utt_clean, utt_attack, torch.zeros(1), torch.ones(1)

	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def prep_utterance(self, data):

		data = np.expand_dims(data, 0)

		if data.shape[2]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[2]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[0]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

		return data_

class Loader_all(Dataset):

	def __init__(self, hdf5_la_clean, hdf5_la_attack, hdf5_pa, hdf5_mix, max_nb_frames, n_cycles=1):
		super(Loader_all, self).__init__()
		self.hdf5_la_clean = hdf5_la_clean
		self.hdf5_la_attack = hdf5_la_attack
		self.hdf5_pa = hdf5_pa
		self.hdf5_mix = hdf5_mix
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

		self.open_file_la_clean = None
		self.open_file_la_attack = None
		self.open_file_pa = None
		self.open_file_mix = None

		print('Number of genuine and spoofing recordings: {}, {}'.format(self.len_1, self.len_2))

	def __getitem__(self, index):

		if not self.open_file_la_clean: self.open_file_la_clean = h5py.File(self.hdf5_la_clean, 'r')
		if not self.open_file_la_attack: self.open_file_la_attack = h5py.File(self.hdf5_la_attack, 'r')
		if not self.open_file_pa: self.open_file_pa = h5py.File(self.hdf5_pa, 'r')
		if not self.open_file_mix: self.open_file_mix = h5py.File(self.hdf5_mix, 'r')

		index_1 = index % self.len_1
		utt_clean_la = self.prep_utterance( self.open_file_la_clean[self.idxlist_1[index_1]][0] )
		utt_clean_pa = self.prep_utterance( self.open_file_pa[self.idxlist_1[index_1]][0] )
		utt_clean_mix = self.prep_utterance( self.open_file_mix[self.idxlist_1[index_1]][0] )

		index_2 = index % self.len_2
		utt_attack = self.prep_utterance( self.open_file_la_attack[self.idxlist_2[index_2]][0] )
		utt_attack_pa = self.prep_utterance( self.open_file_pa[self.idxlist_2[index_2]][0] )
		utt_attack_mix = self.prep_utterance( self.open_file_mix[self.idxlist_2[index_2]][0] )

		return utt_clean_la, utt_clean_pa, utt_clean_mix, utt_attack_la, utt_attack_pa, utt_attack_mix, torch.zeros(1), torch.ones(1)

	def __len__(self):
		return self.n_cycles*np.maximum(self.len_1, self.len_2)

	def prep_utterance(self, data):

		data = np.expand_dims(data, 0)

		if data.shape[2]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[2]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[0]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

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

		if data.shape[2]>self.max_nb_frames:
			ridx = np.random.randint(0, data.shape[2]-self.max_nb_frames)
			data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
		else:
			mul = int(np.ceil(self.max_nb_frames/data.shape[0]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :self.max_nb_frames]

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
