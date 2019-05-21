import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

def read_trials(path, eval_=False):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	if eval_:

		utt_list = []

		for line in utt_labels:
			utt = line.strip('\n')
			utt_list.append(utt)

		return utt_list

	else:

		utt_list, attack_type_list, label_list = [], [], []

		for line in utt_labels:
			_, utt, _, attack_type, label = line.split(' ')
			utt_list.append(utt)
			attack_type_list.append(attack_type)
			label_list.append(label.strip('\n'))

		return utt_list, attack_type_list, label_list

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--path-to-hdf', type=str, default=None, metavar='Path', help='Path to second feats.scp')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--data-type', choices=['spoof', 'bonafide'], default='spoof', help='Data type: spoof or bonafide')
	args = parser.parse_args()

	print('Start of data preparation')

	hdf = h5py.File(args.path_to_hdf, 'a')

	print('Processing file {}'.format(args.path_to_data))

	data = { k:m for k,m in read_mat_scp(args.path_to_data) }

	test_utts, attack_type_list, label_list = read_trials(args.trials_path)

	for i, utt in enumerate(test_utts):

		if label_list[i] == args.data_type:

			print('Storing utterance ' + utt)

			data_ = data[utt]
			#data_ = ( data_ - data_.mean(0) ) / data_.std(0)
			features = data_.T

			print(features.shape)

			if features.shape[0]>0:
				features = np.expand_dims(features, 0)
				hdf.create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
			else:
				print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))

		else:
			print('Skipping utterance ' + utt)
	hdf.close()
