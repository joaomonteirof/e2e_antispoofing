import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

def read_list(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

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
	parser.add_argument('--path-to-more-data', type=str, default=None, metavar='Path', help='Path to second feats.scp')
	parser.add_argument('--path-to-list', type=str, default='./data/list.txt', metavar='Path', help='Path to list of recs and labels')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train', metavar='Path', help='Output hdf file name')
	args = parser.parse_args()

	file_clean = args.out_path+args.out_name+'_clean.hdf'
	file_attack = args.out_path+args.out_name+'_attack.hdf'

	if os.path.isfile(file_clean):
		os.remove(file_clean)
		print(file_clean+' Removed')

	if os.path.isfile(file_attack):
		os.remove(file_attack)
		print(file_attack+' Removed')

	print('Start of data preparation')

	hdf_clean = h5py.File(file_clean, 'a')
	hdf_attack = h5py.File(file_attack, 'a')

	print('Processing file {}'.format(args.path_to_data))

	data = { k:m for k,m in read_mat_scp(args.path_to_data) }

	if args.path_to_more_data is not None:
		print('Processing file {}'.format(args.path_to_more_data))
		for k,m in read_mat_scp(args.path_to_more_data):
			data[k]=m

	utt_list, attack_type_list, label_list = read_list(args.path_to_list)

	for i, utt in enumerate(utt_list):

		print('Storing utterance ' + utt + ':' + label_list[i])

		if label_list[i] == 'spoof':
			hdf = hdf_attack
		elif label_list[i] == 'bonafide':
			hdf = hdf_clean
		else:
			print('Label Error!!')

		data_ = data[utt]
		features = data_.T

		print(features.shape)

		if features.shape[0]>0:
			features = np.expand_dims(features, 0)
			hdf.create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
		else:
			print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))

	hdf_clean.close()
	hdf_attack.close()
