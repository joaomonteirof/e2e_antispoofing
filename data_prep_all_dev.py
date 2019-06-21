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
	parser.add_argument('--path-to-more-data', type=str, default=None, metavar='Path', help='Path to feats.scp')
	parser.add_argument('--all-in-one', action='store_true', default=False, help='Dumps all data into single hdf rather than separate depending on clean/spoof')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--more-trials-path', type=str, default='./data/more_trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--prefix', type=str, default=None)
	args = parser.parse_args()

	if args.all_in_one:

		if args.prefix:
			all_ = args.out_path+args.prefix+'all.hdf'
		else:
			all_ = args.out_path+'all.hdf'

		if os.path.isfile(all_):
			os.remove(all_)
			print(all_+' Removed')

		all_hdf = h5py.File(all_, 'a')

	else:

		if args.prefix:
			clean_ = args.out_path+args.prefix+'valid_clean.hdf'
			spoof_ = args.out_path+args.prefix+'valid_attack.hdf'
		else:
			clean_ = args.out_path+'valid_clean.hdf'
			spoof_ = args.out_path+'valid_attack.hdf'

		if os.path.isfile(clean_):
			os.remove(clean_)
			print(clean_+' Removed')

		if os.path.isfile(spoof_):
			os.remove(spoof_)
			print(spoof_+' Removed')

		clean_hdf = h5py.File(clean_, 'a')
		spoof_hdf = h5py.File(spoof_, 'a')


	data = { k:m for k,m in read_mat_scp(args.path_to_data) }
	test_utts, attack_type_list, label_list = read_trials(args.trials_path)

	if args.path_to_more_data:
		for k,m in read_mat_scp(args.path_to_more_data):
			data[k]=m

		more_test_utts, more_attack_type_list, more_label_list = read_trials(args.more_trials_path)
		test_utts.extend(more_test_utts)
		label_list.extend(more_label_list)

	for i, utt in enumerate(test_utts):

		print('Storing utterance ' + utt)

		if args.all_in_one:
			hdf = all_hdf
		else:
			if label_list[i] == 'bonafide':
				hdf = clean_hdf
			else:
				hdf = spoof_hdf

		data_ = data[utt]
		#data_ = ( data_ - data_.mean(0) ) / data_.std(0)
		features = data_.T

		if features.shape[0]>0:
			features = np.expand_dims(features, 0)
			hdf.create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
		else:
			print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))
