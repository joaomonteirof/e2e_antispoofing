import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path-to-data-clean', type=str, default='./data_clean/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--path-to-data-spoof', type=str, default='./data_clean/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--path-to-more-data-clean', type=str, default='./more_data_clean/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--path-to-more-data-spoof', type=str, default='./more_data_spoof/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--all-in-one', action='store_true', default=False, help='Dumps all data into single hdf rather than separate depending on clean/spoof')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
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
			clean_ = args.out_path+args.prefix+'train_clean.hdf'
			spoof_ = args.out_path+args.prefix+'train_attack.hdf'
		else:
			clean_ = args.out_path+'train_clean.hdf'
			spoof_ = args.out_path+'train_attack.hdf'

		if os.path.isfile(clean_):
			os.remove(clean_)
			print(clean_+' Removed')

		if os.path.isfile(spoof_):
			os.remove(spoof_)
			print(spoof_+' Removed')

		clean_hdf = h5py.File(clean_, 'a')
		spoof_hdf = h5py.File(spoof_, 'a')

	data_clean = { k:m for k,m in read_mat_scp(args.path_to_data_clean) }
	data_spoof = { k:m for k,m in read_mat_scp(args.path_to_data_spoof) }
	for k,m in read_mat_scp(args.path_to_more_data_clean):
		data_clean[k]=m
	for k,m in read_mat_scp(args.path_to_more_data_spoof):
		data_spoof[k]=m


	if args.all_in_one:
		hdf = all_hdf
	else:
		hdf = clean_hdf

	for i, utt in enumerate(data_clean):

		print('Storing utterance ' + utt)

		data_ = data_clean[utt]
		#data_ = ( data_ - data_.mean(0) ) / data_.std(0)
		features = data_.T

		if features.shape[0]>0:
			features = np.expand_dims(features, 0)
			hdf.create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
		else:
			print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))


	if args.all_in_one:
		hdf = all_hdf
	else:
		hdf = spoof_hdf

	for i, utt in enumerate(data_spoof):

		print('Storing utterance ' + utt)

		data_ = data_spoof[utt]
		#data_ = ( data_ - data_.mean(0) ) / data_.std(0)
		features = data_.T

		if features.shape[0]>0:
			features = np.expand_dims(features, 0)
			hdf.create_dataset(utt, data=features, maxshape=(features.shape[0], features.shape[1], features.shape[2]))
		else:
			print('EMPTY FEATURES ARRAY IN FILE {} !!!!!!!!!'.format(utt))

	hdf.close()
