import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--path-to-more-data', type=str, default=None, metavar='Path', help='Path to second feats.scp')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--n-val-speakers', type=int, default=100, help='Number of speakers for valid data')
	args = parser.parse_args()

	if os.path.isfile(args.out_path+'train_'+args.out_name):
		os.remove(args.out_path+'train_'+args.out_name)
		print(args.out_path+'train_'+args.out_name+' Removed')

	if os.path.isfile(args.out_path+'valid_'+args.out_name):
		os.remove(args.out_path+'valid_'+args.out_name)
		print(args.out_path+'valid_'+args.out_name+' Removed')

	print('Start of data preparation')

	train_hdf = h5py.File(args.out_path+'train_'+args.out_name, 'a')
	valid_hdf = h5py.File(args.out_path+'valid_'+args.out_name, 'a')

	print('Processing file {}'.format(args.path_to_data))

	data = { k:m for k,m in read_mat_scp(args.path_to_data) }

	if args.path_to_more_data is not None:
		print('Processing file {}'.format(args.path_to_more_data))
		for k,m in read_mat_scp(args.path_to_more_data):
			data[k]=m

	if len(data)<args.n_val_speakers:
		print('Too many validation speakers. Total number of speakers is:'.format(len(data)))
		exit(1)

	val_idxs = np.random.choice(np.arange(len(data)), replace=False, size=args.n_val_speakers)

	for i, utt in enumerate(data):

		if i in val_idxs:
			hdf = valid_hdf
		else:
			hdf = train_hdf

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

	hdf.close()
