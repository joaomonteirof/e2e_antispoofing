import argparse
import h5py
import numpy as np
import glob
import torch
import os
from kaldi_io import read_mat_scp

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Train data preparation and storage in .hdf')
	parser.add_argument('--path1', type=str, default='./data/feats.scp', metavar='Path', help='Path to feats.scp')
	parser.add_argument('--path2', type=str, default=None, metavar='Path', help='Path to second feats.scp for comparison')
	parser.add_argument('--path3', type=str, default=None, metavar='Path', help='Path to third feats.scp for comparison')
	parser.add_argument('--path-to-hdfs', type=str, default=None, metavar='Path', help='Path to hdf files to be compared with data in path1')
	args = parser.parse_args()

	assert args.path2 is not None or args.path_to_hdfs is not None, 'Either --path2 or --path-to-hdf have to passed along with --path1'

	print('Loading data from path1: {}'.format(args.path1))
	data = { k:m for k,m in read_mat_scp(args.path1) }

	if args.path2 is not None:
		print('Loading data from path2: {}'.format(args.path2))
		data2 = { k:m for k,m in read_mat_scp(args.path2) }

		collision_count = 0

		for k in data:
			if k in data2:
				print('Collision: File {}'.format(k))
				collision_count += 1

		print('Total collisions between data in path1 and path2: {}'.format(collision_count))

	if args.path3 is not None:
		print('Loading data from path3: {}'.format(args.path3))
		data2 = { k:m for k,m in read_mat_scp(args.path3) }

		collision_count = 0

		for k in data:
			if k in data2:
				print('Collision: File {}'.format(k))
				collision_count += 1

		print('Total collisions between data in path1 and path3: {}'.format(collision_count))

	if args.path_to_hdfs is not None:

		hdf_list = glob.glob(args.path_to_hdfs + '*.hdf')

		for file_ in hdf_list:

			print('Loading data from: {}'.format(file_))

			data2 = list(h5py.File(file_, 'r').keys())

			collision_count = 0

			for k in data:
				if k in data2:
					print('Collision: File {}'.format(k))
					collision_count += 1

			print('Total collisions between data in path1 and file {}: {}'.format(file_.split('/')[-1], collision_count))
