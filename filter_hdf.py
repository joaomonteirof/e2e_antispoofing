import argparse
import h5py
import numpy as np
import glob
import torch
import os
import pathlib

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Filter recordings out and rewrite hdf')
	parser.add_argument('--path-to-hdf', type=str, default='./data/feats.hdf', metavar='Path', help='Path to hdf file with features')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--out-name', type=str, default='train.hdf', metavar='Path', help='Output hdf file name')
	parser.add_argument('--filt-str', type=str, default='_spf', metavar='Path', help='filter out instances containing this string')
	args = parser.parse_args()

	pathlib.Path(args.out_path).mkdir(parents=True, exist_ok=True)

	source_hdf = h5py.File(args.path_to_hdf, 'r')
	target_hdf = h5py.File(args.out_path+args.out_name, 'a')

	for key in source_hdf:

		if args.filt_str not in key:

			target_hdf.create_dataset(key, data=source_hdf[key])

	hdf.close()