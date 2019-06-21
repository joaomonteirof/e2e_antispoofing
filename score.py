import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_mat_scp
import model as model_
import scipy.io as sio

from utils import compute_eer_labels, get_freer_gpu, read_trials, change_keys

def prep_feats(data_):

	#data_ = ( data_ - data_.mean(0) ) / data_.std(0)

	features = data_.T

	if features.shape[1]<50:
		mul = int(np.ceil(50/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :50]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute scores')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_34_CC'], default='lcnn_9', help='Model arch')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	parser.add_argument('--no-eer', action='store_true', default=False, help='Disables computation of EER')
	parser.add_argument('--eval', action='store_true', default=False, help='Enables eval trials reading')
	parser.add_argument('--tandem', action='store_true', default=False, help='Scoring with tandem features')
	parser.add_argument('--ncoef', type=int, default=90, metavar='N', help='Number of cepstral coefs (default: 90)')
	parser.add_argument('--init-coef', type=int, default=0, metavar='N', help='First cepstral coefs (default: 0)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	if os.path.isfile(args.out_path):
		os.remove(args.out_path)
		print(args.out_path + ' Removed')

	print('Cuda Mode is: {}'.format(args.cuda))
	print('Selected model is: {}'.format(args.model))

	if args.cuda:
		device = get_freer_gpu()

	if args.model == 'lstm':
		model = model_.cnn_lstm()
	elif args.model == 'resnet':
		model = model_.ResNet()
	elif args.model == 'resnet_pca':
		model = model_.ResNet_pca()
	elif args.model == 'lcnn_9':
		model = model_.lcnn_9layers()
	elif args.model == 'lcnn_29':
		model = model_.lcnn_29layers_v2()
	elif args.model == 'lcnn_9_pca':
		model = model_.lcnn_9layers_pca()
	elif args.model == 'lcnn_29_pca':
		model = model_.lcnn_29layers_v2_pca()
	elif args.model == 'lcnn_9_icqspec':
		model = model_.lcnn_9layers_icqspec()
	elif args.model == 'lcnn_9_prodspec':
		model = model_.lcnn_9layers_prodspec()
	elif args.model == 'lcnn_9_CC':
		model = model_.lcnn_9layers_CC(ncoef=args.ncoef, init_coef=args.init_coef)
	elif args.model == 'lcnn_29_CC':
		model = model_.lcnn_29layers_CC(ncoef=args.ncoef, init_coef=args.init_coef)
	elif args.model == 'resnet_34_CC':
		model = model_.ResNet_34_CC(ncoef=args.ncoef, init_coef=args.init_coef)

	print('Loading model')

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=True)
	model.eval()

	print('Model loaded')

	print('Loading data')

	if args.eval:
		test_utts = read_trials(args.trials_path, eval_=args.eval)
	else:
		test_utts, attack_type_list, label_list = read_trials(args.trials_path, eval_=args.eval)

	data = { k:m for k,m in read_mat_scp(args.path_to_data) }

	if args.tandem:
		data = change_keys(data)

	print('Data loaded')

	print('Start of scores computation')

	score_list = []

	with torch.no_grad():

		for i, utt in enumerate(test_utts):

			feats = prep_feats(data[utt])

			try:
				if args.cuda:
					feats = feats.to(device)
					model = model.to(device)

				score = 1.-torch.sigmoid(model.forward(feats)).item()

			except:
				feats = feats.cpu()
				model = model.cpu()

				score = 1.-torch.sigmoid(model.forward(feats)).item()

			score_list.append(score)

	if not args.no_output_file:

		print('Storing scores in output file:')
		print(args.out_path)

		with open(args.out_path, 'w') as f:
			if args.eval:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, str(score_list[i])+'\n']))
			else:
				for i, utt in enumerate(test_utts):
					f.write("%s" % ' '.join([utt, attack_type_list[i], label_list[i], str(score_list[i])+'\n']))

	if not args.no_eer and not args.eval:
		print('EER: {}'.format(compute_eer_labels(label_list, score_list)))

	print('All done!!')
