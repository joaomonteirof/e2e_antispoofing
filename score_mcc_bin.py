import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_mat_scp
import model as model_
import scipy.io as sio

from utils import compute_eer_labels, set_device, read_trials

def prep_feats(data_):

	#data_ = ( data_ - data_.mean(0) ) / data_.std(0)

	features = data_.T

	if features.shape[1]<50:
		mul = int(np.ceil(50/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :50]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Compute scores for mcc model')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_LA'], default='lcnn_9', help='Model arch')
	parser.add_argument('--n-classes', type=int, default=-1, metavar='N', help='Number of classes for the mcc case (default: binary classification)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	parser.add_argument('--no-eer', action='store_true', default=False, help='Disables computation of EER')
	parser.add_argument('--eval', action='store_true', default=False, help='Enables eval trials reading')
	parser.add_argument('--ncoef', type=int, default=90, metavar='N', help='Number of cepstral coefs for the LA case (default: 90)')
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
		set_device()

	if args.model == 'lstm':
		model = model_.cnn_lstm(nclasses=args.n_classes)
	elif args.model == 'resnet':
		model = model_.ResNet(nclasses=args.n_classes)
	elif args.model == 'resnet_pca':
		model = model_.ResNet_pca(nclasses=args.n_classes)
	elif args.model == 'lcnn_9':
		model = model_.lcnn_9layers(nclasses=args.n_classes)
	elif args.model == 'lcnn_29':
		model = model_.lcnn_29layers_v2(nclasses=args.n_classes)
	elif args.model == 'lcnn_9_pca':
		model = model_.lcnn_9layers_pca(nclasses=args.n_classes)
	elif args.model == 'lcnn_29_pca':
		model = model_.lcnn_29layers_v2_pca(nclasses=args.n_classes)
	elif args.model == 'lcnn_9_icqspec':
		model = model_.lcnn_9layers_icqspec(nclasses=args.n_classes)
	elif args.model == 'lcnn_9_prodspec':
		model = model_.lcnn_9layers_prodspec(nclasses=args.n_classes)

	print('Loading model')

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=False)
	model.eval()

	print('Model loaded')

	print('Loading data')

	if args.eval:
		test_utts = read_trials(args.trials_path, eval_=args.eval)
	else:
		test_utts, attack_type_list, label_list = read_trials(args.trials_path, eval_=args.eval)

	data = { k:m for k,m in read_mat_scp(args.path_to_data) }

	print('Data loaded')

	print('Start of scores computation')

	score_list = []

	with torch.no_grad():

		for i, utt in enumerate(test_utts):


			print('Computing score for utterance '+ utt)

			feats = prep_feats(data[utt])

			try:
				if args.cuda:
					feats = feats.cuda()
					model = model.cuda()

				score = 1.-F.softmax(model.forward(feats), dim=1)[:,1:].sum().item()

			except:
				feats = feats.cpu()
				model = model.cpu()

				score = 1.-F.softmax(model.forward(feats), dim=1)[:,1:].sum().item()

			score_list.append(score)

			print('Score: {}'.format(score_list[-1]))

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
