import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_mat_scp
import model as model_
import scipy.io as sio

from utils import set_device, read_trials, get_freer_gpu

def prep_feats(data_):

	#data_ = ( data_ - data_.mean(0) ) / data_.std(0)

	features = data_.T

	if features.shape[1]<50:
		mul = int(np.ceil(50/features.shape[1]))
		features = np.tile(features, (1, mul))
		features = features[:, :50]

	return torch.from_numpy(features[np.newaxis, np.newaxis, :, :]).float()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluate accuracy for mcc model')
	parser.add_argument('--path-to-data', type=str, default='./data/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--trials-path', type=str, default='./data/trials', metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_34_CC'], default='lcnn_9', help='Model arch')
	parser.add_argument('--n-classes', type=int, default=-1, metavar='N', help='Number of classes for the mcc case (default: binary classification)')
	parser.add_argument('--ncoef', type=int, default=90, metavar='N', help='Number of cepstral coefs (default: 90)')
	parser.add_argument('--init-coef', type=int, default=0, metavar='N', help='First cepstral coefs (default: 0)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	if os.path.isfile(args.out_path):
		os.remove(args.out_path)
		print(args.out_path + ' Removed')

	labels_dict = {'-':0, 'AA':1, 'AB':2, 'AC':3, 'BA':4, 'BB':5, 'BC':6, 'CA':7, 'CB':8, 'CC':9}

	print('Cuda Mode is: {}'.format(args.cuda))
	print('Selected model is: {}'.format(args.model))

	if args.cuda:
		device = get_freer_gpu()

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
	elif args.model == 'lcnn_9_CC':
		model = model_.lcnn_9layers_CC(nclasses=args.n_classes, ncoef=args.ncoef, init_coef=args.init_coef)
	elif args.model == 'lcnn_29_CC':
		model = model_.lcnn_29layers_CC(nclasses=args.n_classes, ncoef=args.ncoef, init_coef=args.init_coef)
	elif args.model == 'resnet_34_CC':
		model = model_.ResNet_34_CC(nclasses=args.n_classes, ncoef=args.ncoef, init_coef=args.init_coef)

	print('Loading model')

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'], strict=False)
	model.eval()

	print('Model loaded')

	print('Loading data')

	test_utts, attack_type_list, label_list = read_trials(args.trials_path)
	data = { k:m for k,m in read_mat_scp(args.path_to_data) }

	print('Data loaded')

	print('Start of predictions')

	correct, total = 0, 0

	with torch.no_grad():

		for i, utt in enumerate(test_utts):

			print('Computing prediction for utterance '+ utt)

			feats = prep_feats(data[utt])

			try:
				if args.cuda:
					feats = feats.to(device)
					model = model.to(device)

			except:
				feats = feats.cpu()
				model = model.cpu()

			pred = F.softmax(model.forward(feats), dim=1).max(1)[1].long().squeeze().item()

			if pred==labels_dict[attack_type_list[i]]:
				correct+=1
			total+=1

	print('All done!!')

	print('Accuracy: {}'.format(float(correct)/total))
