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
	parser.add_argument('--path1-to-data', type=str, default='./data/feats1.scp', metavar='Path', help='Path to input data 1')
	parser.add_argument('--path2-to-data', type=str, default='./data/feats1.scp', metavar='Path', help='Path to input data 2')
	parser.add_argument('--trials-path', type=str, default=None, metavar='Path', help='Path to trials file')
	parser.add_argument('--cp1-path', type=str, default=None, metavar='Path', help='Path for file containing first model')
	parser.add_argument('--cp2-path', type=str, default=None, metavar='Path', help='Path for file containing second model')
	parser.add_argument('--out-path', type=str, default='./out.txt', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--model1', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'TDNN'], default='lcnn_9', help='Model arch')
	parser.add_argument('--resnet1-type', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
	parser.add_argument('--model2', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'TDNN'], default='lcnn_9', help='Model arch')
	parser.add_argument('--resnet2-type', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--prefix', type=str, default='./scores', metavar='Path', help='prefix for score files names')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	parser.add_argument('--no-eer', action='store_true', default=False, help='Disables computation of EER')
	parser.add_argument('--eval', action='store_true', default=False, help='Enables eval trials reading')
	parser.add_argument('--ncoef1', type=int, default=90, metavar='N', help='Number of cepstral coefs (default: 90)')
	parser.add_argument('--init-coef1', type=int, default=0, metavar='N', help='First cepstral coefs (default: 0)')
	parser.add_argument('--ncoef2', type=int, default=90, metavar='N', help='Number of cepstral coefs (default: 90)')
	parser.add_argument('--init-coef2', type=int, default=0, metavar='N', help='First cepstral coefs (default: 0)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp1_path is None or args.cp2_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp1-path and --cp2-path to indicate the paths!')

	print('Cuda Mode is: {}'.format(args.cuda))
	print('Selected model is: {}'.format(args.model))

	if args.cuda:
		device = get_freer_gpu()

	if args.model1 == 'lstm':
		model1 = model_.cnn_lstm()
	elif args.model1 == 'resnet':
		model1 = model_.ResNet(resnet_type=args.resnet1_type)
	elif args.model1 == 'resnet_pca':
		model1 = model_.ResNet_pca(resnet_type=args.resnet1_type)
	elif args.model1 == 'lcnn_9':
		model1 = model_.lcnn_9layers()
	elif args.model1 == 'lcnn_29':
		model1 = model_.lcnn_29layers_v2()
	elif args.model1 == 'lcnn_9_pca':
		model1 = model_.lcnn_9layers_pca()
	elif args.model1 == 'lcnn_29_pca':
		model1 = model_.lcnn_29layers_v2_pca()
	elif args.model1 == 'lcnn_9_icqspec':
		model1 = model_.lcnn_9layers_icqspec()
	elif args.model1 == 'lcnn_9_prodspec':
		model1 = model_.lcnn_9layers_prodspec()
	elif args.model1 == 'lcnn_9_CC':
		model1 = model_.lcnn_9layers_CC(ncoef=args.ncoef1, init_coef=args.init_coef1)
	elif args.model1 == 'lcnn_29_CC':
		model1 = model_.lcnn_29layers_CC(ncoef=args.ncoef1, init_coef=args.init_coef1)
	elif args.model1 == 'resnet_CC':
		model1 = model_.ResNet_CC(ncoef=args.ncoef1, init_coef=args.init_coef1, resnet_type=args.resnet1_type)
	elif args.model1 == 'TDNN':
		model1 = model_.TDNN(ncoef=args.ncoef1, init_coef=args.init_coef1)

	if args.model2 == 'lstm':
		model2 = model_.cnn_lstm()
	elif args.model2 == 'resnet':
		model2 = model_.ResNet(resnet_type=args.resnet2_type)
	elif args.model2 == 'resnet_pca':
		model2 = model_.ResNet_pca(resnet_type=args.resnet2_type)
	elif args.model2 == 'lcnn_9':
		model2 = model_.lcnn_9layers()
	elif args.model2 == 'lcnn_29':
		model2 = model_.lcnn_29layers_v2()
	elif args.model2 == 'lcnn_9_pca':
		model2 = model_.lcnn_9layers_pca()
	elif args.model2 == 'lcnn_29_pca':
		model2 = model_.lcnn_29layers_v2_pca()
	elif args.model2 == 'lcnn_9_icqspec':
		model2 = model_.lcnn_9layers_icqspec()
	elif args.model2 == 'lcnn_9_prodspec':
		model2 = model_.lcnn_9layers_prodspec()
	elif args.model2 == 'lcnn_9_CC':
		model2 = model_.lcnn_9layers_CC(ncoef=args.ncoef2, init_coef=args.init_coef2)
	elif args.model2 == 'lcnn_29_CC':
		model2 = model_.lcnn_29layers_CC(ncoef=args.ncoef2, init_coef=args.init_coef2)
	elif args.model2 == 'resnet_CC':
		model2 = model_.ResNet_CC(ncoef=args.ncoef2, init_coef=args.init_coef2, resnet_type=args.resnet2_type)
	elif args.model2 == 'TDNN':
		model2 = model_.TDNN(ncoef=args.ncoef2, init_coef=args.init_coef2)

	print('Loading model1')

	ckpt1 = torch.load(args.cp1_path, map_location = lambda storage, loc: storage)
	model1.load_state_dict(ckpt1['model_state'], strict=True)
	model1.eval()

	ckpt2 = torch.load(args.cp2_path, map_location = lambda storage, loc: storage)
	model2.load_state_dict(ckpt2['model_state'], strict=True)
	model2.eval()

	print('Model loaded')

	print('Loading data')

	data1 = { k:m for k,m in read_mat_scp(args.path1_to_data) }
	data2 = { k:m for k,m in read_mat_scp(args.path2_to_data) }

	if args.trials_path:
		if args.eval:
			test_utts = read_trials(args.trials_path, eval_=args.eval)
		else:
			test_utts, attack_type_list, label_list = read_trials(args.trials_path, eval_=args.eval)
	else:
		test_utts = list(data1.keys())

	print('Data loaded')

	print('Start of scores computation')

	score_types = ['max', 'min', 'avg']

	scores = {x:[] for x in score_types}
	preds = {x:[] for x in score_types}

	skipped_utterances = 0

	with torch.no_grad():

		for i, utt in enumerate(test_utts):

			try:
				feats1 = prep_feats(data1[utt])
				feats2 = prep_feats(data2[utt])
			except KeyError:
				print('\nSkipping utterance {}. Not found within the data\n'.format(utt))
				skipped_utterances+=1
				continue

			try:
				if args.cuda:
					feats1 = feats1.to(device)
					feats2 = feats2.to(device)
					model1 = model1.to(device)
					model2 = model2.to(device)

				pred1 = torch.sigmoid(model1.forward(feats1)).item()
				pred2 = torch.sigmoid(model2.forward(feats2)).item()
				pred_max = max(pred1, pred2)
				pred_min = min(pred1, pred2)
				pred_avg = 0.5*(pred1 + pred2)
				score_max = 1.-pred_max
				score_min = 1.-pred_min
				score_avg = 1.-pred_avg

			except:
				feats = feats.cpu()
				model = model.cpu()

				pred1 = torch.sigmoid(model1.forward(feats1)).item()
				pred2 = torch.sigmoid(model2.forward(feats2)).item()
				pred_max = max(pred1, pred2)
				pred_min = min(pred1, pred2)
				pred_avg = 0.5*(pred1 + pred2)
				score_max = 1.-pred_max
				score_min = 1.-pred_min
				score_avg = 1.-pred_avg

			scores['max'].append(score_max)
			scores['min'].append(score_min)
			scores['avg'].append(score_avg)
			preds['max'].append(pred_max)
			preds['min'].append(pred_min)
			preds['avg'].append(pred_avg)

	if not args.no_output_file:

		print('Storing scores in folder:')
		print(args.out_path)

		for score_type in score_types:

			file_name = os.path.join(args.out_path, args.prefix, score_type,'.txt')

			with open(file_name, 'w') as f:
				if args.eval or args.trials_path is None:
					for i, utt in enumerate(test_utts):
						f.write("%s" % ' '.join([utt, str(scores[score_type][i])+'\n']))
				else:
					for i, utt in enumerate(test_utts):
						f.write("%s" % ' '.join([utt, attack_type_list[i], label_list[i], str(scores[score_type][i])+'\n']))

	if not args.no_eer and not args.eval and args.trials_path:
		for score_type in score_types:
			print('\nPerformance of scores of type {}'.format(score_types))
			print('EER: {}'.format(compute_eer_labels(label_list, scores[score_types])))
			print('BCE: {}'.format(torch.nn.BCELoss()(torch.Tensor(preds[score_types]), torch.Tensor(y))))

	print('All done!!')
	print('\nTotal skipped trials: {}'.format(skipped_utterances))