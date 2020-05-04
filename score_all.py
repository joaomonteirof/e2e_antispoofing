import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import os
from kaldi_io import read_mat_scp
import model as model_
import scipy.io as sio

from utils import *

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
	parser.add_argument('--path-to-data-la', type=str, default='./data_la/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--path-to-data-pa', type=str, default='./data_pa/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--path-to-data-mix', type=str, default='./data_mix/feats.scp', metavar='Path', help='Path to input data')
	parser.add_argument('--model-la', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'Linear', 'TDNN', 'TDNN_multipool', 'TDNN_LSTM', 'FTDNN', 'mobilenet', 'densenet', 'VGG'], default='lcnn_29_CC', help='Model arch')
	parser.add_argument('--model-pa', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'Linear', 'TDNN', 'TDNN_multipool', 'TDNN_LSTM', 'FTDNN', 'mobilenet', 'densenet', 'VGG'], default='lcnn_9_prodspec', help='Model arch')
	parser.add_argument('--model-mix', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'Linear', 'TDNN', 'TDNN_multipool', 'FTDNN', 'TDNN_LSTM', 'mobilenet', 'densenet', 'VGG'], default='lcnn_29_CC', help='Model arch')
	parser.add_argument('--vgg-type-la', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
	parser.add_argument('--vgg-type-pa', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
	parser.add_argument('--vgg-type-mix', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
	parser.add_argument('--resnet-type-la', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
	parser.add_argument('--resnet-type-pa', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
	parser.add_argument('--resnet-type-mix', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
	parser.add_argument('--train-mode', choices=['mix', 'lapa', 'independent'], default='mix', help='Train mode')
	parser.add_argument('--trials-path', type=str, default=None, metavar='Path', help='Path to trials file')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for file containing model')
	parser.add_argument('--out-path', type=str, default='./', metavar='Path', help='Path to output hdf file')
	parser.add_argument('--prefix', type=str, default='./scores', metavar='Path', help='prefix for score files names')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-output-file', action='store_true', default=False, help='Disables writing scores into out file')
	parser.add_argument('--no-eer', action='store_true', default=False, help='Disables computation of EER')
	parser.add_argument('--eval', action='store_true', default=False, help='Enables eval trials reading')
	parser.add_argument('--ncoef-la', type=int, default=90, metavar='N', help='Number of cepstral coefs for the CC case (default: 90)')
	parser.add_argument('--ncoef-pa', type=int, default=90, metavar='N', help='Number of cepstral coefs for the CC case (default: 90)')
	parser.add_argument('--ncoef-mix', type=int, default=90, metavar='N', help='Number of cepstral coefs for the CC case (default: 90)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	if os.path.isfile(args.out_path):
		os.remove(args.out_path)
		print(args.out_path + ' Removed')

	if args.cuda:
		device = get_freer_gpu()

	if args.model_la == 'lstm':
		model_la = model_.cnn_lstm()
	elif args.model_la == 'VGG':
		model_la = model_.VGG(vgg_name=args.vgg_type_la)
	elif args.model_la == 'resnet':
		model_la = model_.ResNet(resnet_type=args.resnet_type_la)
	elif args.model_la == 'resnet_pca':
		model_la = model_.ResNet_pca(resnet_type=args.resnet_type_la)
	elif args.model_la == 'wideresnet':
		model_la = model_.WideResNet()
	elif args.model_la == 'lcnn_9':
		model_la = model_.lcnn_9layers()
	elif args.model_la == 'lcnn_29':
		model_la = model_.lcnn_29layers_v2()
	elif args.model_la == 'lcnn_9_pca':
		model_la = model_.lcnn_9layers_pca()
	elif args.model_la == 'lcnn_29_pca':
		model_la = model_.lcnn_29layers_v2_pca()
	elif args.model_la == 'lcnn_9_icqspec':
		model_la = model_.lcnn_9layers_icqspec()
	elif args.model_la == 'lcnn_9_prodspec':
		model_la = model_.lcnn_9layers_prodspec()
	elif args.model_la == 'lcnn_9_CC':
		model_la = model_.lcnn_9layers_CC(ncoef=args.ncoef_la)
	elif args.model_la == 'lcnn_29_CC':
		model_la = model_.lcnn_29layers_CC(ncoef=args.ncoef_la)
	elif args.model_la == 'resnet_CC':
		model_la = model_.ResNet_CC(ncoef=args.ncoef_la, resnet_type=args.resnet_type_la)
	elif args.model_la == 'TDNN':
		model_la = model_.TDNN(ncoef=args.ncoef_la)
	elif args.model_la == 'TDNN_multipool':
		model_la = model_.TDNN_multipool(ncoef=args.ncoef_la)
	elif args.model_la == 'TDNN_LSTM':
		model_la = model_.TDNN_LSTM(ncoef=args.ncoef_la)
	elif args.model_la == 'FTDNN':
		model_la = model_.FTDNN(ncoef=args.ncoef_la)
	elif args.model_la == 'Linear':
		model_la = model_.Linear(ncoef=args.ncoef_la)
	elif args.model_la == 'mobilenet':
		model_la = model_.MobileNetV3_Small()
	elif args.model_la == 'densenet':
		model_la = model_.DenseNet()

	if args.model_pa == 'lstm':
		model_pa = model_.cnn_lstm()
	elif args.model_pa == 'VGG':
		model_pa = model_.VGG(vgg_name=args.vgg_type_pa)
	elif args.model_pa == 'resnet':
		model_pa = model_.ResNet(resnet_type=args.resnet_type_pa)
	elif args.model_pa == 'resnet_pca':
		model_pa = model_.ResNet_pca(resnet_type=args.resnet_type_pa)
	elif args.model_pa == 'wideresnet':
		model_pa = model_.WideResNet()
	elif args.model_pa == 'lcnn_9':
		model_pa = model_.lcnn_9layers()
	elif args.model_pa == 'lcnn_29':
		model_pa = model_.lcnn_29layers_v2()
	elif args.model_pa == 'lcnn_9_pca':
		model_pa = model_.lcnn_9layers_pca()
	elif args.model_pa == 'lcnn_29_pca':
		model_pa = model_.lcnn_29layers_v2_pca()
	elif args.model_pa == 'lcnn_9_icqspec':
		model_pa = model_.lcnn_9layers_icqspec()
	elif args.model_pa == 'lcnn_9_prodspec':
		model_pa = model_.lcnn_9layers_prodspec()
	elif args.model_pa == 'lcnn_9_CC':
		model_pa = model_.lcnn_9layers_CC(ncoef=args.ncoef_pa)
	elif args.model_pa == 'lcnn_29_CC':
		model_pa = model_.lcnn_29layers_CC(ncoef=args.ncoef_pa)
	elif args.model_pa == 'resnet_CC':
		model_pa = model_.ResNet_CC(ncoef=args.ncoef_pa, resnet_type=args.resnet_type_pa)
	elif args.model_pa == 'TDNN':
		model_pa = model_.TDNN(ncoef=args.ncoef_pa)
	elif args.model_pa == 'TDNN_multipool':
		model_pa = model_.TDNN_multipool(ncoef=args.ncoef_pa)
	elif args.model_pa == 'TDNN_LSTM':
		model_pa = model_.TDNN_LSTM(ncoef=args.ncoef_pa)
	elif args.model_pa == 'FTDNN':
		model_pa = model_.FTDNN(ncoef=args.ncoef_pa)
	elif args.model_pa == 'Linear':
		model_pa = model_.Linear(ncoef=args.ncoef_pa)
	elif args.model_pa == 'mobilenet':
		model_pa = model_.MobileNetV3_Small()
	elif args.model_pa == 'densenet':
		model_pa = model_.DenseNet()

	if args.model_mix == 'lstm':
		model_mix = model_.cnn_lstm()
	elif args.model_mix == 'VGG':
		model_mix = model_.VGG(vgg_name=args.vgg_type_mix)
	elif args.model_mix == 'resnet':
		model_mix = model_.ResNet(resnet_type=args.resnet_type_mix)
	elif args.model_mix == 'resnet_pca':
		model_mix = model_.ResNet_pca(resnet_type=args.resnet_type_mix)
	elif args.model_mix == 'wideresnet':
		model_mix = model_.WideResNet()
	elif args.model_mix == 'lcnn_9':
		model_mix = model_.lcnn_9layers()
	elif args.model_mix == 'lcnn_29':
		model_mix = model_.lcnn_29layers_v2()
	elif args.model_mix == 'lcnn_9_pca':
		model_mix = model_.lcnn_9layers_pca()
	elif args.model_mix == 'lcnn_29_pca':
		model_mix = model_.lcnn_29layers_v2_pca()
	elif args.model_mix == 'lcnn_9_icqspec':
		model_mix = model_.lcnn_9layers_icqspec()
	elif args.model_mix == 'lcnn_9_prodspec':
		model_mix = model_.lcnn_9layers_prodspec()
	elif args.model_mix == 'lcnn_9_CC':
		model_mix = model_.lcnn_9layers_CC(ncoef=args.ncoef_mix)
	elif args.model_mix == 'lcnn_29_CC':
		model_mix = model_.lcnn_29layers_CC(ncoef=args.ncoef_mix)
	elif args.model_mix == 'resnet_CC':
		model_mix = model_.ResNet_CC(ncoef=args.ncoef_mix, resnet_type=args.resnet_type_mix)
	elif args.model_mix == 'TDNN':
		model_mix = model_.FTDNN(ncoef=args.ncoef_mix)
	elif args.model_mix == 'TDNN_multipool':
		model_mix = model_.TDNN_multipool(ncoef=args.ncoef_mix)
	elif args.model_mix == 'TDNN_LSTM':
		model_mix = model_.TDNN_LSTM(ncoef=args.ncoef_mix)
	elif args.model_mix == 'FTDNN':
		model_mix = model_.TDNN(ncoef=args.ncoef_mix)
	elif args.model_mix == 'Linear':
		model_mix = model_.Linear(ncoef=args.ncoef_mix)
	elif args.model_mix == 'mobilenet':
		model_mix = model_.MobileNetV3_Small()
	elif args.model_mix == 'densenet':
		model_mix = model_.DenseNet()

	print('Loading model')

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)
	model_la.load_state_dict(ckpt['model_la_state'])
	model_pa.load_state_dict(ckpt['model_pa_state'])
	model_mix.load_state_dict(ckpt['model_mix_state'])
	model_la.eval()
	model_pa.eval()
	model_mix.eval()

	print('Model loaded')

	print('Loading data')

	data_la = { k:m for k,m in read_mat_scp(args.path_to_data_la) }
	data_pa = { k:m for k,m in read_mat_scp(args.path_to_data_pa) }
	data_mix = { k:m for k,m in read_mat_scp(args.path_to_data_mix) }


	if args.trials_path:
		if args.eval:
			test_utts = read_trials(args.trials_path, eval_=args.eval)
		else:
			test_utts, attack_type_list, label_list = read_trials(args.trials_path, eval_=args.eval)
	else:
		test_utts = list(data_la.keys())

	print('Data loaded')

	print('Start of scores computation')

	scores = {}

	scores['all'] = []
	scores['la'] = []
	scores['pa'] = []
	scores['mix'] = []
	scores['fusion'] = []

	with torch.no_grad():

		for i, utt in enumerate(test_utts):

			feats_la = prep_feats(data_la[utt])
			feats_pa = prep_feats(data_pa[utt])
			feats_mix = prep_feats(data_mix[utt])

			try:
				if args.cuda:
					feats_la = feats_la.to(device)
					feats_pa = feats_pa.to(device)
					feats_mix = feats_mix.to(device)
					model_la = model_la.to(device)
					model_pa = model_pa.to(device)
					model_mix = model_mix.to(device)

				pred_la = model_la.forward(feats_la).squeeze()
				pred_pa = model_pa.forward(feats_pa).squeeze()
				pred_mix = model_mix.forward(feats_mix).squeeze()

			except:
				feats_la = feats_la.cpu()
				feats_pa = feats_pa.cpu()
				feats_mix = feats_mix.cpu()
				model_la = model_la.cpu()
				model_pa = model_pa.cpu()
				model_mix = model_mix.cpu()

				pred_la = model_la.forward(feats_la).squeeze()
				pred_pa = model_pa.forward(feats_pa).squeeze()
				pred_mix = model_mix.forward(feats_mix).squeeze()

			if args.train_mode == 'mix':
				mixture_coef = 1.0-torch.sigmoid(pred_mix).squeeze()
				score_all = 1.0-torch.sigmoid(mixture_coef*pred_la + (1.-mixture_coef)*pred_pa).squeeze().cpu().item()
				score_la = 1.0-torch.sigmoid(pred_la).squeeze().cpu().item()
				score_pa = 1.0-torch.sigmoid(pred_pa).squeeze().cpu().item()
				score_mix = 1.0-2*abs(mixture_coef.cpu().item()-0.5)
				score_fusion = (score_all+score_la+score_pa+score_mix)/4.

			elif args.train_mode == 'lapa':
				score_all = 0.0
				score_la = 1.0-2*abs(torch.sigmoid(pred_la)-0.5).cpu().numpy().item()
				score_pa = 1.0-2*abs(torch.sigmoid(pred_pa)-0.5).cpu().numpy().item()
				score_mix = 1.0-2*abs(torch.sigmoid(pred_mix)-0.5).cpu().numpy().item()
				score_fusion = (score_la+score_pa+score_mix)/3.

			elif args.train_mode == 'independent':
				score_all = 0.0
				score_la = 1.0-torch.sigmoid(pred_la).cpu().numpy().item()
				score_pa = 1.0-torch.sigmoid(pred_pa).cpu().numpy().item()
				score_mix = 1.0-torch.sigmoid(pred_mix).cpu().numpy().item()
				score_fusion = (score_la+score_pa+score_mix)/3.

			scores['all'].append(score_all)
			scores['la'].append(score_la)
			scores['pa'].append(score_pa)
			scores['mix'].append(score_mix)
			scores['fusion'].append(score_fusion)

	if not args.no_output_file:

		print('Storing scores in output file:')
		print(args.out_path)

		for score_type, score_list in scores.items():

			if (args.train_mode=='lapa' or args.train_mode=='independent') and score_type=='all':
				continue

			file_name = args.out_path+args.prefix+'_'+score_type+'.txt'

			with open(file_name, 'w') as f:
				if args.eval or args.trials_path is None:
					for i, utt in enumerate(test_utts):
						f.write("%s" % ' '.join([utt, str(score_list[i])+'\n']))
				else:
					for i, utt in enumerate(test_utts):
						f.write("%s" % ' '.join([utt, attack_type_list[i], label_list[i], str(score_list[i])+'\n']))

	if not args.no_eer and not args.eval and args.trials_path:
		for score_type, score_list in scores.items():
			print('\nEER {}: {}\n'.format(score_type, compute_eer_labels(label_list, score_list)))

	print('All done!!')
