from __future__ import print_function
import argparse
import torch
from train_loop_all import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader_all, Loader_all_valid

from utils import *

# Training settings
parser = argparse.ArgumentParser(description='Speaker embbedings with contrastive loss')
parser.add_argument('--model-la', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC'], default='lcnn_29_CC', help='Model arch')
parser.add_argument('--model-pa', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC'], default='lcnn_9_prodspec', help='Model arch')
parser.add_argument('--model-mix', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC'], default='lcnn_29_CC', help='Model arch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for validation (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='alpha', help='Alpha (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--patience', type=int, default=5, metavar='N', help='number of epochs without improvement to wait before reducing lr (default: 5)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-la-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--pretrained-pa-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--pretrained-mix-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--train-la-path', type=str, default='./data/', metavar='Path', help='Path to folder containing hdfs with clean and attack data')
parser.add_argument('--train-pa-hdf', type=str, default='./data/pa.hdf', metavar='Path', help='Path to single hdf file containing all clean and attack recordings')
parser.add_argument('--train-mix-hdf', type=str, default='./data/mix.hdf', metavar='Path', help='Path to single hdf file containing all clean and attack recordings')
parser.add_argument('--valid-la-path', type=str, default='./data/', metavar='Path', help='Path to folder containing hdfs with clean and attack data')
parser.add_argument('--valid-pa-hdf', type=str, default='./data/pa.hdf', metavar='Path', help='Path to single hdf file containing all clean and attack recordings')
parser.add_argument('--valid-mix-hdf', type=str, default='./data/mix.hdf', metavar='Path', help='Path to single hdf file containing all clean and attack recordings')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--n-frames', type=int, default=1000, metavar='N', help='maximum number of frames per utterance (default: 1000)')
parser.add_argument('--n-cycles', type=int, default=30000, metavar='N', help='number of examples to complete 1 epoch')
parser.add_argument('--valid-n-cycles', type=int, default=1000, metavar='N', help='number of examples to complete 1 epoch')
parser.add_argument('--ncoef-la', type=int, default=90, metavar='N', help='Number of cepstral coefs for the CC case (default: 90)')
parser.add_argument('--ncoef-pa', type=int, default=90, metavar='N', help='Number of cepstral coefs for the CC case (default: 90)')
parser.add_argument('--ncoef-mix', type=int, default=90, metavar='N', help='Number of cepstral coefs for the CC case (default: 90)')
parser.add_argument('--lists-path', type=str, default=None, metavar='Path', help='Path to list files per attack')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	device = get_freer_gpu()
else:
	device = torch.device('cpu')

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

train_dataset = Loader_all(hdf5_la_clean = args.train_la_path+'train_clean.hdf', hdf5_la_attack = args.train_la_path+'train_attack.hdf', hdf5_pa=args.train_pa_hdf, hdf5_mix=args.train_mix_hdf, max_nb_frames = args.n_frames, n_cycles=args.n_cycles)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

valid_dataset = Loader_all_valid(hdf5_la_clean = args.valid_la_path+'valid_clean.hdf', hdf5_la_attack = args.valid_la_path+'valid_attack.hdf', hdf5_pa=args.valid_pa_hdf, hdf5_mix=args.valid_mix_hdf, max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, worker_init_fn=set_np_randomseed)

if args.model_la == 'lstm':
	model_la = model_.cnn_lstm()
elif args.model_la == 'resnet':
	model_la = model_.ResNet()
elif args.model_la == 'resnet_pca':
	model_la = model_.ResNet_pca()
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
	model_la = model_.ResNet_CC(ncoef=args.ncoef_la)

if args.model_pa == 'lstm':
	model_pa = model_.cnn_lstm()
elif args.model_pa == 'resnet':
	model_pa = model_.ResNet()
elif args.model_pa == 'resnet_pca':
	model_pa = model_.ResNet_pca()
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
	model_pa = model_.ResNet_CC(ncoef=args.ncoef_pa)

if args.model_mix == 'lstm':
	model_mix = model_.cnn_lstm()
elif args.model_mix == 'resnet':
	model_mix = model_.ResNet()
elif args.model_mix == 'resnet_pca':
	model_mix = model_.ResNet_pca()
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
	model_mix = model_.ResNet_CC(ncoef=args.ncoef_mix)

if args.pretrained_la_path is not None:
	ckpt = torch.load(args.pretrained_la_path, map_location = lambda storage, loc: storage)
	try:
		model_la.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
		model_la.load_state_dict(ckpt['model_state'], strict=False)
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.pretrained_pa_path is not None:
	ckpt = torch.load(args.pretrained_pa_path, map_location = lambda storage, loc: storage)
	try:
		model_pa.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
		model_pa.load_state_dict(ckpt['model_state'], strict=False)
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.pretrained_mix_path is not None:
	ckpt = torch.load(args.pretrained_mix_path, map_location = lambda storage, loc: storage)
	try:
		model_mix.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
		model_mix.load_state_dict(ckpt['model_state'], strict=False)
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.cuda:
	model_la = model_la.to(device)
	model_pa = model_pa.to(device)
	model_mix = model_mix.to(device)

optimizer_la = optim.SGD(model_la.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
optimizer_pa = optim.SGD(model_pa.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
optimizer_mix = optim.SGD(model_mix.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model_la, model_pa, model_mix, optimizer_la, optimizer_pa, optimizer_mix, train_loader, valid_loader, patience=args.patience, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode: {}'.format(args.cuda))
print('Device: {}'.format(device))
print('Selected models (LA, PA, MIX): {}, {}, {}'.format(args.model_la, args.model_pa, args.model_mix))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('Momentum: {}'.format(args.momentum))
print('l2: {}'.format(args.l2))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
