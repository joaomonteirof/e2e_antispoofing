from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
from train_loop_mcc import TrainLoop_mcc
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader, Loader_mcc

from utils import *

# Training settings
parser = argparse.ArgumentParser(description='Speaker embbedings with contrastive loss')
parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_34_CC'], default='lcnn_9', help='Model arch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for validation (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='alpha', help='Alpha (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--train-hdf-path', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--n-frames', type=int, default=1000, metavar='N', help='maximum number of frames per utterance (default: 1000)')
parser.add_argument('--n-cycles', type=int, default=30000, metavar='N', help='number of examples to complete 1 epoch')
parser.add_argument('--valid-n-cycles', type=int, default=1000, metavar='N', help='number of examples to complete 1 epoch')
parser.add_argument('--n-classes', type=int, default=-1, metavar='N', help='Number of classes for the mcc case (default: binary classification)')
parser.add_argument('--ncoef', type=int, default=90, metavar='N', help='Number of cepstral coefs for the LA case (default: 90)')
parser.add_argument('--init-coef', type=int, default=0, metavar='N', help='First cepstral coefs (default: 0)')
parser.add_argument('--lists-path', type=str, default=None, metavar='Path', help='Path to list files per attack')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	device = get_freer_gpu()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

if args.n_classes>2:
	assert args.lists_path is not None, 'Pass the path for the lists of utterances per attack'
	train_dataset = Loader_mcc(hdf5_clean = args.train_hdf_path+'train_clean.hdf', hdf5_attack = args.train_hdf_path+'train_attack.hdf', max_nb_frames = args.n_frames, n_cycles=args.n_cycles, file_lists_path=args.lists_path)
else:
	train_dataset = Loader(hdf5_clean = args.train_hdf_path+'train_clean.hdf', hdf5_attack = args.train_hdf_path+'train_attack.hdf', max_nb_frames = args.n_frames, n_cycles=args.n_cycles)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

if args.valid_hdf_path is not None:
	if args.n_classes>2:	
		valid_dataset = Loader_mcc(hdf5_clean = args.valid_hdf_path+'valid_clean.hdf', hdf5_attack = args.valid_hdf_path+'valid_attack.hdf', max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles, file_lists_path=args.lists_path)
	else:
		valid_dataset = Loader(hdf5_clean = args.valid_hdf_path+'valid_clean.hdf', hdf5_attack = args.valid_hdf_path+'valid_attack.hdf', max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, worker_init_fn=set_np_randomseed)
else:
	valid_loader=None

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

if args.pretrained_path is not None:
	ckpt = torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)

	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
		model.load_state_dict(ckpt['model_state'], strict=False)
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.cuda:
	model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

if args.n_classes>2:
	trainer = TrainLoop_mcc(model, optimizer, train_loader, valid_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)
else:
	trainer = TrainLoop(model, optimizer, train_loader, valid_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode: {}'.format(args.cuda))
print('Device: {}'.format(device))
print('Selected model: {}'.format(args.model))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('Momentum: {}'.format(args.momentum))
print('l2: {}'.format(args.l2))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
