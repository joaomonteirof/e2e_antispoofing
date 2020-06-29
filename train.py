from __future__ import print_function
import argparse
import torch
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader
from torch.utils.tensorboard import SummaryWriter
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='Speaker embbedings with contrastive loss')
parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'TDNN', 'TDNN_multipool', 'TDNN_LSTM', 'FTDNN', 'mobilenet', 'densenet', 'VGG'], default='lcnn_9', help='Model arch')
parser.add_argument('--resnet-type', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
parser.add_argument('--vgg-type', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for validation (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='alpha', help='Alpha (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--patience', type=int, default=5, metavar='N', help='number of epochs without improvement to wait before reducing lr (default: 5)')
parser.add_argument('--smoothing', type=float, default=0.1, metavar='l', help='Label smoothing (default: 0.1) - Disable by setting it to 0')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path for pre trained model')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--train-hdf-path', type=str, default='./data/train.hdf', metavar='Path', help='Path to hdf data')
parser.add_argument('--valid-hdf-path', type=str, default=None, metavar='Path', help='Path to hdf data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--n-frames', type=int, default=1000, metavar='N', help='maximum number of frames per utterance (default: 1000)')
parser.add_argument('--n-cycles', type=int, default=30000, metavar='N', help='number of examples to complete 1 epoch')
parser.add_argument('--valid-n-cycles', type=int, default=1000, metavar='N', help='number of examples to complete 1 epoch')
parser.add_argument('--ncoef', type=int, default=90, metavar='N', help='Number of cepstral coefs for the LA case (default: 90)')
parser.add_argument('--init-coef', type=int, default=0, metavar='N', help='First cepstral coefs (default: 0)')
parser.add_argument('--lists-path', type=str, default=None, metavar='Path', help='Path to list files per attack')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

print(args)

if args.cuda:
	device = get_freer_gpu()
else:
	device = torch.device('cpu')

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model, purge_step=True)
else:
	writer = None

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

train_dataset = Loader(hdf5_clean = args.train_hdf_path+'train_clean.hdf', hdf5_attack = args.train_hdf_path+'train_attack.hdf', max_nb_frames = args.n_frames, n_cycles=args.n_cycles, augment=True, label_smoothing=args.smoothing)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

if args.valid_hdf_path is not None:
	valid_dataset = Loader(hdf5_clean = args.valid_hdf_path+'valid_clean.hdf', hdf5_attack = args.valid_hdf_path+'valid_attack.hdf', max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles, augment=False, label_smoothing=0.0)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, worker_init_fn=set_np_randomseed)
else:
	valid_loader=None

if args.model == 'lstm':
	model = model_.cnn_lstm()
elif args.model == 'resnet':
	model = model_.ResNet(resnet_type=args.resnet_type)
elif args.model == 'VGG':
	model = model_.VGG(vgg_name=args.vgg_type)
elif args.model == 'resnet_pca':
	model = model_.ResNet_pca(resnet_type=args.resnet_type)
elif args.model == 'wideresnet':
	model = model_.WideResNet()
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
elif args.model == 'resnet_CC':
	model = model_.ResNet_CC(ncoef=args.ncoef, init_coef=args.init_coef, resnet_type=args.resnet_type)
elif args.model == 'TDNN':
	model = model_.TDNN(ncoef=args.ncoef, init_coef=args.init_coef)
elif args.model == 'TDNN_multipool':
	model = model_.TDNN_multipool(ncoef=args.ncoef, init_coef=args.init_coef)
elif args.model == 'TDNN_LSTM':
	model = model_.TDNN_LSTM(ncoef=args.ncoef)
elif args.model == 'FTDNN':
	model = model_.FTDNN(ncoef=args.ncoef, init_coef=args.init_coef)
elif args.model == 'mobilenet':
	model = model_.MobileNetV3_Small()
elif args.model == 'densenet':
	model = model_.DenseNet()

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

model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, patience=args.patience, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, logger=writer)

print('Cuda Mode: {}'.format(args.cuda))
print('Device: {}'.format(device))
print('Selected model: {}'.format(args.model))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('Momentum: {}'.format(args.momentum))
print('l2: {}'.format(args.l2))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)