from __future__ import print_function
import argparse
import torch
from train_loop_all import TrainLoop
import torch.optim as optim
import torch.utils.data
import model as model_
import numpy as np
from data_load import Loader_all, Loader_all_valid
from torch.utils.tensorboard import SummaryWriter
from utils import *
from optimizer import TransformerOptimizer

# Training settings
parser = argparse.ArgumentParser(description='Speaker embbedings with contrastive loss')
parser.add_argument('--model-la', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'Linear', 'TDNN', 'TDNN_LSTM', 'FTDNN', 'mobilenet', 'densenet', 'VGG'], default='lcnn_29_CC', help='Model arch')
parser.add_argument('--model-pa', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'Linear', 'TDNN', 'TDNN_LSTM', 'FTDNN', 'mobilenet', 'densenet', 'VGG'], default='lcnn_9_prodspec', help='Model arch')
parser.add_argument('--model-mix', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'Linear', 'TDNN', 'TDNN_LSTM', 'FTDNN', 'mobilenet', 'densenet', 'VGG'], default='lcnn_29_CC', help='Model arch')
parser.add_argument('--resnet-type-la', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
parser.add_argument('--resnet-type-pa', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
parser.add_argument('--resnet-type-mix', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
parser.add_argument('--vgg-type-la', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
parser.add_argument('--vgg-type-pa', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
parser.add_argument('--vgg-type-mix', choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], default='VGG16', help='VGG arch')
parser.add_argument('--train-mode', choices=['mix', 'lapa', 'independent'], default='mix', help='Train mode')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for validation (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--lr-la', type=float, default=0.0, metavar='LR_LA', help='Used for LA model instead of lr if greater than 0')
parser.add_argument('--lr-pa', type=float, default=0.0, metavar='LR_PA', help='Used for PA model instead of lr if greater than 0')
parser.add_argument('--lr-mix', type=float, default=0.0, metavar='LR_MIX', help='Used for MIX model instead of lr if greater than 0')
parser.add_argument('--b1', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--b2', type=float, default=0.98, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001)')
parser.add_argument('--warmup', type=int, default=4000, metavar='N', help='Iterations until reach lr (default: 4000)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
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
parser.add_argument('--smoothing', type=float, default=0.1, metavar='l', help='Label smoothing (default: 0.1) - Disable by setting it to 0')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	device = get_freer_gpu()
else:
	device = torch.device('cpu')

if args.logdir:
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model_la+'-_-'+args.model_pa+'-_-'+args.model_mix, purge_step=True if args.checkpoint_epoch is None else False)
else:
	writer = None

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

train_dataset = Loader_all(hdf5_la_clean = args.train_la_path+'train_clean.hdf', hdf5_la_attack = args.train_la_path+'train_attack.hdf', hdf5_pa=args.train_pa_hdf, hdf5_mix=args.train_mix_hdf, max_nb_frames = args.n_frames, label_smoothing=args.smoothing, n_cycles=args.n_cycles)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

valid_dataset = Loader_all_valid(hdf5_la_clean = args.valid_la_path+'valid_clean.hdf', hdf5_la_attack = args.valid_la_path+'valid_attack.hdf', hdf5_pa=args.valid_pa_hdf, hdf5_mix=args.valid_mix_hdf, max_nb_frames = args.n_frames, n_cycles=args.valid_n_cycles)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, worker_init_fn=set_np_randomseed)

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
elif args.model_la == 'TDNN_LSTM':
	model_la = model_.TDNN_LSTM(ncoef=args.ncoef_la)
elif args.model_la == 'FTDNN':
	model_la = model_.FTDNN(ncoef=args.ncoef_la)
elif args.model_la == 'Linear':
	model_la = model_.Linear(ncoef=args.ncoef_la)
elif args.model_la == 'mobilenet':
	model_la = model_.MobileNetV2()
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
	model_mix = model_.TDNN(ncoef=args.ncoef_mix)
elif args.model_mix == 'TDNN_LSTM':
	model_mix = model_.TDNN_LSTM(ncoef=args.ncoef_mix)
elif args.model_mix == 'FTDNN':
	model_mix = model_.FTDNN(ncoef=args.ncoef_mix)
elif args.model_mix == 'Linear':
	model_mix = model_.Linear(ncoef=args.ncoef_mix)
elif args.model_mix == 'mobilenet':
	model_mix = model_.MobileNetV3_Small()
elif args.model_mix == 'densenet':
	model_mix = model_.DenseNet()

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

model_la = model_la.to(device)
model_pa = model_pa.to(device)
model_mix = model_mix.to(device)

optimizer_la = TransformerOptimizer(optim.Adam(model_la.parameters(), betas=(args.b1, args.b2), weight_decay=args.l2), lr=args.lr_la if args.lr_la>0.0 else args.lr, warmup_steps=args.warmup)
optimizer_pa = TransformerOptimizer(optim.Adam(model_pa.parameters(), betas=(args.b1, args.b2), weight_decay=args.l2), lr=args.lr_pa if args.lr_pa>0.0 else args.lr, warmup_steps=args.warmup)
optimizer_mix = TransformerOptimizer(optim.Adam(model_mix.parameters(), betas=(args.b1, args.b2), weight_decay=args.l2), lr=args.lr_mix if args.lr_mix>0.0 else args.lr, warmup_steps=args.warmup)

trainer = TrainLoop(model_la, model_pa, model_mix, optimizer_la, optimizer_pa, optimizer_mix, train_loader, valid_loader, train_mode=args.train_mode, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda, logger=writer)

print('Cuda Mode: {}'.format(args.cuda))
print('Device: {}'.format(device))
print('Train mode: {}'.format(args.train_mode))
print('Selected models (LA, PA, MIX): {}, {}, {}'.format(args.model_la, args.model_pa, args.model_mix))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('LRs LA, PA, and MIX: {}, {}, {}'.format(args.lr_la, args.lr_pa, args.lr_mix))
print('B1 and B2: {}, {}'.format(args.b1, args.b2))
print('l2: {}'.format(args.l2))
print('Warmup iterations: {}'.format(args.smoothing))
print('l2: {}'.format(args.l2))
print('Label smoothing: {}'.format(args.smoothing))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
