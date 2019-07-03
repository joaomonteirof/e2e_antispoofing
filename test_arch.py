from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import model as model_
from data_load import Loader

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'all'], default='resnet', help='Model arch')
args = parser.parse_args()

if args.model == 'lstm' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.cnn_lstm()
	mu = model.forward(batch)
	print('lstm', mu.size())
if args.model == 'resnet' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.ResNet()
	mu = model.forward(batch)
	print('resnet', mu.size())
if args.model == 'resnet_pca' or args.model == 'all':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.ResNet_pca()
	mu = model.forward(batch)
	print('resnet_pca', mu.size())
if args.model == 'lcnn_9' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.lcnn_9layers()
	mu = model.forward(batch)
	print('lcnn_9', mu.size())
if args.model == 'lcnn_29' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.lcnn_29layers_v2()
	mu = model.forward(batch)
	print('lcnn_29', mu.size())
if args.model == 'lcnn_9_pca' or args.model == 'all':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.lcnn_9layers_pca()
	mu = model.forward(batch)
	print('lcnn_9_pca', mu.size())
if args.model == 'lcnn_29_pca' or args.model == 'all':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.lcnn_29layers_v2_pca()
	mu = model.forward(batch)
	print('lcnn_29_pca', mu.size())
if args.model == 'lcnn_9_icqspec' or args.model == 'all':
	batch = torch.rand(3, 1, 256, 300)
	model = model_.lcnn_9layers_icqspec()
	mu = model.forward(batch)
	print('lcnn_9_icqspec', mu.size())
if args.model == 'lcnn_9_prodspec' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.lcnn_9layers_prodspec()
	mu = model.forward(batch)
	print('lcnn_9_prodspec', mu.size())
if args.model == 'lcnn_9_CC' or args.model == 'all':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.lcnn_9layers_CC()
	mu = model.forward(batch)
	print('lcnn_9_CC', mu.size())
if args.model == 'lcnn_29_CC' or args.model == 'all':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.lcnn_29layers_CC()
	mu = model.forward(batch)
	print('lcnn_29_CC', mu.size())
if args.model == 'resnet_CC' or args.model == 'all':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.ResNet_CC()
	mu = model.forward(batch)
	print('resnet_CC', mu.size())
