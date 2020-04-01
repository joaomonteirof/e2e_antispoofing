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
parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'wideresnet', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_CC', 'TDNN', 'FTDNN', 'Linear', 'mobilenet', 'densenet', 'all'], default='resnet', help='Model arch')
parser.add_argument('--resnet-type', choices=['18', '28', '34', '50', '101', 'se_18', 'se_28', 'se_34', 'se_50', 'se_101'], default='18', help='Resnet arch')
args = parser.parse_args()

if args.model == 'lstm' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.cnn_lstm()
	mu = model.forward(batch)
	print('lstm', mu.size())
if args.model == 'resnet' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.ResNet(resnet_type=args.resnet_type)
	mu = model.forward(batch)
	print('resnet', mu.size())
if args.model == 'resnet_pca' or args.model == 'all':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.ResNet_pca(resnet_type=args.resnet_type)
	mu = model.forward(batch)
	print('resnet_pca', mu.size())
if args.model == 'wideresnet' or args.model == 'all':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.WideResNet()
	mu = model.forward(batch)
	print('wideresnet', mu.size())
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
	model = model_.ResNet_CC(resnet_type=args.resnet_type)
	mu = model.forward(batch)
	print('resnet_CC', mu.size())
if args.model == 'TDNN' or args.model == 'all':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.TDNN()
	mu = model.forward(batch)
	print('TDNN', mu.size())
if args.model == 'FTDNN' or args.model == 'all':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.FTDNN()
	mu = model.forward(batch)
	print('FTDNN', mu.size())
if args.model == 'Linear' or args.model == 'all':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.Linear()
	mu = model.forward(batch)
	print('Linear', mu.size())
if args.model == 'mobilenet' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.MobileNetV3_Small()
	mu = model.forward(batch)
	print('MobileNet', mu.size())
if args.model == 'densenet' or args.model == 'all':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.DenseNet()
	mu = model.forward(batch)
	print('DenseNet', mu.size())
