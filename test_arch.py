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
parser.add_argument('--model', choices=['lstm', 'resnet', 'resnet_pca', 'lcnn_9', 'lcnn_29', 'lcnn_9_pca', 'lcnn_29_pca', 'lcnn_9_prodspec', 'lcnn_9_icqspec', 'lcnn_9_CC', 'lcnn_29_CC', 'resnet_34_CC'], default='resnet', help='Model arch')
args = parser.parse_args()

if args.model == 'lstm':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.cnn_lstm()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.ResNet()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_pca':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.ResNet_pca()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_9':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.lcnn_9layers()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_29':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.lcnn_29layers_v2()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_9_pca':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.lcnn_9layers_pca()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_29_pca':
	batch = torch.rand(3, 1, 120, 300)
	model = model_.lcnn_29layers_v2_pca()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_9_icqspec':
	batch = torch.rand(3, 1, 256, 300)
	model = model_.lcnn_9layers_icqspec()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_9_prodspec':
	batch = torch.rand(3, 1, 257, 300)
	model = model_.lcnn_9layers_prodspec()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_9_CC':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.lcnn_9layers_CC()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'lcnn_29_CC':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.lcnn_29layers_CC()
	mu = model.forward(batch)
	print(mu.size())
elif args.model == 'resnet_34_CC':
	batch = torch.rand(3, 1, 90, 300)
	model = model_.ResNet_34_CC()
	mu = model.forward(batch)
	print(mu.size())
