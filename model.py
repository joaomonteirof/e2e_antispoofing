import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SelfAttention(nn.Module):
	def __init__(self, hidden_size):
		super(SelfAttention, self).__init__()

		#self.output_size = output_size
		self.hidden_size = hidden_size
		self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),requires_grad=True)

		init.kaiming_uniform_(self.att_weights)

	def forward(self, inputs):

		batch_size = inputs.size(0)
		weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

		if inputs.size(0)==1:
			attentions = F.softmax(torch.tanh(weights),dim=1)
			weighted = torch.mul(inputs, attentions.expand_as(inputs))
		else:
			attentions = F.softmax(torch.tanh(weights.squeeze()),dim=1)
			weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

		noise = 1e-5*torch.randn(weighted.size())

		if inputs.is_cuda:
			noise = noise.to(inputs.device)

		avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)

		representations = torch.cat((avg_repr,std_repr),1)

		return representations

class PreActBlock(nn.Module):
	'''Pre-activation version of the BasicBlock.'''
	expansion = 1

	def __init__(self, in_planes, planes, stride, *args, **kwargs):
		super(PreActBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out += shortcut
		return out

class PreActBottleneck(nn.Module):
	'''Pre-activation version of the original Bottleneck module.'''
	expansion = 4

	def __init__(self, in_planes, planes, stride, *args, **kwargs):
		super(PreActBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out += shortcut
		return out

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None,
				 *, reduction=16):
		super(SEBasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes, 1)
		self.bn2 = nn.BatchNorm2d(planes)
		self.se = SELayer(planes, reduction)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.se(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class SEBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None,
				 *, reduction=16):
		super(SEBottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.se = SELayer(planes * 4, reduction)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)
		out = self.se(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class cnn_lstm(nn.Module):
	def __init__(self, n_z=256, nclasses=-1):
		super(cnn_lstm, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,3), bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=(5,5), padding=(1,2), dilation=(1,2), stride=(2,2), bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=(5,5), padding=(1,2), dilation=(1,1), stride=(2, 1), bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU() )

		self.conv_fin = nn.Conv2d(256, 256, kernel_size=(15,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn_fin = nn.BatchNorm2d(256)

		self.lstm = nn.LSTM(256, 512, 2, bidirectional=True, batch_first=False)

		self.fc_mu = nn.Linear(512*2, nclasses) if nclasses>2 else nn.Linear(512*2, 1)

		self.initialize_params()

	def forward(self, x):

		x = self.features(x)
		x = self.conv_fin(x)
		feats = F.relu(self.bn_fin(x)).squeeze(2)
		feats = feats.permute(2,0,1)
		batch_size = feats.size(1)
		seq_size = feats.size(0)

		h0 = torch.zeros(2*2, batch_size, 512)
		c0 = torch.zeros(2*2, batch_size, 512)

		if x.is_cuda:
			h0 = h0.to(x.device)
			c0 = c0.to(x.device)

		out_seq, h_c = self.lstm(feats, (h0, c0))

		out_end = out_seq.mean(0)

		mu = self.fc_mu(out_end)

		return mu

	def initialize_params(self):
		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight)
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

RESNET_CONFIGS = {'18':[[2,2,2,2], PreActBlock],
					'28':[[3,4,6,3], PreActBlock],
					'34':[[3,4,6,3], PreActBlock],
					'50':[[3,4,6,3], PreActBottleneck],
					'101':[[3,4,23,3], PreActBottleneck],
					'se_18':[[2,2,2,2], SEBasicBlock],
					'se_28':[[3,4,6,3], SEBasicBlock],
					'se_34':[[3,4,6,3], SEBasicBlock],
					'se_50':[[3,4,6,3], SEBottleneck],
					'se_101':[[3,4,23,3], SEBottleneck]}

class ResNet(nn.Module):
	def __init__(self, resnet_type='18', nclasses=-1):
		self.in_planes = 16
		super(ResNet, self).__init__()

		layers, block = RESNET_CONFIGS[resnet_type]
	
		self._norm_layer = nn.BatchNorm2d

		self.conv1 = nn.Conv2d(1, 16, kernel_size=(9,3), stride=(3,1), padding=(1,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.conv5 = nn.Conv2d(512*block.expansion, 256, kernel_size=(11,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn5 = nn.BatchNorm2d(256)

		self.fc = nn.Linear(256*2,256)
		self.lbn = nn.BatchNorm1d(256)

		self.fc_mu = nn.Linear(256, nclasses) if nclasses>2 else nn.Linear(256, 1)

		self.initialize_params()

		self.attention = SelfAttention(256)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride=1):
		norm_layer = self._norm_layer
		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:
			downsample = nn.Sequential( conv1x1(self.in_planes, planes * block.expansion, stride), norm_layer(planes * block.expansion) )
		layers = []
		layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
		self.in_planes = planes * block.expansion
		for _ in range(1, num_blocks):
			layers.append(block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.conv5(x)
		x = self.activation(self.bn5(x)).squeeze(2)
		
		stats = self.attention(x.permute(0,2,1).contiguous())
		fc = F.relu(self.lbn(self.fc(stats)))
		mu = self.fc_mu(fc)

		#embs = torch.div(mu, torch.norm(mu, 2, 1).unsqueeze(1).expand_as(mu))

		return mu

class ResNet_pca(nn.Module):
	def __init__(self, resnet_type='18', nclasses=-1):
		self.in_planes = 16
		super(ResNet_pca, self).__init__()

		layers, block = RESNET_CONFIGS[resnet_type]

		self._norm_layer = nn.BatchNorm2d
	
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(9,3), stride=(3,1), padding=(1,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.conv5 = nn.Conv2d(512*block.expansion, 256, kernel_size=(5,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn5 = nn.BatchNorm2d(256)

		self.fc = nn.Linear(256*2,256)
		self.lbn = nn.BatchNorm1d(256)

		self.fc_mu = nn.Linear(256, nclasses) if nclasses>2 else nn.Linear(256, 1)

		self.initialize_params()

		self.attention = SelfAttention(256)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride=1):
		norm_layer = self._norm_layer
		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:
			downsample = nn.Sequential( conv1x1(self.in_planes, planes * block.expansion, stride), norm_layer(planes * block.expansion) )
		layers = []
		layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
		self.in_planes = planes * block.expansion
		for _ in range(1, num_blocks):
			layers.append(block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):
	
		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.conv5(x)

		x = self.activation(self.bn5(x)).squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.elu(self.lbn(self.fc(stats)))
		mu = self.fc_mu(fc)

		#embs = torch.div(mu, torch.norm(mu, 2, 1).unsqueeze(1).expand_as(mu))

		return mu

class ResNet_CC(nn.Module):
	def __init__(self, n_z=256, resnet_type='18', nclasses=-1, ncoef=90, init_coef=0):
		self.in_planes = 16
		super(ResNet_CC, self).__init__()

		layers, block = RESNET_CONFIGS[resnet_type]

		self._norm_layer = nn.BatchNorm2d

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.conv1 = nn.Conv2d(1, 16, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()
		
		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.fc_1 = nn.Linear(block.expansion*512*2,256)
		self.lbn = nn.BatchNorm1d(256)

		self.fc_2 = nn.Linear(256, nclasses) if nclasses>2 else nn.Linear(256, 1)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*512)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride=1):
		norm_layer = self._norm_layer
		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:
			downsample = nn.Sequential( conv1x1(self.in_planes, planes * block.expansion, stride), norm_layer(planes * block.expansion) )
		layers = []
		layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
		self.in_planes = planes * block.expansion
		for _ in range(1, num_blocks):
			layers.append(block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):

		x = x[:,:,self.init_coef:,:]

		x = self.conv1(x)
		x = self.activation(self.bn1(x))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())

		fc = F.relu(self.lbn(self.fc_1(stats)))

		out = self.fc_2(fc)

		return out

class mfm(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
		super(mfm, self).__init__()
		self.out_channels = out_channels
		if type == 1:
			self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		else:
			self.filter = nn.Linear(in_channels, 2*out_channels)

	def forward(self, x):
		x = self.filter(x)
		out = torch.split(x, self.out_channels, 1)
		return torch.max(out[0], out[1])

class group(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(group, self).__init__()
		self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
		self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

	def forward(self, x):
		x = self.conv_a(x)
		x = self.conv(x)
		return x

class resblock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(resblock, self).__init__()
		self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		res = x
		out = self.conv1(x)
		out = self.conv2(out)
		out = out + res
		return out

class lcnn_9layers(nn.Module):
	def __init__(self, nclasses=-1):
		super(lcnn_9layers, self).__init__()
		self.features = nn.Sequential(
			mfm(1, 48, 5, 1, 2), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(48, 96, 3, 1, 1), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			group(96, 192, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(192, 128, 3, 1, 1),
			group(128, 128, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) )

		self.conv_final = nn.Conv2d(128, 128, kernel_size=(17,3), stride=(1,1), padding=(0,1), bias=False)
		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = mfm(128, 128, type=0)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)

	def forward(self, x):
		x = self.features(x)

		x = F.relu(self.conv_final(x)).squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		x = self.fc1(x)
		x = F.dropout(x, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_9layers_CC(nn.Module):
	def __init__(self, nclasses=-1, ncoef=90, init_coef=0):
		super(lcnn_9layers_CC, self).__init__()

		self.conv1 = nn.Conv2d(1, 16, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.features = nn.Sequential(
			mfm(16, 48, 5, 1, 2), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(48, 96, 3, 1, 1), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			group(96, 192, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(192, 128, 3, 1, 1),
			group(128, 128, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) )

		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = mfm(128, 128, type=0)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)

	def forward(self, x):

		x = x[:,:,self.init_coef:,:]

		x = self.conv1(x)
		x = self.activation(self.bn1(x))

		x = self.features(x).squeeze(2)

		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		x = self.fc1(x)
		x = F.dropout(x, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_29layers_CC(nn.Module):
	def __init__(self, block=resblock, layers=[1, 2, 3, 4], nclasses=-1, ncoef=90, init_coef=0):
		super(lcnn_29layers_CC, self).__init__()

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.conv1_ = nn.Conv2d(1, 32, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.activation = nn.ReLU()

		self.conv1 = mfm(32, 48, 5, 1, 2)
		self.block1 = self._make_layer(block, layers[0], 48, 48)
		self.group1 = group(48, 96, 3, 1, 1)
		self.block2 = self._make_layer(block, layers[1], 96, 96)
		self.group2 = group(96, 192, 3, 1, 1)
		self.block3 = self._make_layer(block, layers[2], 192, 192)
		self.group3 = group(192, 128, 3, 1, 1)
		self.block4 = self._make_layer(block, layers[3], 128, 128)
		self.group4 = group(128, 128, 3, 1, 1)

		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = nn.Linear(128, 128)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)
			
	def _make_layer(self, block, num_blocks, in_channels, out_channels):
		layers = []
		for i in range(0, num_blocks):
			layers.append(block(in_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):

		x = x[:,:,self.init_coef:,:]

		x = self.conv1_(x)
		x = self.activation(self.bn1(x))

		x = self.conv1(x)

		x = F.max_pool2d(x, 2, ceil_mode=True) + F.avg_pool2d(x, 2, ceil_mode=True)

		x = self.block1(x)
		x = self.group1(x)
		x = F.max_pool2d(x, 2, ceil_mode=True) + F.avg_pool2d(x, 2, ceil_mode=True)
		x = self.block2(x)
		x = self.group2(x)
		x = F.max_pool2d(x, 2, ceil_mode=True) + F.avg_pool2d(x, 2, ceil_mode=True)
		x = self.block3(x)
		x = self.group3(x)
		x = self.block4(x)
		x = self.group4(x)
		x = F.max_pool2d(x, 2, ceil_mode=True) + F.avg_pool2d(x, 2, ceil_mode=True)
		x = x.squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		fc = self.fc1(x)
		x = F.dropout(fc, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_29layers_v2(nn.Module):
	def __init__(self, block=resblock, layers=[1, 2, 3, 4], nclasses=-1):
		super(lcnn_29layers_v2, self).__init__()
		self.conv1 = mfm(1, 48, 5, 1, 2)
		self.block1 = self._make_layer(block, layers[0], 48, 48)
		self.group1 = group(48, 96, 3, 1, 1)
		self.block2 = self._make_layer(block, layers[1], 96, 96)
		self.group2 = group(96, 192, 3, 1, 1)
		self.block3 = self._make_layer(block, layers[2], 192, 192)
		self.group3 = group(192, 128, 3, 1, 1)
		self.block4 = self._make_layer(block, layers[3], 128, 128)
		self.group4 = group(128, 128, 3, 1, 1)

		self.conv_final = nn.Conv2d(128, 128, kernel_size=(16,3), stride=(1,1), padding=(0,1), bias=False)
		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = nn.Linear(128, 128)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)
			
	def _make_layer(self, block, num_blocks, in_channels, out_channels):
		layers = []
		for i in range(0, num_blocks):
			layers.append(block(in_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = self.block1(x)
		x = self.group1(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = self.block2(x)
		x = self.group2(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = self.block3(x)
		x = self.group3(x)
		x = self.block4(x)
		x = self.group4(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = F.relu(self.conv_final(x)).squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		fc = self.fc1(x)
		x = F.dropout(fc, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_9layers_pca(nn.Module):
	def __init__(self, nclasses=-1):
		super(lcnn_9layers_pca, self).__init__()
		self.features = nn.Sequential(
			mfm(1, 48, 5, 1, 2), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(48, 96, 3, 1, 1), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			group(96, 192, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(192, 128, 3, 1, 1),
			group(128, 128, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) )

		self.conv_final = nn.Conv2d(128, 128, kernel_size=(8,3), stride=(1,1), padding=(0,1), bias=False)
		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = mfm(128, 128, type=0)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)

	def forward(self, x):
		x = self.features(x)

		x = F.relu(self.conv_final(x)).squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		x = self.fc1(x)
		x = F.dropout(x, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_9layers_prodspec(nn.Module):
	def __init__(self, nclasses=-1):
		super(lcnn_9layers_prodspec, self).__init__()
		self.features = nn.Sequential(
			mfm(1, 48, 5, 1, 2), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(48, 96, 3, 1, 1), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			group(96, 192, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(192, 128, 3, 1, 1),
			group(128, 128, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) )

		self.conv_final = nn.Conv2d(128, 128, kernel_size=(17,3), stride=(1,1), padding=(0,1), bias=False)
		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = mfm(128, 128, type=0)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)

	def forward(self, x):
		x = self.features(x)

		x = F.relu(self.conv_final(x)).squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		x = self.fc1(x)
		x = F.dropout(x, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_9layers_icqspec(nn.Module):
	def __init__(self, nclasses=-1):
		super(lcnn_9layers_icqspec, self).__init__()
		self.features = nn.Sequential(
			mfm(1, 48, 5, 1, 2), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(48, 96, 3, 1, 1), 
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			group(96, 192, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
			group(192, 128, 3, 1, 1),
			group(128, 128, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) )

		self.conv_final = nn.Conv2d(128, 128, kernel_size=(16,3), stride=(1,1), padding=(0,1), bias=False)
		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = mfm(128, 128, type=0)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)

	def forward(self, x):
		x = self.features(x)

		x = F.relu(self.conv_final(x)).squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		x = self.fc1(x)
		x = F.dropout(x, training=self.training)
		out = self.fc2(x)
		return out

class lcnn_29layers_v2_pca(nn.Module):
	def __init__(self, block=resblock, layers=[1, 2, 3, 4], nclasses=-1):
		super(lcnn_29layers_v2_pca, self).__init__()
		self.conv1 = mfm(1, 48, 5, 1, 2)
		self.block1 = self._make_layer(block, layers[0], 48, 48)
		self.group1 = group(48, 96, 3, 1, 1)
		self.block2 = self._make_layer(block, layers[1], 96, 96)
		self.group2 = group(96, 192, 3, 1, 1)
		self.block3 = self._make_layer(block, layers[2], 192, 192)
		self.group3 = group(192, 128, 3, 1, 1)
		self.block4 = self._make_layer(block, layers[3], 128, 128)
		self.group4 = group(128, 128, 3, 1, 1)

		self.conv_final = nn.Conv2d(128, 128, kernel_size=(7,3), stride=(1,1), padding=(0,1), bias=False)
		self.attention = SelfAttention(128)
		self.fc = nn.Linear(2*128,128)

		self.fc1 = nn.Linear(128, 128)
		self.fc2 = nn.Linear(128, nclasses) if nclasses>2 else nn.Linear(128, 1)
			
	def _make_layer(self, block, num_blocks, in_channels, out_channels):
		layers = []
		for i in range(0, num_blocks):
			layers.append(block(in_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = self.block1(x)
		x = self.group1(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = self.block2(x)
		x = self.group2(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = self.block3(x)
		x = self.group3(x)
		x = self.block4(x)
		x = self.group4(x)
		x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

		x = F.relu(self.conv_final(x)).squeeze(2)
		stats = self.attention(x.permute(0,2,1).contiguous())
		x = self.fc(stats)

		fc = self.fc1(x)
		x = F.dropout(fc, training=self.training)
		out = self.fc2(x)
		return out

class StatisticalPooling(nn.Module):

	def forward(self, x):
		# x is 3-D with axis [B, feats, T]
		mu = x.mean(dim=2, keepdim=False)
		std = (x+torch.randn_like(x)*1e-6).std(dim=2, keepdim=False)
		return torch.cat((mu, std), dim=1)

class TDNN(nn.Module):
	def __init__(self, nclasses=-1, ncoef=90, init_coef=0):
		super(TDNN, self).__init__()

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.model = nn.Sequential( nn.Conv1d(ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=2, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=3, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential(nn.Linear(3000, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, nclasses) if nclasses>2 else nn.Linear(512, 1) )

	def forward(self, x):

		x = x[:,:,self.init_coef:,:].squeeze(1)

		x = self.model(x)
		x = self.pooling(x)
		out = self.post_pooling(x)

		return out

class Linear(nn.Module):
	def __init__(self, nclasses=-1, ncoef=90, init_coef=0):
		super(Linear, self).__init__()

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential( nn.Linear(2*(ncoef-init_coef), nclasses) if nclasses>2 else nn.Linear(2*(ncoef-init_coef), 1) )

	def forward(self, x):

		x = x[:,:,self.init_coef:,:].squeeze(1)
		x = self.pooling(x)
		out = self.post_pooling(x)

		return out

class SOrthConv(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_mode='zeros'):
		'''
		Conv1d with a method for stepping towards semi-orthongonality
		http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
		'''
		super(SOrthConv, self).__init__()

		kwargs = {'bias': False}
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode)
		self.reset_parameters()

	def forward(self, x):
		x = self.conv(x)
		return x

	def step_semi_orth(self):
		with torch.no_grad():
			M = self.get_semi_orth_weight(self.conv)
			self.conv.weight.copy_(M)

	def reset_parameters(self):
		# Standard dev of M init values is inverse of sqrt of num cols
		nn.init._no_grad_normal_(self.conv.weight, 0.,
								 self.get_M_shape(self.conv.weight)[1]**-0.5)

	def orth_error(self):
		return self.get_semi_orth_error(self.conv).item()

	@staticmethod
	def get_semi_orth_weight(conv1dlayer):
		# updates conv1 weight M using update rule to make it more semi orthogonal
		# based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
		# includes the tweaks related to slowing the update speed
		# only an implementation of the 'floating scale' case
		with torch.no_grad():
			update_speed = 0.125
			orig_shape = conv1dlayer.weight.shape
			# a conv weight differs slightly from TDNN formulation:
			# Conv weight: (out_filters, in_filters, kernel_width)
			# TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
			# the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
			M = conv1dlayer.weight.reshape(
				orig_shape[0], orig_shape[1]*orig_shape[2]).T
			# M now has shape (in_dim[rows], out_dim[cols])
			mshape = M.shape
			if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
				M = M.T
			P = torch.mm(M, M.T)
			PP = torch.mm(P, P.T)
			trace_P = torch.trace(P)
			trace_PP = torch.trace(PP)
			ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

			# the following is the tweak to avoid divergence (more info in Kaldi)
			assert ratio > 0.99
			if ratio > 1.02:
				update_speed *= 0.5
				if ratio > 1.1:
					update_speed *= 0.5

			scale2 = trace_PP/trace_P
			update = P - (torch.matrix_power(P, 0) * scale2)
			alpha = update_speed / scale2
			update = (-4.0 * alpha) * torch.mm(update, M)
			updated = M + update
			# updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
			# Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
			# Then reshape to (cols, in_filters, kernel_width)
			return updated.reshape(*orig_shape) if mshape[0] > mshape[1] else updated.T.reshape(*orig_shape)

	@staticmethod
	def get_M_shape(conv_weight):
		orig_shape = conv_weight.shape
		return (orig_shape[1]*orig_shape[2], orig_shape[0])

	@staticmethod
	def get_semi_orth_error(conv1dlayer):
		with torch.no_grad():
			orig_shape = conv1dlayer.weight.shape
			M = conv1dlayer.weight.reshape(
				orig_shape[0], orig_shape[1]*orig_shape[2]).T
			mshape = M.shape
			if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
				M = M.T
			P = torch.mm(M, M.T)
			PP = torch.mm(P, P.T)
			trace_P = torch.trace(P)
			trace_PP = torch.trace(PP)
			scale2 = torch.sqrt(trace_PP/trace_P) ** 2
			update = P - (torch.matrix_power(P, 0) * scale2)
			return torch.norm(update, p='fro')


class SharedDimScaleDropout(nn.Module):
	def __init__(self, alpha: float = 0.5, dim=1):
		'''
		Continuous scaled dropout that is const over chosen dim (usually across time)
		Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
		'''
		super(SharedDimScaleDropout, self).__init__()
		if alpha > 0.5 or alpha < 0:
			raise ValueError("alpha must be between 0 and 0.5")
		self.alpha = alpha
		self.dim = dim
		self.register_buffer('mask', torch.tensor(0.))

	def forward(self, X):
		if self.training:
			if self.alpha != 0.:
				# sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
				tied_mask_shape = list(X.shape)
				tied_mask_shape[self.dim] = 1
				repeats = [1 if i != self.dim else X.shape[self.dim]
						   for i in range(len(X.shape))]
				return X * self.mask.repeat(tied_mask_shape).uniform_(1 - 2*self.alpha, 1 + 2*self.alpha).repeat(repeats)
				# expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
		return X


class FTDNNLayer(nn.Module):

	def __init__(self, in_dim, out_dim, bottleneck_dim, context_size=2, dilations=None, paddings=None, alpha=0.0):
		'''
		3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
		'''
		super(FTDNNLayer, self).__init__()
		paddings = [1, 1, 1] if not paddings else paddings
		dilations = [2, 2, 2] if not dilations else dilations
		assert len(paddings) == 3
		assert len(dilations) == 3
		self.factor1 = SOrthConv(
			in_dim, bottleneck_dim, context_size, padding=paddings[0], dilation=dilations[0])
		self.factor2 = SOrthConv(bottleneck_dim, bottleneck_dim,
								 context_size, padding=paddings[1], dilation=dilations[1])
		self.factor3 = nn.Conv1d(bottleneck_dim, out_dim, context_size,
								 padding=paddings[2], dilation=dilations[2], bias=False)
		self.nl = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_dim)
		self.dropout = SharedDimScaleDropout(alpha=alpha, dim=1)

	def forward(self, x):
		''' input (batch_size, seq_len, in_dim) '''
		assert (x.shape[-1] == self.factor1.conv.weight.shape[1])
		x = self.factor1(x.transpose(1, 2))
		x = self.factor2(x)
		x = self.factor3(x)
		x = self.nl(x)
		x = self.bn(x).transpose(1, 2)
		x = self.dropout(x)
		return x

	def step_semi_orth(self):
		for layer in self.children():
			if isinstance(layer, SOrthConv):
				layer.step_semi_orth()

	def orth_error(self):
		orth_error = 0
		for layer in self.children():
			if isinstance(layer, SOrthConv):
				orth_error += layer.orth_error()
		return orth_error


class DenseReLU(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(DenseReLU, self).__init__()
		self.fc = nn.Linear(in_dim, out_dim)
		self.bn = nn.BatchNorm1d(out_dim)
		self.nl = nn.ReLU()

	def forward(self, x):
		x = self.fc(x)
		x = self.nl(x)
		if len(x.shape) > 2:
			x = self.bn(x.transpose(1, 2)).transpose(1, 2)
		else:
			x = self.bn(x)
		return x


class StatsPool(nn.Module):

	def __init__(self, floor=1e-10, bessel=False):
		super(StatsPool, self).__init__()
		self.floor = floor
		self.bessel = bessel

	def forward(self, x):
		means = torch.mean(x, dim=1)
		_, t, _ = x.shape
		if self.bessel:
			t = t - 1
		residuals = x - means.unsqueeze(1)
		numerator = torch.sum(residuals**2, dim=1)
		stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
		x = torch.cat([means, stds], dim=1)
		return x


class TDNN_(nn.Module):

	def __init__(
		self,
		input_dim=23,
		output_dim=512,
		context_size=5,
		stride=1,
		dilation=1,
		batch_norm=True,
		dropout_p=0.0,
		padding=0
	):
		super(TDNN_, self).__init__()
		self.context_size = context_size
		self.stride = stride
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.dilation = dilation
		self.dropout_p = dropout_p
		self.padding = padding

		self.kernel = nn.Conv1d(self.input_dim,
								self.output_dim,
								self.context_size,
								stride=self.stride,
								padding=self.padding,
								dilation=self.dilation)

		self.nonlinearity = nn.ReLU()
		self.batch_norm = batch_norm
		if batch_norm:
			self.bn = nn.BatchNorm1d(output_dim)
		self.drop = nn.Dropout(p=self.dropout_p)

	def forward(self, x):
		'''
		input: size (batch, seq_len, input_features)
		outpu: size (batch, new_seq_len, output_features)
		'''

		_, _, d = x.shape
		assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
			self.input_dim, d)

		x = self.kernel(x.transpose(1, 2))
		x = self.nonlinearity(x)
		x = self.drop(x)

		if self.batch_norm:
			x = self.bn(x)
		return x.transpose(1, 2)


class FTDNN(nn.Module):

	def __init__(self, nclasses=-1, ncoef=90, init_coef=0):
		'''
		The FTDNN architecture from
		"State-of-the-art speaker recognition with neural network embeddings in 
		NIST SRE18 and Speakers in the Wild evaluations"
		https://www.sciencedirect.com/science/article/pii/S0885230819302700
		'''
		super(FTDNN, self).__init__()

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.layer01 = TDNN_(input_dim=self.ncoef, output_dim=512, context_size=5, padding=2)
		self.layer02 = FTDNNLayer(512, 1024, 256, context_size=2, dilations=[ 2, 2, 2], paddings=[1, 1, 1])
		self.layer03 = FTDNNLayer(1024, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0])
		self.layer04 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
		self.layer05 = FTDNNLayer(2048, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0])
		self.layer06 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
		self.layer07 = FTDNNLayer(3072, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
		self.layer08 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
		self.layer09 = FTDNNLayer(3072, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0])
		self.layer10 = DenseReLU(1024, 2048)
		self.layer11 = StatsPool()
		self.layer12 = DenseReLU(4096, 512)
		self.out_layer = nn.Linear(512, nclasses) if nclasses>2 else nn.Linear(512, 1)

	def forward(self, x):
		'''
		Input must be (batch_size, seq_len, in_dim)
		'''
		x = x.squeeze(1).transpose(1,-1)
		x = self.layer01(x)
		x_2 = self.layer02(x)
		x_3 = self.layer03(x_2)
		x_4 = self.layer04(x_3)
		skip_5 = torch.cat([x_4, x_3], dim=-1)
		x = self.layer05(skip_5)
		x_6 = self.layer06(x)
		skip_7 = torch.cat([x_6, x_4, x_2], dim=-1)
		x = self.layer07(skip_7)
		x_8 = self.layer08(x)
		skip_9 = torch.cat([x_8, x_6, x_4], dim=-1)
		x = self.layer09(skip_9)
		x = self.layer10(x)
		x = self.layer11(x)
		x = self.layer12(x)
		x = self.out_layer(x)
		return x

	def step_ftdnn_layers(self):
		for layer in self.children():
			if isinstance(layer, FTDNNLayer):
				layer.step_semi_orth()

	def set_dropout_alpha(self, alpha):
		for layer in self.children():
			if isinstance(layer, FTDNNLayer):
				layer.dropout.alpha = alpha

	def get_orth_errors(self):
		errors = 0.
		with torch.no_grad():
			for layer in self.children():
				if isinstance(layer, FTDNNLayer):
					errors += layer.orth_error()
		return errors