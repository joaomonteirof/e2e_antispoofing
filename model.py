import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

	def __init__(self, in_planes, planes, stride=1):
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

class ResNet(nn.Module):
	def __init__(self, layers=[2,2,2,2], block=PreActBlock, nclasses=-1):
		self.in_planes = 16
		super(ResNet, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(9,3), stride=(3,1), padding=(1,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()
		
		self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

		self.conv5 = nn.Conv2d(256, 256, kernel_size=(11,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn5 = nn.BatchNorm2d(256)

		self.fc = nn.Linear(block.expansion*256*2,256)
		self.lbn = nn.BatchNorm1d(256)

		self.fc_mu = nn.Linear(256, nclasses) if nclasses>2 else nn.Linear(256, 1)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*256)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
	def __init__(self, layers=[2,2,2,2], block=PreActBlock, nclasses=-1):
		self.in_planes = 16
		super(ResNet_pca, self).__init__()
	
		self.conv1 = nn.Conv2d(1, 16, kernel_size=(9,3), stride=(3,1), padding=(1,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()
		
		self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

		self.conv5 = nn.Conv2d(256, 256, kernel_size=(5,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn5 = nn.BatchNorm2d(256)

		self.fc = nn.Linear(block.expansion*256*2,256)
		self.lbn = nn.BatchNorm1d(256)

		self.fc_mu = nn.Linear(256, nclasses) if nclasses>2 else nn.Linear(256, 1)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*256)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
	def __init__(self, n_z=256, layers=[2,2,2,2], block=PreActBlock, nclasses=-1, ncoef=90, init_coef=0):
		self.in_planes = 16
		super(ResNet_CC, self).__init__()

		self.ncoef=ncoef
		self.init_coef=init_coef

		self.conv1 = nn.Conv2d(1, 16, kernel_size=(ncoef,3), stride=(1,1), padding=(0,1), bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.activation = nn.ReLU()
		
		self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

		self.fc_1 = nn.Linear(block.expansion*256*2,512)
		self.lbn = nn.BatchNorm1d(512)

		self.fc_2 = nn.Linear(512, nclasses) if nclasses>2 else nn.Linear(512, 1)

		self.initialize_params()

		self.attention = SelfAttention(block.expansion*256)

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
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
