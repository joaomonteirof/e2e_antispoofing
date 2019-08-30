## Adapted from https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/optimizer.py
import torch

class TransformerOptimizer(object):
	"""A simple wrapper class for learning rate scheduling"""

	def __init__(self, optimizer, lr, k=1, warmup_steps=4000):
		self.optimizer = optimizer
		self.k = k
		self.init_lr = lr
		self.warmup_steps = warmup_steps
		self.step_num = 0

	def zero_grad(self):
		self.optimizer.zero_grad()

	def step(self):
		self._update_lr()
		self.optimizer.step()

	def _update_lr(self):
		self.step_num += 1
		lr = self.k * self.init_lr * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5)))
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def load_state_dict(self, state_dict):
		self.optimizer.load_state_dict(state_dict)

	def state_dict(self):
		return self.optimizer.state_dict()
