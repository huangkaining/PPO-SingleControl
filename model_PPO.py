"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.LReLU = nn.LeakyReLU(0.01)
		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)
		self.reset_parameters()
		self.train()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('leaky_relu')
		nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('leaky_relu'))
		nn.init.xavier_uniform_(self.layer2.weight, gain=nn.init.calculate_gain('leaky_relu'))
		nn.init.xavier_uniform_(self.layer3.weight, gain=nn.init.calculate_gain('leaky_relu'))


	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = self.LReLU(self.layer1(obs))
		activation2 = self.LReLU(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
