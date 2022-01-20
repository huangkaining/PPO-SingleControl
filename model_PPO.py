"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class ActorNN(nn.Module):
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
		super(ActorNN, self).__init__()
		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.mean_layer = nn.Linear(64, out_dim)
		self.std_layer = nn.Linear(64, out_dim)
		self.scale = torch.tensor(1.)
		torch.nn.init.orthogonal_(self.layer1.weight)
		torch.nn.init.orthogonal_(self.layer2.weight)
		torch.nn.init.orthogonal_(self.mean_layer.weight)
		torch.nn.init.orthogonal_(self.std_layer.weight)




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

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		mean = self.mean_layer(activation2)
		log_std = self.std_layer(activation2)
		return mean,log_std

	def sample(self,obs):
		mean, log_std = self.forward(obs)
		#std = log_std.exp()
		std = torch.full(size=(8,),fill_value=0.5)
		normal = Normal(mean, std)
		action_origin = normal.rsample()
		action =  torch.tanh(action_origin)
		log_prob = normal.log_prob(action_origin)
		log_prob -= torch.log(self.scale*(torch.tensor(1) - action.pow(2)) + torch.tensor(1e-8))
		log_prob = log_prob.sum()
		return action, log_prob



class CriticNN(nn.Module):
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
		super(CriticNN, self).__init__()
		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)



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

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output


