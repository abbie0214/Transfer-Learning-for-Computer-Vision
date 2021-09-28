import torch.nn as nn

class regression_model(nn.Module):
	"""docstring for regression_model"""
	def __init__(self):
		super(regression_model, self).__init__()
		self.fc1 = nn.Linear(1, 40)
		self.fc2 = nn.Linear(40, 40)
		self.out = nn.Linear(40, 1)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.out(x)

		return x
