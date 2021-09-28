import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import random

import torch
import torch.nn as nn

from model import regression_model

col = ['#eb403450', '#ebba3450', '#b1eb3450',
		'#5feb3450', '#34ebb450', '#34a8eb50',
		'#343deb50', '#a234eb50', '#eb34d350', '#d334eb50']

r = lambda: random.randint(0, 255)

alpha = 0.1 # learning rate for meta learner
beta = 0.1 # learning rate for learning individual task

# in this work, we use the same optimizer with learning rate = alpha

def gen_points(K=10):
	# range of input
	x_min = -5
	x_max = 5

	# range of amplitude
	A_min = 1.
	A_max = 3.

	# range of phase
	ph_min = np.pi/8.
	ph_max = 5*np.pi/8.

	# randomly sample from input space
	x = np.random.uniform(x_min, x_max, K)
	A = np.random.uniform(A_min, A_max) # sample a value for amplitude
	ph = np.random.uniform(ph_min, ph_max) # sample a value for phase
	y = A*np.sin(ph*x) # calculating y

	return x, y, A, ph

def gen_tasks(T):
	X = []
	Y = []
	A = []
	Ph = []

	# generate T number of tasks
	for t in range(T):
		x, y, a, ph = gen_points(K=20)
		X.append(x)
		Y.append(y)
		A.append(a)
		Ph.append(ph)

	X = np.array(X)
	Y = np.array(Y)

	return X, Y, A, Ph

def plot_graphs(error, fname='training_error'):
	# plotting the training error for different tasks
	(E, T) = error.shape
	for i in range(T):
		plt.plot(range(E), error[..., i], '#%02X%02X%02X' % (r(),r(),r()))

	plt.grid()
	plt.ylabel("Error")
	plt.xlabel("Training steps")
	plt.title("Training error vs Training steps")
	plt.savefig('{}.png'.format(fname))
	plt.clf()

def test_model(MAML_model, t):
	X, Y, A, ph = gen_tasks(T=1) # one task at a time

	x = np.linspace(-5, 5, 100)

	# plot the true sinusoid
	plt.plot(X.reshape(-1), Y.reshape(-1), 'rx')
	plt.plot(x, A[0]*np.sin(ph[0]*x), 'r')

	scratch_model = regression_model()

	epochs = 5

	X = torch.from_numpy(X[0].astype(np.float32).reshape(-1, 1)).to(device)
	Y = torch.from_numpy(Y[0].astype(np.float32).reshape(-1, 1)).to(device)
	
	loss = nn.MSELoss()
	
	# optimizers for training MAML-initialized model and a model from scratch
	maml_optim = torch.optim.Adam(MAML_model.parameters(), lr=beta)
	scratch_optim = torch.optim.Adam(scratch_model.parameters(), lr=beta)

	maml_error = np.zeros(epochs)
	scratch_error = np.zeros(epochs)

	for e in range(epochs):
		y_ = MAML_model(X)

		# plot the default output of MAML-initialized model
		if e == 0:
			x = X.detach().cpu().numpy().reshape(-1)
			y = y_.detach().cpu().numpy().reshape(-1)
			y = y[x.argsort()]
			x = np.sort(x)
			plt.plot(x, y, 'g:', label='MAML start')
		error = loss(y_, Y)
		error_val = error.detach().cpu().numpy()
		maml_error[e] = error_val
		print("MAML: Epoch {} Error {}".format(e, error_val))
		error.backward()
		maml_optim.step()

		y_ = scratch_model(X)

		# plot the default output of randomly initialized model
		if e == 0:
			x = X.detach().cpu().numpy().reshape(-1)
			y = y_.detach().cpu().numpy().reshape(-1)
			y = y[x.argsort()]
			x = np.sort(x)
			plt.plot(x, y, 'b:', label='Scratch start')
		error = loss(y_, Y)
		error_val = error.detach().cpu().numpy()
		scratch_error[e] = error_val
		print("scratch: Epoch {} Error {}".format(e, error_val))
		error.backward()
		scratch_optim.step()

	# plot the final output of MAML-initialized model
	y_ = MAML_model(X)
	x = X.detach().cpu().numpy().reshape(-1)
	y = y_.detach().cpu().numpy().reshape(-1)
	y = y[x.argsort()]
	x = np.sort(x)
	plt.plot(x, y, 'g-', label='MAML end')

	# plot the final output of MAML-initialized model
	y_ = scratch_model(X)
	x = X.detach().cpu().numpy().reshape(-1)
	y = y_.detach().cpu().numpy().reshape(-1)
	y = y[x.argsort()]
	x = np.sort(x)
	plt.plot(x, y, 'b-', label='Scratch end')

	plt.grid()
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.legend()
	plt.savefig('regression_{}.png'.format(t))
	plt.clf()

	plot_graphs(maml_error.reshape(-1, 1), 'maml_testing_error_{}'.format(t))
	plot_graphs(scratch_error.reshape(-1, 1), 'scratch_testing_error_{}'.format(t))

def main():
	T = 10 # number of tasks from the distribution of tasks

	# generating data points
	X, Y, A, ph = gen_tasks(T)

	print("Input data matrix shape", X.shape)
	print("Label data matrix shape", Y.shape)

	# model
	model = regression_model()

	loss = nn.MSELoss()
	optim = torch.optim.Adam(model.parameters(), lr=alpha)

	epochs = 10

	error_plot = np.zeros([epochs, T])

	for e in range(epochs):
		# at beginning of every epoch, store the weights of the model
		# we will use these weights after training for one task
		fc1_w = model.fc1.weight.data
		fc1_b = model.fc1.bias.data
		fc2_w = model.fc2.weight.data
		fc2_b = model.fc2.bias.data
		out_w = model.out.weight.data
		out_b = model.out.bias.data

		# initialize variables to hold the gradients from each task
		grad_fc1_w = torch.zeros_like(fc1_w)
		grad_fc1_b = torch.zeros_like(fc1_b)
		grad_fc2_w = torch.zeros_like(fc2_w)
		grad_fc2_b = torch.zeros_like(fc2_b)
		grad_out_w = torch.zeros_like(out_w)
		grad_out_b = torch.zeros_like(out_b)
		
		for t in range(T):
			optim.zero_grad()
			X_in = torch.from_numpy(X[t].astype(np.float32).reshape(-1, 1)).to(device)
			Y_true = torch.from_numpy(Y[t].astype(np.float32).reshape(-1, 1)).to(device)
			
			# run the model with parameters 'theta'
			y_ = model(X_in)
			error = loss(y_, Y_true)

			error_val = error.detach().cpu().numpy()
			error_plot[e, t] = error_val
			print("Epoch {} Task {} Error {}".format(e, t, error_val))
			
			error.backward()
			optim.step() # update weights using these gradients
			optim.zero_grad()

			# run the model on the same task with updated weights
			y_ = model(X_in)
			error = loss(y_, Y_true)
			error.backward() # calculate the gradients

			# store these gradients
			grad_fc1_w += model.fc1.weight.grad
			grad_fc2_w += model.fc2.weight.grad
			grad_out_w += model.out.weight.grad
			grad_fc1_b += model.fc1.bias.grad
			grad_fc2_b += model.fc2.bias.grad
			grad_out_b += model.out.bias.grad

			# restore the initial weights from the beginning of the epoch
			# and train on the second task
			model.fc1.weight.data = fc1_w
			model.fc1.bias.data = fc1_b
			model.fc2.weight.data = fc2_w
			model.fc2.bias.data = fc2_b
			model.out.weight.data = out_w
			model.out.bias.data = out_b

		# average the gradients from different tasks
		grad_fc1_w /= T
		grad_fc2_w /= T
		grad_out_w /= T
		grad_fc1_b /= T
		grad_fc2_b /= T
		grad_out_b /= T

		# set gradient of the weights to these new values
		model.fc1.weight.grad = grad_fc1_w
		model.fc2.weight.grad = grad_fc2_w
		model.out.weight.grad = grad_out_w
		model.fc1.bias.grad = grad_fc1_b
		model.fc2.bias.grad = grad_fc2_b
		model.out.bias.grad = grad_out_b

		optim.step()
		optim.zero_grad()

	# plot the training error for different tasks
	plot_graphs(error_plot)

	# test it for different tasks
	test_tasks = 5
	for t in range(test_tasks):
		# test the model for different tasks from the same distribution
		test_model(model, t)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	main()
