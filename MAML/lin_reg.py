import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

col = ['#eb403450', '#ebba3450', '#b1eb3450',
		'#5feb3450', '#34ebb450', '#34a8eb50',
		'#343deb50', '#a234eb50', '#eb34d350', '#d334eb50']

alpha = 0.1
beta = 0.1

def plot_graphs(X, Y, m, c, w_MAML, w_scratch):
	T = X.shape[0]

	fig, axs = plt.subplots(T//2, 2)
	fig.tight_layout(pad=1.0)
	fig.suptitle('Testing on {} tasks'.format(T))
	plt.grid()
	for i in range(T):
		axs[i//2, i%2].plot(X[i], Y[i], 'bx', label='Data points') # data points
		axs[i//2, i%2].plot(X[i], m[i]*X[i]+c[i], 'b--', label='True distribution')
		axs[i//2, i%2].plot(X[i], w_MAML[0, i]*X[i]+w_MAML[1, i], 'r-', label='MAML-trained model')
		axs[i//2, i%2].plot(X[i], w_scratch[0, i]*X[i]+w_scratch[1, i], 'g-', label='Model trained from scratch')
	# plt.legend()
	plt.savefig('testing_error.png')

def gen_points(m, c, K=20, std=0.1):
	x = np.linspace(-5, 5, K)
	y = m*x+c + np.random.normal(0, std, K)

	return x, y

def gen_task_set(m_min, m_max, c_min, c_max, T, K=20):
	X = [] # input data matrix
	Y = [] # label matrix

	m = np.random.uniform(m_min, m_max, T)
	c = np.random.uniform(c_min, c_max, T)

	for i in range(T):
		x, y = gen_points(m[i], c[i], K)
		X.append(x)
		Y.append(y)

	X = np.array(X)
	Y = np.array(Y)

	return X, Y, m, c

def test_model(w, m_min, m_max, c_min, c_max, T=10):
	X, Y, m, c = gen_task_set(m_min, m_max, c_min, c_max, T, K=5) # generating tasks for testing

	w_MAML = w.copy() # training from the point given by MAML
	w_MAML = np.tile(w_MAML.reshape(2, 1), (1, T))
	w_scratch = np.zeros([2, T]) # training from scratch

	epochs = 5

	# training error on the new tasks
	maml_error = np.zeros([epochs, T])
	scratch_error = np.zeros([epochs, T])

	for i in range(epochs):
		Y_ = w_MAML[0].reshape(-1, 1)*X+w_MAML[1].reshape(-1, 1)

		error = 0.5*np.mean((Y_-Y)**2, 1)
		maml_error[i] = error

		grad1 = np.mean((Y_-Y)*X, 1)
		grad2 = np.mean(Y_-Y, 1)

		w_MAML[0] = w_MAML[0] - alpha*grad1
		w_MAML[1] = w_MAML[1] - alpha*grad2

		Y_ = w_scratch[0].reshape(-1, 1)*X+w_scratch[1].reshape(-1, 1)

		error = 0.5*np.mean((Y_-Y)**2, 1)
		scratch_error[i] = error

		grad1 = np.mean((Y_-Y)*X, 1)
		grad2 = np.mean(Y_-Y, 1)

		w_scratch[0] = w_scratch[0] - alpha*grad1
		w_scratch[1] = w_scratch[1] - alpha*grad2

	plot_graphs(X, Y, m, c, w_MAML, w_scratch)

m_min = -4
m_max = 4
c_min = -2
c_max = 2

T = 10 # number of tasks
K = 20

X, Y, m, c = gen_task_set(m_min, m_max, c_min, c_max, T)

print("Size of input data matrix", X.shape)
print("Size of label matrix", Y.shape)

w = np.zeros(2) # weights
w_ = np.zeros([2, T]) # one-step-ahead training weights

epochs = 10

for e in range(epochs):
	plt.plot(w[0], w[1], 'bx')
	Y_ = w[0]*X + w[1] # predictions

	error = 0.5*np.mean((Y_-Y)**2, 1) # error between the true labels and the predictions

	grad1 = np.mean((Y_-Y)*X, 1) # gradient of error w.r.t. w[0]
	grad2 = np.mean(Y_-Y, 1) # gradient of error w.r.t. w[1]

	grad1_ = np.mean(X*X, 1) # derivative of grad1 w.r.t. w[0]
	grad2_ = np.mean(X, 1) # derivative of grad2 w.r.t. w[1]

	w_[0] = w[0]-alpha*grad1
	w_[1] = w[1]-alpha*grad2

	for j in range(T):
		plt.plot([w[0], w_[0, j]], [w[1], w_[1, j]], col[j], linestyle=':')

	Y_ = w_[0].reshape(-1, 1)*X+w_[1].reshape(-1, 1)

	metagrad1 = np.mean(np.mean((Y_-Y)*X*np.tile((1-alpha*grad1_).reshape(-1, 1), (1, K)), 1))
	metagrad2 = np.mean(np.mean((Y_-Y)*np.tile((1-alpha*grad2_).reshape(-1, 1), (1, K)), 1))

	oldw = w.copy()

	w[0] -= beta*metagrad1
	w[1] -= beta*metagrad2

	plt.plot([oldw[0], w[0]], [oldw[1], w[1]], 'r--')

plt.grid()
plt.savefig('weights.png')
plt.clf()
test_model(w, m_min, m_max, c_min, c_max)
