import numpy as np
import matplotlib.pyplot as plt
import random

T = 5
K = 10

r = lambda: random.randint(0, 255)

for i in range(T):
	x = np.random.uniform(-5, 5, K)
	A = np.random.uniform(1, 3)
	ph = np.random.uniform(np.pi/8, 5*np.pi/8)
	y = A*np.sin(ph*x)
	c = '#%02X%02X%02X' % (r(),r(),r())
	plt.plot(x, y, 'rx')
	x = np.linspace(-5, 5, 100)
	y = A*np.sin(ph*x)
	plt.plot(np.sort(x), y[x.argsort()], c, label='Task {}'.format(i+1))

plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('tasks.png')
