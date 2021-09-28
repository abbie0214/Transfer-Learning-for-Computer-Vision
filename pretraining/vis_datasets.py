'''
This code will sample 50 images each from CIFAR/MNIST datasets
and will display them in a grid.
'''

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torchvision import transforms

batchsize = 50

# transforms.Compose([transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist = datasets.MNIST(root='MNIST/mnist_data', train=False, download=True,
						transform=transforms.Compose([transforms.ToTensor()]))
fashionmnist = datasets.FashionMNIST(root='MNIST/fashion_mnist_data', train=False, download=True,
									transform=transforms.Compose([transforms.ToTensor()]))
cifar10 = datasets.CIFAR10(root='CIFAR/cifar10', train=False, download=True, 
							transform=transforms.Compose([transforms.ToTensor()]))
cifar100 = datasets.CIFAR100(root='CIFAR/cifar100', train=False, download=True, 
							transform=transforms.Compose([transforms.ToTensor()]))

test_mnist = torch.utils.data.DataLoader(mnist, batch_size=batchsize, shuffle=True, num_workers=0)
test_fashionmnist = torch.utils.data.DataLoader(fashionmnist, batch_size=batchsize, shuffle=True, num_workers=0)
test_cifar10 = torch.utils.data.DataLoader(cifar10, batch_size=batchsize, shuffle=True, num_workers=0)
test_cifar100 = torch.utils.data.DataLoader(cifar100, batch_size=batchsize, shuffle=True, num_workers=0)

mnist_data = next(iter(test_mnist))[0].numpy()
fashionmnist_data = next(iter(test_fashionmnist))[0].numpy()
cifar10_data = next(iter(test_cifar10))[0].numpy()
cifar100_data = next(iter(test_cifar100))[0].numpy()

pos = np.random.choice(100, 50, replace=False)

mnist_grid = np.zeros([28*10, 28*10])
cifar_grid = np.zeros([32*10, 32*10, 3])

c = 0

for p in pos:
	# print("mnist", 28*(p//10), 28*(p//10 + 1), 28*(p%10), 28*(p%10 + 1))
	# print("cifar", 32*(p//10), 32*(p//10 + 1), 32*(p%10), 32*(p%10 + 1))
	mnist_grid[28*(p//10):28*(p//10 + 1), 28*(p%10):28*(p%10 + 1)] = mnist_data[c, 0]
	cifar_grid[32*(p//10):32*(p//10 + 1), 32*(p%10):32*(p%10 + 1)] = np.transpose(cifar10_data[c], [1, 2, 0])
	c += 1

pos_ = [_ for _ in range(100) if _ not in pos]

c = 0

for p in pos_:
	mnist_grid[28*(p//10):28*(p//10 + 1), 28*(p%10):28*(p%10 + 1)] = fashionmnist_data[c, 0]
	cifar_grid[32*(p//10):32*(p//10 + 1), 32*(p%10):32*(p%10 + 1)] = np.transpose(cifar100_data[c], [1, 2, 0])
	c += 1

plt.imshow(mnist_grid, 'gray')
plt.title('A mix of MNIST and Fashion-MNIST data')
plt.axis('off')
plt.savefig('mnist_grid.png')
plt.clf()

plt.imshow(cifar_grid)
plt.title('A mix of CIFAR-10 and CIFAR-100 data')
plt.axis('off')
plt.savefig('cifar_grid.png')
plt.clf()
