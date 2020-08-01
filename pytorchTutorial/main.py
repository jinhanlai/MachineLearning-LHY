# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2020/7/21 18:46
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(446)
np.random.seed(446)
"""
    Reference:https://colab.research.google.com/drive/1Xed5YSpLsLfkn66OhhyNzr05VE89enng#scrollTo=0d77LgKaMTih
"""

"""
    tensor is similar to numpy's ndarrays
"""
# we create tensors in a similar way to numpy nd arrays
x_numpy = np.array([0.1, 0.2, 0.3])
x_torch = torch.tensor([0.1, 0.2, 0.3])
print('x_numpy, x_torch')
print(x_numpy, x_torch)
print()

# to and from numpy, pytorch
print('to and from numpy and pytorch')
print(torch.from_numpy(x_numpy), x_torch.numpy())
print()

# we can do basic operations like +-*/
y_numpy = np.array([3, 4, 5.])
y_torch = torch.tensor([3, 4, 5.])
print("x+y")
print(x_numpy + y_numpy, x_torch + y_torch)
print()

# many functions that are in numpy are also in pytorch
print("norm")
print(np.linalg.norm(x_numpy), torch.norm(x_torch))
print()

# to apply an operation along a dimension,
# we use the dim keyword argument instead of axis
print("mean along the 0th dimension")
x_numpy = np.array([[1, 2], [3, 4.]])
x_torch = torch.tensor([[1, 2], [3, 4.]])
print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))

"""
    Tensor.View is similarly to numpy.reshape    
"""
N, C, W, H = 10000, 3, 28, 28
X = torch.randn((N, C, W, H))

print(X.shape)
print(X.view(N, C, 784).shape)
print(X.view(-1, C, 784).shape)  # automatically choose the 0th dimension

"""
    Broadcastable semantices,PyTorch operations support NumPy Broadcasting Semantics.
    Requir:
        每个tensor至少有一维
        遍历所有的维度，从尾部维度开始，每个对应的维度大小要么相同，要么其中一个是1，要么其中一个不存在
"""
x = torch.empty(5, 1, 4, 1)
y = torch.empty(   3, 1, 1)
print((x + y).size())  # 5,3,4,1

"""
    Computation graphs
"""
a = torch.tensor(2.0, requires_grad=True)  # we set requires_grad=True to let PyTorch know to keep the graph
b = torch.tensor(1.0, requires_grad=True)
c = a + b
d = b + 1
e = c * d
print('c', c)
print('d', d)
print('e', e)

"""
    CUDA semantics
    It's easy cupy tensor from cpu to gpu or from gpu to cpu.
"""
cpu = torch.device("cpu")
gpu = torch.device("cuda")

x = torch.rand(10)
print(x)
x = x.to(gpu)
print(x)
x = x.to(cpu)
print(x)

"""
    PyTorch as an auto grad framework
    We make a backward() call on the leaf variable (y) in the computation, computing all the gradients of y at once.
    grad :
        requires_grad=True 这个参数表示需要计算梯度，False表示冻结网络
        .data 返回的是新的Tensor对象，返回的对象跟之前对象的id不同，属于不同的Tensor，但共享同一个数据存储空间，一个改变另一个也跟着改变
        requires_grad_() 这个函数表示会改变requires_grad这个属性，并返回Tensor
        detach()函数会返回一个新的Tensor对象，并且新的对象与当前计算图是分离的，requires_grad属性是False，反向传播不会计算梯度；
            新对象与原对象共享数据的存储空间，二者指向同一块内存，只共享数据部分
        torch.no_grad() 一个上下文管理器，用来禁止梯度的计算，通常用来网络推断中，它可以减少计算内存的使用量。
        detach_() 把当前变量从图中分离出来，而detach()是生成了一个新的图
        
        
        
"""


def f(x):
    return (x - 2) ** 2


def fp(x):
    return 2 * (x - 2)


x = torch.tensor([1.0], requires_grad=True)

y = f(x)
y.backward()

print('Analytical f\'(x):', fp(x))
print('PyTorch\'s f\'(x):', x.grad)
"""
    Using the gradients
"""
x = torch.tensor([5.0], requires_grad=True)
step_size = 0.25

print('iter,\tx,\tf(x),\tf\'(x),\tf\'(x) pytorch')
for i in range(15):
    y = f(x)
    y.backward()  # compute the gradient

    print('{},\t{:.3f},\t{:.3f},\t{:.3f},\t{:.3f}'.format(i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))

    x.data = x.data - step_size * x.grad  # perform a GD update step

    # We need to zero the grad variable since the backward()
    # call accumulates the gradients in .grad instead of overwriting.
    # The detach_() is for efficiency. You do not need to worry too much about it.
    x.grad.detach_()
    x.grad.zero_()
"""
Linear Regression
"""
d = 2
n = 50
X = torch.randn(n, d)
true_w = torch.tensor([[-1.0], [2.0]])
y = X @ true_w + torch.randn(n, 1) * 0.1  # @是矩阵乘法， * 是对应元素相乘
print('X shape', X.shape)
print('y shape', y.shape)
print('w shape', true_w.shape)

"""
    torch.nn.Module
        Module is PyTorch's way of performing operations on tensors. Modules are implemented as subclasses of the 
            torch.nn.Module class. All modules are callable and can be composed together to create complex functions.

    torch.nn docs
        Note: most of the functionality implemented for modules can be accessed in a functional form via torch.nn.functional,
            but these require you to create and manage the weight tensors yourself.
    torch.nn.functional docs.
"""

# Linear Module
d_in = 3
d_out = 4
linear_module = nn.Linear(d_in, d_out)

example_tensor = torch.tensor([[1.,2,3], [4,5,6]])
# applys a linear transformation to the data
transformed = linear_module(example_tensor)
print('example_tensor', example_tensor.shape)
print('transormed', transformed.shape)
print()
print('We can see that the weights exist in the background\n')
print('W:', linear_module.weight)
print('b:', linear_module.bias)

"""
    Activation functions
    PyTorch implements a number of activation functions including but not limited to ReLU, Tanh, and Sigmoid. 
        Since they are modules, they need to be instantiated.
"""
activation_fn = nn.ReLU() # we instantiate an instance of the ReLU module
example_tensor = torch.tensor([-1.0, 1.0, 0.0])
activated = activation_fn(example_tensor)
print('example_tensor', example_tensor)
print('activated', activated)

"""
    Sequential
    Many times, we want to compose Modules together. torch.nn.Sequential provides a good interface for composing simple modules
"""
d_in = 3
d_hidden = 4
d_out = 1
model = torch.nn.Sequential(
                            nn.Linear(d_in, d_hidden),
                            nn.Tanh(),
                            nn.Linear(d_hidden, d_out),
                            nn.Sigmoid()
                           )

example_tensor = torch.tensor([[1.,2,3],[4,5,6]])
transformed = model(example_tensor)
print('transformed', transformed.shape)

params = model.parameters() #可以获得模型的所有参数
for param in params:
    print(param)

"""
    Loss functions
    PyTorch implements many common loss functions including MSELoss and CrossEntropyLoss.
"""
mse_loss_fn = nn.MSELoss()
input = torch.tensor([[0., 0, 0]])
target = torch.tensor([[1., 0, -1]])
loss = mse_loss_fn(input, target)
print(loss)

"""
    torch.optim
    PyTorch implements a number of gradient-based optimization methods in torch.optim, including Gradient Descent. 
        At the minimum, it takes in the model parameters and a learning rate.
    Optimizers do not compute the gradients for you, so you must call backward() yourself. You also must call the 
        optim.zero_grad() function before calling backward() since by default PyTorch does and inplace add to the .grad 
        member variable rather than overwriting it.

This does both the detach_() and zero_() calls on all tensor's grad variables.
https://pytorch.org/docs/stable/optim.html

"""
# create a simple model
model = nn.Linear(1, 1)
# create a simple dataset
X_simple = torch.tensor([[1.]])
y_simple = torch.tensor([[2.]])

# create our optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2) # optim必须传入model的参数
mse_loss_fn = nn.MSELoss()

y_hat = model(X_simple)
print('model params before:', model.weight)
loss = mse_loss_fn(y_hat, y_simple)
optim.zero_grad()  #梯度归零，不然会累加之前的梯度
loss.backward()
optim.step() # 会 自动做SGD or Adam
print('model params after:', model.weight)


"""
    Convolutions
    Custom Datasets, DataLoaders
"""




