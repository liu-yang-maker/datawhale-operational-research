# 动量方法 (Momentum)

## 1.1 背景

使用梯度下降法，每次都会朝着目标函数下降最快的方向，这种更新方法看似非常快，实际上存在一些问题。

考虑一个二维输入，$[x_1, x_2]$，输出的损失函数 $L: R^2 \rightarrow R$，下面是这个函数的等高线。可以想象成一个很扁的漏斗，这样在竖直方向上，梯度就非常大，在水平方向上，梯度就相对较小，所以我们在设置学习率的时候就不能设置太大，为了防止竖直方向上参数更新太过了，这样一个较小的学习率又导致了水平方向上参数在更新的时候太过于缓慢，所以就导致最终收敛起来非常慢。

![Moment](.\images\SGD.jpg)

为了克服这一缺陷，人们提出了动量方法（momentum），即带有动量的梯度下降算法（简称动量），旨在加速学习。动量法借鉴了物理学的思想，想象一下在无摩擦的碗里滚动一个球，没有阻力时，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。

![1](.\images\1.gif)

动量法的思想是：参数更新时在一定程度上保留之前更新的方向，同时又利用当前 `batch` 的梯度微调最终方向，简而言之就是通过积累之前的动量来加速当前的梯度。这样一来，可以在一定程度上增加稳定性，从而学习得更快，并且还有一定摆脱局部最优解的能力[^1]。综上，动量法具有两个明显的优点：

1. 动量移动得更快(因为它积累的所有动量)
2. 动量有机会逃脱局部极小值(因为动量可能推动它脱离局部极小值)。同样，我们将在后面看到，它也将更好地通过高原区

## 1.2 动量法原理

### 1.2.1 公式推导

从形式上来看，动量方法将当前的梯度与上一步移动方向相结合，以加速算法的收敛。具体而言，它引入了一个速度变量 $v$，它代表参数移动的方向和大小。

首先，我们回顾一下标准梯度下降的迭代公式：

$$x_{t+1} = x_t - \alpha\nabla f(x_t)$$

其中，$\alpha$ 叫做步长，或者叫做学习率（`learning rate`），而动量法在此基础上引入了“动量项“  $\beta(x_t - x_{t-1})$。因此，完整的动量迭代[^3]：

$$ v_t = \beta v_{t-1} - \alpha\nabla f(x_t)$$

$$ x_t = x_{t-1} + v_t$$

 $$v_t = \beta(x_t - x_{t-1})$$

其中 $v_t$ 是当前速度，$\beta$ 是动量参数，是一个小于 1的正数，$\alpha$ 是学习率。

如果我们把当前迭代想象成一个有质量的小球，那么我们的梯度下降更新应该与之前的步长成正比。整理可得：

$$x_t+1 = x_t - \eta \nabla f(x_t) + \beta(x_t - x_{t-1})$$

### 1.2.2 实例推导

接下来，我们以一个简单的凸函数来推导动量法，考虑一个简单的二次目标[^2]：

$$f(x) = \frac{h}{2} x^2$$

于是又动量更新规则为：
$$
\begin{array}{l}
x_{t+1} &= x_t - \alpha \nabla f(x_t) + \beta (x_t - x_{t-1}) \\
	&= x_t -\alpha h x_t + \beta (x_t - x_{t-1}) \\
	&= (1+\beta -\alpha h) x_t - \beta x_{t-1}
\end{array}
$$
进而可以得到线性表达式：
$$
\left[\begin{array}{c}
x_{t+1} \\
x_{t}
\end{array}\right]=\left[\begin{array}{cc}
1-\alpha h+\beta & -\beta \\
1 & 0
\end{array}\right]\left[\begin{array}{c}
x_{t} \\
x_{t-1}
\end{array}\right]
$$
令 $A = \left[\begin{array}{cc} 1-\alpha h+\beta & -\beta \\ 1 & 0 \end{array}\right]$，因此可以将 $A$ 进行递归 $t$ 步得到 $x_{t+1}, x_t$ 和 $x_1, x_0$ 之间的关系，有：
$$
\left[\begin{array}{c}
x_{t+1} \\
x_{t}
\end{array}\right]=A^{t}\left[\begin{array}{l}
x_{1} \\
x_{0}
\end{array}\right]
$$
考虑将 $x_t$ 与最优的 $x$ 进行比较，有：
$$
\left[\begin{array}{c}
x_{t+1}-x^{*} \\
x_{t}-x^{*}
\end{array}\right]=A^{t}\left[\begin{array}{l}
x_{1}-x^{*} \\
x_{0}-x^{*}
\end{array}\right]
$$
取范数：
$$
\left\|\left[\begin{array}{c}
x_{t+1}-x^{*} \\
x_{t}-x^{*}
\end{array}\right]\right\|_{2}=\left\|A^{t}\left[\begin{array}{l}
x_{1}-x^{*} \\
x_{0}-x^{*}
\end{array}\right]\right\|_{2} \leq\left\|A^{t}\right\|_{2}\left\|\left[\begin{array}{l}
x_{1}-x^{*} \\
x_{0}-x^{*}
\end{array}\right]\right\|_{2}
$$

##### Lemma 1.  给定向量空间 $M_n$ 内的矩阵 $A$ 和 $\varepsilon > 0$，存在矩阵范数 $\|\cdot\|$ 满足 $\|A\| \leq \rho(A)+\varepsilon$，其中，$\rho (A) = \max \{ |\lambda_1, \cdots, \lambda_n|\}$（即特征向量中的最大值）。[^6]

所以，由 `Lemma 1`可知，存在一个矩阵范数满足：
$$
\left\|A^{t}\right\| \leq(\rho(A)+\epsilon)^{t}
$$
其中，$\rho (A) = \max \{ |\lambda_1, \lambda_2|\}$，$\lambda_1, \lambda_2$分别表示特征向量，因此有：
$$
\left\|\left[\begin{array}{c}
x_{t+1}-x^{*} \\
x_{t}-x^{*}
\end{array}\right]\right\|_{2} \leq(\rho(A)+\epsilon)^{t}\left\|\left[\begin{array}{l}
x_{1}-x^{*} \\
x_{0}-x^{*}
\end{array}\right]\right\|_{2}
$$
可以发现，经过迭代最终能够收敛到一个稳定的点。

## 1.3 如何理解动量法

如果我们把当前迭代想象成一个有质量的小球，那么我们的梯度下降更新应该与之前的步长成正比。让我们考虑两个极端情况来更好地理解动量。如果动量参数  $\gamma = 0$，那么它与最初始的梯度下降完全相同，如果 $\gamma=1$，那么相当于最开始的无摩擦的碗类比一样，前后不停地摇摆，这肯定不是我们想要的结果。通常动量参数选择在 `0.8 - 0.9` 左右，它就像一个有一点摩擦的表面，所以它最终会减慢并停止[^4]。

![Momentum-decy](.\images\Momentum-decy.gif)

综上，动量法相当于每次在进行参数更新的时候，都会将之前的速度考虑进来，每个参数在各方向上的移动幅度不仅取决于当前的梯度，还取决于过去各个梯度在各个方向上是否一致，如果一个梯度一直沿着当前方向进行更新，那么每次更新的幅度就越来越大，如果一个梯度在一个方向上不断变化，那么其更新幅度就会被衰减，这样我们就可以使用一个较大的学习率，使得收敛更快，同时梯度比较大的方向就会因为动量的关系每次更新的幅度减少，如下图：

![Moment](.\images\Moment.jpg)

比如我们的梯度每次都等于 $g$，而且方向都相同，那么动量法在该反方向上使参数加速移动，有下面的公式：

$$ v_0 = 0$$

$$ v_1 = \beta v_0 + \alpha g = \alpha g$$

$$ v_2 = \beta v_1 + \alpha g = (1 + \beta ) \alpha g$$

$$ v_3 = \beta v_2 + \alpha g = (1 + \beta + \beta ^2) \alpha g$$

$$ \cdots$$

$$ v_{+ \infty} = (1 + \beta + \beta ^2 + \beta ^3 + \cdots) \alpha g = \frac{1}{1 - \beta } \alpha g$$

如果我们把 $\beta $ 定为 0.9，那么更新幅度的峰值就是原本梯度乘学习率的 10 倍。

本质上说，动量法就仿佛我们从高坡上推一个球，小球在向下滚动的过程中积累了动量，在途中也会变得越来越快，最后会达到一个峰值，对应于我们的算法中就是，动量项会沿着梯度指向方向相同的方向不断增大，对于梯度方向改变的方向逐渐减小，得到了更快的收敛速度以及更小的震荡。

## 1.4 动量法与梯度下降法直观对比

设二次函数 $f(x, y) = x^2 + 10 y^2$，分别取初始点 $(x^0, y^0)$ 取为 $(10, 1)$ 和 $(-10, -1)$，我们使用梯度法和动量法进行 `15` 次迭代，结果如下图所示。可以看到普通梯度法生成的点列会在椭圆的短轴方向上来回移动，而动量方法生成的点列更快收敛到了最小值点。

<img src=".\images\image-20220211162555542.png" alt="image-20220211162555542" style="zoom:150%;" />

<center> Fig 动量方法和梯度下降法的表现对比 </center>

## 1.5 动量法的实现

### 1.5.1 自定义函数实现动量法

下面，我们手动实现一个动量法，公式已在上面给出。

- 导入模块

  ```python 
  import torch
  import torch.utils.data as Dataa
  import torch.nn.functional as F
  from torch.autograd import Variable
  import matplotlib.pyplot as plt
  %matplotlib inline
  
  torch.manual_seed(1)    # 固定随机数，让结果可复现
  ```

- 定义动量法函数

  ```python
  def sgd_momentum(parameters, vs, lr, gamma):
      for param, v in zip(parameters, vs):
          v[:] = gamma * v + lr * param.grad.data
          param.data = param.data - v
  ```

- 生成数据并创建 `dataset`

  为简单起见，我们去一个简单的函数，$f(x) = x^2 + 0.1\epsilon$，其中 $\epsilon$ 为服从标准正态分布的扰动项。

  ```python 
  # generate data
  x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
  y = x.pow(2) + 0.1*torch.randn(*x.size())
  
  # batch training
  torch_dataset = Data.TensorDataset(x, y)
  loader = Data.DataLoader(
      dataset=torch_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True)
  
  # plot dataset
  plt.scatter(x.numpy(), y.numpy())
  plt.show()
  ```

  Results:

  ![data](.\images\data.png)

- 定义简单的线性神经网络

  ```python 
  # 设置超参数
  LR = 0.01
  BATCH_SIZE = 32
  EPOCH = 12
  
  # 定义神经网络
  class Net(torch.nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.hidden = torch.nn.Linear(1, 20)  
          self.predict = torch.nn.Linear(20, 1)
  
      def forward(self, x):
          x = F.relu(self.hidden(x)) # activation function for hidden layer
          x = self.predict(x) # linear output
          return x
  
  # 初始化
  net_Momentum = Net()
  
  # loss function
  loss_func = torch.nn.MSELoss()
  losses_momentum = []   # record loss
  ```

- 定义动量法函数

  ```python
  # 将速度初始化为和参数形状相同的零张量
  vs = []
  for param in net.parameters():
      vs.append(torch.zeros_like(param.data))
  
  def sgd_momentum(parameters, vs, lr, gamma):
      for param, v in zip(parameters, vs):
          v[:] = gamma * v + lr * param.grad.data
          param.data = param.data - v
  ```

- 模型训练并绘制 `loss` 图

  ```python 
  # training
  for epoch in range(EPOCH):
      train_loss = 0 
      for step, (batch_x, batch_y) in enumerate(loader):          # for each training step
          b_x = Variable(batch_x)
          b_y = Variable(batch_y)
  
          output = net_Momentum(b_x)              # get output for every net
          loss = loss_func(output, b_y)  # compute loss for every net
          net_Momentum.zero_grad()                # clear gradients for next train
          loss.backward()                # backpropagation, compute gradients
          sgd_momentum(net_Momentum.parameters(), vs, 1e-2, 0.8) # 使用的动量参数为 0.8，学习率 0.01
          
          train_loss += loss.item()
          losses_momentum.append(loss.item())     # loss recoder
      
      print('epoch: {}, Train Loss: {:.6f}'
        .format(epoch, train_loss / len(train_data)))
  
  plt.plot(losses_momentum, label='Momentum')
  plt.legend(loc='best')
  plt.xlabel('Steps')
  plt.ylabel('Loss')
  plt.show()
  ```

  Results:

  ![loss-momentum](.\images\loss-momentum.png)

#### 完整代码[^5]

```python
import torch
import torch.utils.data as Dataa
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(1)    # reproducible

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# generate data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.randn(*x.size())
# unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

# batch training
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

# define neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x)) # activation function for hidden layer
        x = self.predict(x) # linear output
        return x

# different optimizer
net_Momentum    = Net()

# loss function
loss_func = torch.nn.MSELoss()
losses_momentum = []   # record loss

# 将速度初始化为和参数形状相同的零张量
vs = []
for param in net_Momentum.parameters():
    vs.append(torch.zeros_like(param.data))

# training
for epoch in range(EPOCH):
    train_loss = 0 
    for step, (batch_x, batch_y) in enumerate(loader):          # for each training step
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        output = net_Momentum(b_x)              # get output for every net
        loss = loss_func(output, b_y)  # compute loss for every net
        net_Momentum.zero_grad()                # clear gradients for next train
        loss.backward()                # backpropagation, compute gradients
        sgd_momentum(net_Momentum.parameters(), vs, 1e-2, 0.8) # 使用的动量参数为 0.8，学习率 0.01
        
        train_loss += loss.item()
        losses_momentum.append(loss.item())     # loss recoder
    
    print('epoch: {}, Train Loss: {:.6f}'
      .format(epoch, train_loss / len(train_data)))

plt.plot(losses_momentum, label='Momentum')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
```

### 1.5.2 调用 Pytorch 内置函数实现动量法

事实上，`pytorch` 内置了动量法的实现，非常简单，直接在 `torch.optim.SGD(momentum=0.8)` 即可，下面实现一下

```
import torch
import torch.utils.data as Dataa
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(1)    # reproducible

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# generate datas
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.randn(*x.size())
# unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

# batch training
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

# define neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x)) # activation function for hidden layer
        x = self.predict(x) # linear output
        return x

# different optimizer
net_Momentum    = Net()

# loss function
loss_func = torch.nn.MSELoss()
losses_momentum = []   # record loss
opt_Momentum  = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)

# 将速度初始化为和参数形状相同的零张量
vs = []
for param in net_Momentum.parameters():
    vs.append(torch.zeros_like(param.data))

# training
# training
for epoch in range(EPOCH):
    train_loss = 0 
    for step, (batch_x, batch_y) in enumerate(loader):          # for each training step
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        output = net_Momentum(b_x)              # get output for every net
        loss = loss_func(output, b_y)  # compute loss for every net
        net_Momentum.zero_grad()                # clear gradients for next train
        loss.backward()                # backpropagation, compute gradients
        opt_Momentum.step() # 使用的动量参数为 0.9，学习率 0.01
        
        train_loss += loss.item()
        losses_momentum.append(loss.item())     # loss recoder
    
    print('epoch: {}, Train Loss: {:.6f}'
      .format(epoch, train_loss / len(train_data)))

plt.plot(losses_momentum, label='Momentum')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
```

## 1.6 动量法和随机梯度下降法对比

我们可以对比一下动量法与不加动量的随机梯度下降法：

```python
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline

torch.manual_seed(1)    # reproducible

# hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# generate datas
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.randn(*x.size())
# unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

# batch training
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

# define neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x)) # activation function for hidden layer
        x = self.predict(x) # linear output
        return x

# different optimizer
net_SGD         = Net()
net_Momentum    = Net()
# net_RMSprop     = Net()
# net_Adam        = Net()
nets = [net_SGD, net_Momentum]

opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
# opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

optimizers = [opt_SGD, opt_Momentum]

# loss function
loss_func = torch.nn.MSELoss()
losses_his = [[], []]   # record loss

# training
# training
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):          # for each training step
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)              # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            l_his.append(loss.item())     # loss recoder

labels = ['SGD', 'Momentum']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.ylim((0, 0.2))
plt.show()
```

Results:

![comparison](.\images\comparison.png)

可以看到加完动量之后的 `loss` 下降的程度更低了，可以将动量理解为一种惯性作用，所以每次更新的幅度都会比不加动量的情况更多。

---

## Reference

1. 刘浩洋, 户将, 李勇锋, 文再文. (2021). 最优化：建模、算法与理论. 北京: 高教出版社.
2. IFT 6169: Theoretical principles for deep learning，https://mitliagkas.github.io/ift6085-dl-theory-class/
3. Pytorch 中文手册, https://wizardforcel.gitbooks.io/learn-dl-with-pytorch-liaoxingyu/content/3.6.2.html
4. 梯度下降的可视化解释(Adam，AdaGrad，Momentum，RMSProp)，https://mp.weixin.qq.com/s/LyNrPoEirLk0zwBxu0c18g
5. Pytorch 学习笔记，https://www.yangsuoly.com/2021/04/08/Pytorch/
6. S. Foucart. Matrix norm and spectral radius. University Lecture, 2012. URL http://www.math.drexel.edu/~foucart/TeachingFiles/F12/M504Lect6.pdf