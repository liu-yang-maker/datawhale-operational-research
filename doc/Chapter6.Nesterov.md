# Nesterov 加速算法

Nesterov加速方法的基本迭代形式为：
$$
\begin{aligned}
v_{t} &=\mu_{t-1} v_{t-1}-\epsilon_{t-1} \nabla g\left(\theta_{t-1}+\mu_{t-1} v_{t-1}\right) \\
\theta_{t} &=\theta_{t-1}+v_{t}
\end{aligned}
$$
和动量方法的区别在于二者用到了不同点的梯度，动量方法采用的是上一步 $\theta_{t-1}$ 的梯度方向，而Nesterov加速方法则是从 $\theta_{t-1}$ 朝着 $v_{t-1}$ 往前一步。 一种解释是，反正要朝着方 $v_{t-1}$ 向走，不如先利用了这个信息。 接下来我来推导出第二种等价形式
$$
\begin{aligned}
\theta_{t} &=\theta_{t-1}+v_{t} \\
&=\theta_{t-1}+\mu_{t-1} v_{t-1}-\epsilon_{t-1} \nabla g\left(\theta_{t-1}+\mu_{t-1} v_{t-1}\right) \\
&=\theta_{t-1}+\mu_{t-1}\left(\theta_{t-1}-\theta_{t-2}\right)-\epsilon_{t-1} \nabla g\left(\theta_{t-1}+\mu_{t-1}\left(\theta_{t-1}-\theta_{t-2}\right)\right)
\end{aligned}
$$
然后引入中间变量 $y_{t-1}$ ，使得它满足
$$
y_{t-1}=\theta_{t-1}+\mu_{t-1}\left(\theta_{t-1}-\theta_{t-2}\right)
$$
然后得到第二种等价形式
$$
\begin{aligned}
&\theta_{t}=y_{t-1}-\epsilon_{t-1} \nabla g\left(y_{t-1}\right) \\
&y_{t}=\theta_{t}+\mu_{t}\left(\theta_{t}-\theta_{t-1}\right)
\end{aligned}
$$
这可以理解为，先走个梯度步，然后再走个加速步。 这两种形式是完全等价的。