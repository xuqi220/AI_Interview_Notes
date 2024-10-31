# 深度学习基础
## 激活函数的作用
为什么要在网络中引入激活函数？如果不引入激活函数那么我们的网络是这个样子的$f(x)=A(B(CX))=ABCX=DX,ABC=D$。那么，我们的神经网络的就变成了一个线性函数，是空间中的一个超平面，大大限制了神经网络的拟合能力。激活函数就是给神经网络引入非线性，使得神经网络可以拟合各种复杂函数。那么任何激活函数可以随便选择吗？

### Sigmoid
* 函数公式：

$$f(x)=\frac{1}{1+e^{-x}}$$

<img src="asset/sigmoid.png">

* 导数公式：

$$f^{'}(x)=f(x)(1-f(x))$$

<img src="asset/sigmoid_diff.png">

* 优点
    * 函数平滑、易于求导
    * 可直接作为分类模型的输出

* 缺点
    * 计算量大，包含幂运算，以及除法运算；
    * sigmoid 导数的取值范围是 [0, 0.25]，最大值都是小于 1 的，反向传播时又是"链式传导"，经过几次相乘之后很容易就会出现梯度消失的问题，不利于模型加深；
    * sigmoid 的输出的均值不是0（即zero-centered），这会导致当前层接收到上一层的非0均值的信号作为输入，随着网络的加深，会改变数据的原始分布，不利于模型的收敛。

### Tanh


### ReLU
ReLU 全称为 Rectified Linear Unit，即修正线性单元函数。该函数的公式比较简单，相应的公式和图像如下表所示。
* 函数公式
$$ReLU(x)=\begin{cases}  
0 & if & x \leq 0\\
x & if & x > 0 
\end{cases}$$

<img src="asset/relu.png">

* 导函数公式
$$ReLU(x)=\begin{cases}  
0 & if \; x \leq 0\\
1 & if \; x > 0 
\end{cases}$$

<img src="asset/relu_diff.png">