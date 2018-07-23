---
title: coursera_deeplearning.ai_c1_week4
date: 2018-04-27 07:52:21
tags: [deeplearning.ai]
---


Regularization(正则化)

Dropout regularizetion(随机失活正则法（丢弃法）)

dropout的实现：

Inverted dropout:反向随机失活，

```
layer l = 3

keep-prob = 0.8  # 保留网络中某一节点的概率值，0.8意味着layer中的节点有20%的概率被丢弃

d3 = np.random.rand(a3.shape[0],a3.shape[1] ) < keep-prob) ? 1 : 0

a3 = np.multiply(a3,d3)  # 把某些元素置为0，表示此节点被隐藏

a3 /= keep-prob 
```

最后一步的作用是"保证下一层 $Z^{[4]} = W^{[4]}  a^{[3]} + b^{[4]}$ 的期望值不会降低。

**如果觉得某一层比其他层更容易发生过拟合，给这一层设置更低的保留率（keep-prop）**

测试阶段：No dropout

为什么dropout有效： **Can't rely on any one feature,so have to spread out weights.**

shrink weights 压缩权重的平方范数。

normalizing归一化

1. 均值归零：
$$ \mu = \frac{1}{m} \sum_{i=1}^m x^{(i)} \,\,\,\,\,\,\,\,\,  x= x - \mu $$
2. 方差归一化： 
$$ \sigma ^ {2} = \frac{1}{m} \sum_{i=1}^m (x^{(i)})^2  \,\,\,\,  x=   \frac{x}{\sigma ^ {2}} $$


Mini-batch gradient descent 小批量梯度下降法

如果训练集非常大，比如m = 5,000,000 ，则训练会非常慢。

mini_batch

batch_size = 1 : 随机梯度下降（SDG，stochasit gradient descent）
batch_size = m : Batch gradient descent

如何选Batch_size:

- 1.训练集小( <= 2000)： Batch gradient descent
- 2.typical batch_size: 64-512之间的2^n


Batch Gradient Descent:

```
X = daty_input
Y = labels

parameters = initialize_parameters(layers,dims)

for i in range(0,num_iterations):
    a,caches = forword_propagation(X, parameters)
    cost = compute_cost(a,Y)
    grads = backword_propagation(a,caches,parameters)
    parameters = update_parameters(parameters,grads)
```

Stochastic Gradient Descent:

```
X = daty_input
Y = labels

parameters = initialize_parameters(layers,dims)

for i in range(0,num_iterations):
    for j in range(0,m):
        a,caches = forword_propagation(X[:,j], parameters)
        cost = compute_cost(a,Y[:,Y])
        grads = backword_propagation(a,caches,parameters)
        parameters = update_parameters(parameters,grads)
```

Mini Batch Gradient descent 准备数据的两个步骤;

- 1.Shuffle：洗牌：把原输入随机打乱，当然要保持X和Y对应一致
- 2.Partition: 分区：按batch_size分区，前面t-1个分区的大小都是batch_size,最后一个可能不足batch_size

