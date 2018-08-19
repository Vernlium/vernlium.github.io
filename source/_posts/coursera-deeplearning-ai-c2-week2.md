---
title: coursera-deeplearning-ai-c2-week2
date: 2018-07-05 07:16:02
tags:
---

本周课程主要讲解了神经网络的一些优化算法，要点：



## 课程笔记

### Optimization algorithms

#### 课程视频Mini-batch gradient descent


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

#### 课程视频Understanding mini-batch gradient descent
#### 课程视频Exponentially weighted averages
#### 课程视频Understanding exponentially weighted averages
#### 课程视频Bias correction in exponentially weighted averages
#### 课程视频Gradient descent with momentum
#### 课程视频RMSprop
#### 课程视频Adam optimization algorithm
#### 课程视频Learning rate decay
#### 课程视频The problem of local optima

### Programming assignment

#### 编程作业: Optimization
