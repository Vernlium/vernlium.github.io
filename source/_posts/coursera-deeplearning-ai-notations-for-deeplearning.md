---
title: coursera-deeplearning-ai-notations_for_deeplearning
date: 2018-07-15 16:31:39
tags:
---

本文主要总结一些深度学习中的一些符号。

#### 一般

- 上标 $i$: 表示第$i$个训练实例
- 上标 $l$: 表示第$l$层

#### 尺寸（size）

- $m$: 数据集中实例的数量
- $n_x$: 输入的大小
- $n_y$: 输出的大小（或者是类别的数量）
- $n_h^{[l]}$: 网络中第$l$层隐藏单元的数量。在循环中，一般： $ n_x = n_h^{[0]} $，$ n_y = n_h^{[number of layers + 1]} $ 
- $L$: 网络中层的个数

#### 对象（Objects）

- $X \in \mathbb{R}^{n_x \times m}$ : 输入矩阵
- $ x^{(i)} \in \mathbb{R}^{n_x}$: 列向量表示的第$i$个实例
- $Y \in \mathbb{R}^{n_y \times m}$ : 输出标签矩阵
- $ y^{(i)} \in \mathbb{R}^{n_y}$: 第$i$个实例输出标签
- $W^{[l]} \in \mathbb{R}^{number of units in next layer \times number of units in the previous layer}$ : 权重(weight)矩阵，上标$[l]$表示层
- $b^{[l]} \in \mathbb{R}^{number of units in next layer}$ : 第$l$层的偏置（bias）矩阵
- $ \hat{y} \in \mathbb{R}^{n_y}$: 预测输出向量，也可以表示为$a^{[L]}$，其中$L$是网络中层的数量

#### 前向传播公式示例

- $ a = g^{[l]}(W_{x} x^{(i)} + b_1) = g^{[l]}(z_1)$，其中$g^{[l]}$表示第$l$层的激活函数
- $ \hat{y}^{(i)} = softmax(W_h h + b_2) $
- 一般激活公式：
$$ a_j^{[l]} = g^{[l]}(\sum_k w_j^{[l]} a_k^{[l-1]} + b_j^{[l]}) = g^{[l]}(z_j^{[l]}) $$
- $J(x,W,b,y)$ 或 $J(\hat{y},y)$:表示损失函数



#### 损失函数示例

- $$ J_{CE}(\hat{y},y) = - \sum_{i=0}^m y^{(i)}log(\hat{y}^{(i)}) $$
- $$ J_{1}(\hat{y},y) = - \sum_{i=0}^m |y^{(i)} - \hat{y}^{(i)}| $$