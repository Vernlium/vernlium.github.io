---
title: coursera_deeplearning.ai_c1_week2
date: 2018-04-27 07:52:09
tags: [deeplearning.ai]
---

### 逻辑回归的实现

#### 计算图

前向计算：

梯度计算：

参数更新：

### 矢量化

对于每一个 $x^{(i)}$:

$$ z^{(i)} = w^T x^{(i)} + b $$

$$ \hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)}) $$ 

$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)}) $$

The cost is then computed by summing over all training examples:

$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)}) $$

前向计算：

- 得到输入X
- 前向计算： $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
- 计算损失函数: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
    
计算梯度：
    
$$ dw = \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T $$
$$ db = \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)}) $$

参数更新：

$$ w = w - \alpha * dw $$
$$ b = b - \alpha * db $$