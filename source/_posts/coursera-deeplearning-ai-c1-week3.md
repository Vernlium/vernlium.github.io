---
title: coursera_deeplearning.ai_c1_week3
date: 2018-07-25 07:52:15
tags: [deeplearning.ai]
---

本周课程要点：
- 含有单独隐藏层的神经网络
- 参数初始化
- 使用前向传播进行预测
- 以及在梯度下降时使用反向传播中 
- 涉及的导数计算 

## 课程笔记

### Neural Network Overview

{% asset_img neural_networks_overview.jpg just the right amount %}

### Neural Network Represention

{% asset_img neural_networks_representation.jpg just the right amount %}

### Neural Network Computing and Vectorizing

{% asset_img neural_networks_computing.jpg just the right amount %}

对于一层的神经网络，每个网络节点都会进行如下两个运算：

$ z = w^T x + b $

$ \hat{y} = a = \sigma(z) $ 

{% asset_img neural_networks_computing_mulnode.jpg just the right amount %}

上图中的手写部分，是把输入和参数矢量化。

{% asset_img neural_networks_computing_2_layer.jpg just the right amount %}

上图中是一个两层网络，第一层是4个节点，第二层1个节点，通过矢量化，并且把输入作为$a^{[0]}$，可以归结为右侧的4个公式。

{% asset_img recap_of_vectorizing_across_multiple_examples.jpg just the right amount %}

### 激活函数

week2中用到的只有一个激活函数：`sigmoid`函数，除此之外还有其他激活函数。

- sigmoid函数
$$ sigmoid(z) =\frac{1}{1+e^{-z}}  $$
- tanh函数
$$ tanh(z) =\frac{e^z-e^{-z}}{e^z+e^{-z}}  $$
- ReLU
$$
        f(x) =
        \begin{cases}
        x,  & \text{if $x$ >= 0} \\
        0, & \text{if $x$ < 0}
        \end{cases}
$$
- LReLU
$$
        f(x) =
        \begin{cases}
        x,  & \text{if $x$ >= 0} \\
        \alpha x, & \text{if $x$ < 0}
        \end{cases}
$$

曲线分别为：

{% asset_img activation_function_curve.jpg just the right amount %}

#### Why do you need non-linear activation functions?

**为什么要使用激活函数？**

如果你使用“线性激活函数”或者叫“恒等激活函数”，那么神经网络的输出仅仅是输入函数的线性变化。深度网络会有很多层，很多隐藏层的神经网络。如果使用线性激活函数 或者说 没有使用激活函数，那么无论神经网络有多少层，它所做的仅仅是计算线性激活函数，这还不如去除所有隐藏层。请记得：线性的隐藏层没有任何用处，因为两个线性函数的组合，仍然是线性函数，除非在这里引入非线性函数，否则无论神经网络模型包含多少隐藏层，都无法实现更有趣的功能。

### 神经网络的梯度下降

对于一个2层的神经网络，使用梯度下降法计算的方式为：

- Paramters :
$$
W^{[1]},\,\,\,b^{[1]},\,\,\,W^{[2]},\,\,\,b^{[2]}
$$

其维度分别为：

$$
dims:(n^{[1]},\,\,\,n^{[0]}),\,\,\,(n^{[1]},1),\,\,\,(n^{[2]},n^{[1]}),\,\,\,(n^{[2]},1)
$$

- Cost Function :
$$ J(W^{[1]},b^{[1]},W^{[2]},b^{[2]}) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)}) $$
- Gradient descent :
$$
\begin{array}{l}
\text{Repeat \{} \\
\,\,\,\,\,\,\,\,Compute\,\, predicts(\hat{y}_{(i)},i=1,2,\cdots,m) \\
\,\,\,\,\,\,\,\, dW^{[1]} = \frac{d\mathcal{L}}{dW^{[1]}},db^{[1]} = \frac{d\mathcal{L}}{db^{[1]}} \\
\,\,\,\,\,\,\,\, dW^{[2]} = \frac{d\mathcal{L}}{dW^{[2]}},db^{[2]} = \frac{d\mathcal{L}}{db^{[2]}} \\
\,\,\,\,\,\,\,\,W^{[1]} = W^{[1]} - \alpha * dW^{[1]} \\
\,\,\,\,\,\,\,\,b^{[1]} = b^{[1]} - \alpha * db^{[1]} \\
\,\,\,\,\,\,\,\,W^{[2]} = W^{[2]} - \alpha * dW^{[2]} \\
\,\,\,\,\,\,\,\,b^{[2]} = b^{[2]} - \alpha * db^{[2]} \\
\text{\}} 
\end{array}
$$

#### 进行梯度计算的公式

##### 前向计算

$ Z^{[1]} = {W^{[1]}} X + b^{[1]} $

$ A^{[1]} =  g^{[1]}(Z^{[1]}) $

$ Z^{[2]} = {W^{[2]}} X + b^{[2]} $

$ A^{[2]} = g^{[2]}(Z^{[2]}) $

对二分类问题，$g^{[2]}$依然使用$sigmoid$函数。

##### 反向传播

$ dZ^{[2]} = A^{[2]} - Y $

$ dW^{[2]} =\frac{1}{m} dZ^{[2]} {A^{[1]}}^{T} $

$ db^{[2]} =\frac{1}{m} np.sum(dZ^{[2]},axis=1) $

$ dZ^{[1]} = (W^{[2]})^T dZ^{[2]} (g^{[1]})'(Z^{[1]}) $

$ dW^{[1]} =\frac{1}{m} dZ^{[1]} X^{T} $

$ db^{[1]} =\frac{1}{m} np.sum(dZ^{[1]},axis=1) $

### 反向传播的公式推导

{% asset_img neural_networks_gradients.jpg just the right amount %}

{% asset_img grad_summary.png just the right amount %}

上图中左侧是单个实例的计算公式，右侧是m个实例矢量化后的计算公式。

### 随机初始化

当你开始训练神经网络时，将权重参数进行随机初始化非常重要。在逻辑回归的问题中，把权重参数初始化为零是可行的， 但把神经网络的权重参数全部初始化为零， 并使用梯度下降， 将无法获得预期的效果。

这里有两个输入样本参数，因此$n^{[0]}$等于2，还有两个隐藏单元，因此$n^{[1]}$也等于2。所以与隐藏层关联的权重$w^{[1]}$ 是一个2x2的矩阵。现在我们将这个矩阵的初始值都设为0，同样我们将矩阵$b^{[1]}$的值也都初始化为零。偏差矩阵$b^{[1]}$的初始值都是0，不会影响最终结果。**但是将权重参数矩阵$w^{[1]}$初始值都设为零，会引起某些问题。这样的初始权重参数会导致，无论使用什么样的样本进行训练$ a^{[1]}_1$与$a^{[1]}_2$始终是相同的，这第一个激活函数和这第二个激活函数将是完全一致的，因为这些隐藏神经元在进行完全相同的计算工作**。当你进行反向传播的计算时，由于 **对称问题**，这些隐藏单元将会在同样的条件下被初始化，最终导致$z^{[1]}_1$的导数和$z^{[1]}_2$的导数也不会有差别。同样的，假设输出的权重也是相同的，所以输出权重参数矩阵$w^{[2]}$也等于$[0,0]$。但如果使用这种方法来初始化神经网络，那么上面这个隐藏单元和下面这个隐藏单元也是相同的，它们实现的是完全相同的功能，看可以称这是“对称”的。

 {% asset_img  what_happens_if_you_initialize_weights_to_zero.jpg just the right amount %}

归纳一下这个结果：经过每一次训练迭代，将会得到两个实现完全相同功能的隐藏单元，$W$的导数将会是一个矩阵且每一行都是相同的，然后进行一次权重更新，当在实际操作时$ w^{[1]} $将被更新成$ w^{[1]}-α*dw $时，将会发现，经过每一次迭代后 $w^{[1]}$的第一行与第二行是相同的。所以根据上述信息来归纳，可以得到一个证明结果：**如果在神经网络中，将所有权重参数矩阵w的值初始化为零，由于两个隐藏单元肩负着相同的计算功能，并且也将同样的影响作用在输出神经元上，经过一次迭代后，依然会得到相同的结果。这两个隐藏神经元依然是“对称”的。同样推导下去，经过两次迭代 三次迭代，以及更多次迭代，无论将这个神经网络训练多久，这两个隐藏单元仍然在使用同样的功能进行运算。**

## 编程练习

### Planar data classification with one hidden layer

Welcome to your week 3 programming assignment. It's time to build your first neural network, which will have a hidden layer. You will see a big difference between this model and the one you implemented using logistic regression. 

**You will learn how to:**
- Implement a 2-class classification neural network with a single hidden layer
- Use units with a non-linear activation function, such as tanh 
- Compute the cross entropy loss 
- Implement forward and backward propagation

#### 1 - Packages

Let's first import all the packages that you will need during this assignment.
- [numpy](www.numpy.org) is the fundamental package for scientific computing with Python.
- [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis. 
- [matplotlib](http://matplotlib.org) is a library for plotting graphs in Python.
- testCases provides some test examples to assess the correctness of your functions
- planar_utils provide various useful functions used in this assignment

```python
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().magic('matplotlib inline')

np.random.seed(1) # set a seed so that the results are consistent
```

#### 2 - Dataset

First, let's get the dataset you will work on. The following code will load a "flower" 2-class dataset into variables `X` and `Y`.


```python
X, Y = load_planar_dataset()
```

Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) and some blue (y=1) points. Your goal is to build a model to fit this data. 


```python
# Visualize the data:

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
```

{% asset_img  output_6_0.png just the right amount %}

You have:
- a numpy-array (matrix) X that contains your features (x1, x2)
- a numpy-array (vector) Y that contains your labels (red:0, blue:1).

Lets first get a better sense of what our data is like. 

**Exercise**: How many training examples do you have? In addition, what is the `shape` of the variables `X` and `Y`? 

**Hint**: How do you get the shape of a numpy array? [(help)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html)

```python

### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]  # training set size
### END CODE HERE ###

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
```

    The shape of X is: (2, 400)
    The shape of Y is: (1, 400)
    I have m = 400 training examples!

#### 3 - Simple Logistic Regression

Before building a full neural network, lets first see how logistic regression performs on this problem. You can use sklearn's built-in functions to do that. Run the code below to train a logistic regression classifier on the dataset.

```python
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
```

You can now plot the decision boundary of these models. Run the code below.

```python
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
```

    Accuracy of logistic regression: 47 % (percentage of correctly labelled datapoints)


{% asset_img  output_13_1.png just the right amount %}

**Interpretation**: The dataset is not linearly separable, so logistic regression doesn't perform well. Hopefully a neural network will do better. Let's try this now! 

#### 4 - Neural Network model

Logistic regression did not work well on the "flower dataset". You are going to train a Neural Network with a single hidden layer.

**Here is our model**:

{% asset_img  classification_kiank.png just the right amount %}

**Mathematically**:

For one example $x^{(i)}$:

$$ z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1]} $$ 

$$ a^{[1] (i)} = \tanh(z^{[1] (i)}) $$

$$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]} $$

$$ \hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)}) $$

$$ y^{(i)}_{prediction} = 
\begin{cases} 
1 & \text{if } a^{[2](i)} > 0.5 \\ 
0 & \text{otherwise } 
\end{cases} 
$$

Given the predictions on all the examples, you can also compute the cost $J$ as follows: 

$$
J = - \frac{1}{m} \sum_{i=0}^{m} (y^{(i)}\log(a^{[2] (i)}) + (1-y^{(i)})\log(1- a^{[2] (i)}))
 $$

**Reminder**: The general methodology to build a Neural Network is to:

1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
2. Initialize the model's parameters
3. Loop:
    - Implement forward propagation
    - Compute loss
    - Implement backward propagation to get the gradients
    - Update parameters (gradient descent)

You often build helper functions to compute steps 1-3 and then merge them into one function we call `nn_model()`. Once you've built `nn_model()` and learnt the right parameters, you can make predictions on new data.

##### 4.1 - Defining the neural network structure

**Exercise**: Define three variables:

- n_x: the size of the input layer
- n_h: the size of the hidden layer (set this to 4) 
- n_y: the size of the output layer

**Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

```python
GRADED FUNCTION: layer_sizes
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    ### END CODE HERE ###
    return (n_x, n_h, n_y)
```

```python
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
```

    The size of the input layer is: n_x = 5
    The size of the hidden layer is: n_h = 4
    The size of the output layer is: n_y = 2


##### 4.2 - Initialize the model's parameters

**Exercise**: Implement the function `initialize_parameters()`.

**Instructions**:
- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
- You will initialize the weights matrices with random values. 
    - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vectors as zeros. 
    - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.

```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```



##### 4.3 - The Loop

**Question**: Implement `forward_propagation()`.

**Instructions**:
- Look above at the mathematical representation of your classifier.
- You can use the function `sigmoid()`. It is built-in (imported) in the notebook.
- You can use the function `np.tanh()`. It is part of the numpy library.
- The steps you have to implement are:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
    2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
- Values needed in the backpropagation are stored in "`cache`". The `cache` will be given as an input to the backpropagation function.

```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    ### END CODE HERE ###
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
```

Now that you have computed $A^{[2]}$ (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for every example, you can compute the cost function as follows:

$$ 
J = - \frac{1}{m} \sum_{i = 0}^{m} ( y^{(i)}\log(a^{[2] (i)}) + (1-y^{(i)})\log(1- a^{[2] (i)} ))
 $$

**Exercise**: Implement `compute_cost()` to compute the value of the cost $J$.

**Instructions**:
- There are many ways to implement the cross-entropy loss. To help you, we give you how we would have implemented
$$ \sum_{i=0}^{m}  y^{(i)}\log(a^{[2](i)}) $$

```python
logprobs = np.multiply(np.log(A2),Y)
cost = - np.sum(logprobs)                # no need to use a for loop!
```

(you can use either `np.multiply()` and then `np.sum()` or directly `np.dot()`).


```python
# GRADED FUNCTION: compute_cost

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2), 1- Y)
    cost = -1 / m * np.sum(logprobs) 
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
```

Using the cache computed during forward propagation, you can now implement backward propagation.

**Question**: Implement the function `backward_propagation()`.

**Instructions**:
Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.  

{% asset_img summary_of_gradient_descent_2.jpg just the right amount %}

- Tips:
    - To compute dZ1 you'll need to compute $g^{[1]'}(Z^{[1]})$. Since $g^{[1]}(.)$ is the tanh activation function, if $a = g^{[1]}(z)$ then $g^{[1]'}(z) = 1-a^2$. So you can compute 
    $g^{[1]'}(Z^{[1]})$ using `(1 - np.power(A1, 2))`.


```python
# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache["A1"]
    A2 = cache["A2"]
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2 -Y
    dW2 = 1 / m * np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1,2))
    dW1 = 1 / m * np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
```


**Question**: Implement the update rule. Use gradient descent. You have to use $(dW1, db1, dW2, db2)$ in order to update $(W1, b1, W2, b2)$.

**General gradient descent rule**: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.

**Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.

{% asset_img sgd.gif just the right amount %}
{% asset_img sgd_bad.gif just the right amount %}


```python
# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```

##### 4.4 - Integrate parts 4.1, 4.2 and 4.3 in nn_model() ####

**Question**: Build your neural network model in `nn_model()`.

**Instructions**: The neural network model has to use the previous functions in the right order.

```python
GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y ,parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```


```python
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

    Cost after iteration 0: 0.692739
    Cost after iteration 1000: 0.000218
    Cost after iteration 2000: 0.000107
    Cost after iteration 3000: 0.000071
    Cost after iteration 4000: 0.000053
    Cost after iteration 5000: 0.000042
    Cost after iteration 6000: 0.000035
    Cost after iteration 7000: 0.000030
    Cost after iteration 8000: 0.000026
    Cost after iteration 9000: 0.000023
    W1 = [[-0.65848169  1.21866811]
     [-0.76204273  1.39377573]
     [ 0.5792005  -1.10397703]
     [ 0.76773391 -1.41477129]]
    b1 = [[ 0.287592  ]
     [ 0.3511264 ]
     [-0.2431246 ]
     [-0.35772805]]
    W2 = [[-2.45566237 -3.27042274  2.00784958  3.36773273]]
    b2 = [[ 0.20459656]]

##### 4.5 Predictions

**Question**: Use your model to predict by building predict().
Use forward propagation to predict results.

**Reminder**:
$$ 
predictions =  y_{prediction} =
\begin{cases}
      1 & \text{if}\ activation > 0.5 \\
      0 & \text{otherwise}
\end{cases}
$$ 

As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: ```X_new = (X > threshold)```


```python
# GRADED FUNCTION: predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    ### END CODE HERE ###
    
    return predictions
```



```python
parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))
```

    predictions mean = 0.666666666667


It is time to run the model and see how it performs on a planar dataset. Run the following code to test your model with a single hidden layer of $n_h$ hidden units.

```python
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
```

```
    Cost after iteration 0: 0.693048
    Cost after iteration 1000: 0.288083
    Cost after iteration 2000: 0.254385
    Cost after iteration 3000: 0.233864
    Cost after iteration 4000: 0.226792
    Cost after iteration 5000: 0.222644
    Cost after iteration 6000: 0.219731
    Cost after iteration 7000: 0.217504
    Cost after iteration 8000: 0.219454
    Cost after iteration 9000: 0.218607
```

{% asset_img output_50_2.png just the right amount %}


Accuracy is really high compared to Logistic Regression. The model has learnt the leaf patterns of the flower! Neural networks are able to learn even highly non-linear decision boundaries, unlike logistic regression. 

Now, let's try out several hidden layer sizes.

##### 4.6 - Tuning hidden layer size (optional/ungraded exercise) ###

Run the following code. It may take 1-2 minutes. You will observe different behaviors of the model for various hidden layer sizes.


```python
# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
```


    Accuracy for 1 hidden units: 67.5 %
    Accuracy for 2 hidden units: 67.25 %
    Accuracy for 3 hidden units: 90.75 %
    Accuracy for 4 hidden units: 90.5 %
    Accuracy for 5 hidden units: 91.25 %
    Accuracy for 20 hidden units: 90.0 %
    Accuracy for 50 hidden units: 90.25 %


{% asset_img output_56_1.png just the right amount %}


**Interpretation**:
- The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data. 
- The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to  fits the data well without also incurring noticable overfitting.
- You will also learn later about regularization, which lets you use very large models (such as n_h = 50) without much overfitting. 

**Optional questions**:

**Note**: Remember to submit the assignment but clicking the blue "Submit Assignment" button at the upper-right. 

Some optional/ungraded questions that you can explore if you wish: 
- What happens when you change the tanh activation for a sigmoid activation or a ReLU activation?
- Play with the learning_rate. What happens?
- What if we change the dataset? (See part 5 below!)

**You've learnt to:**
- Build a complete neural network with a hidden layer
- Make a good use of a non-linear unit
- Implemented forward propagation and backpropagation, and trained a neural network
- See the impact of varying the hidden layer size, including overfitting.

Nice work! 

#### 5) Performance on other datasets

If you want, you can rerun the whole notebook (minus the dataset part) for each of the following datasets.

```python
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
```

{% asset_img output_63_0.png just the right amount %}


Congrats on finishing this Programming Assignment!

Reference:
- http://scs.ryerson.ca/~aharley/neural-networks/
- http://cs231n.github.io/neural-networks-case-study/



## 小结 

本周的课程包含：
- 含有单独隐藏层的神经网络
- 参数初始化
- 使用前向传播进行预测
- 以及在梯度下降时使用反向传播中 
- 涉及的导数计算 