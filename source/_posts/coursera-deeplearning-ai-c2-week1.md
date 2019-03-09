---
title: coursera-deeplearning-ai-c2-week1
mathjax: true
date: 2018-08-12 17:15:57
tags:
---

本周课程主要讲解了神经网络中的优化的一些方法，要点：
- Train/Dev/Test（训练/开发/测试数据集）
- 偏差和方差
- 欠拟合和过拟合
- Regularization正则化
- Dropout随机失活
- Normalizing归一化
- 梯度消失和梯度爆炸
- 梯度检查

本周课程将从实际应用的角度介绍深度学习，上周课程已经学会了如何实现一个神经网络，本周将学习，实际应用中如何使神经网络高效工作。本周将学习在实际应用中如何使神经网络高效工作。这些方法包括超参数调整，数据准备，再到如何确保优化算法运行得足够快，以使得学习算法能在合理的时间内完成学习任务。

**学习目标**

- Recall that different types of initializations lead to different results
- Recognize the importance of initialization in complex neural networks.
- Recognize the difference between train/dev/test sets
- Diagnose the bias and variance issues in your model
- Learn when and how to use regularization methods such as dropout or L2 regularization.
- Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
- Use gradient checking to verify the correctness of your backpropagation implementation

## 课程笔记

### Setting up your Machine Learning Application

#### Train / Dev / Test sets

数据集分为**训练集**、**开发集**和**测试集**。

假设这是你的训练数据，把它画成一个大矩形，那么传统的做法是你可能会从所有数据中，取出一部分用作训练集，然后再留出一部分作为hold-out交叉验证集（hold-out cross validation set）。这个数据集有时也称为开发集，为了简洁，把它称为"dev set"。再接下来从最后取出一部分用作测试集。

整个工作流程：

- 首先不停地用**训练集**来训练你的算法
- 然后用你的**开发集**或说hold-out交叉验证集来测试，许多不同的模型里哪一个在开发集上效果最好
- 最后评估最终的训练结果，可以用**测试集**对结果中最好的模型进行评估，这样以使得评估算法性能时不引入偏差 


在上一个时代的机器学习中，通常的分割法是，训练集和测试集分别占整体数据70%和30%。如果你明确地设定了开发集，那比例可能是60/20/20%，也就是测试集占60%，开发集20%，测试集20%，

数年以前这个比例被广泛认为是，机器学习中的最佳方法，如果一共只有100个样本，也许1000个样本，甚至到1万个样本时，这些比例作为最佳选择都是合理的。

但是在这个大数据的时代，趋势可能会变化，可能有多达100万个训练样本，而开发集，和测试集在总体数据中所占的比例就变小了，这是因为，**开发集存在的意义是用来，测试不同的算法并确定哪种最好**，所以开发集只要足够大到，能够用来在评估两种不同的算法，或是十种不同的算法时快速选出较好的一种，达成这个目标可能不需要多达20%的数据。所以如果有100万个训练样本，可能开发集只要1万个样本就足够了，足够用来评估两种算法中哪一种更好。与开发集相似，**测试集的主要功能是，对训练好的分类器的性能，给出可信度较高的评估**。同样如果可能有100万个样本，但是只要1万个，就足够评估单个分类器的性能，能够对其性能给出比较准确的估计了。

如果有100万个样本，而只需要1万个用作开发集，1万个用作测试集，那么1万个只是100万个的百分之一，所以比例就是98/1/1%。还有一些应用的样本可能多于100万个，分割比率可能会变成99.5/0.25/0.25%，或者开发集占0.4%，测试集占0.1%。

所以总结起来，**当设定机器学习问题时，通常将数据分为训练集，开发集和测试集。如果数据集比较小，也许就可以采用传统的分割比率，但如果数据集大了很多，那也可以使开发集，和测试集远小于总数据20%，甚至远少于10%。**

当前的深度学习中还有一个趋势是，**有越来越多的人的训练集与测试集的数据分布不匹配**。假设构建一个应用，允许用户上传大量图片，目标是找出猫的图片再展示给用户，也许因为用户都是爱猫之人，而训练集可能来自网上下载的猫的图片，而开发集和测试集则包含用户用应用上传的图片，所以，一边是训练集可能有很多从网上爬取的图片，另一边是，开发集和测试集中有许多用户上传的图片，会发现许多网页上的图片都是高分辨率，专业制作过，构图也很漂亮的猫的图片，而用户上传的图则相对比较模糊，分辨率低，用手机在更加随意的情况下拍摄的，所以这可能就造成两种不同的分布，在这种情况下建议的经验法则是，**确保开发集和测试集中的数据分布相同**。

**Make sure that the dev and test sets come from the same distribution.**

即使没有测试集也许也是可以的。测试集的目的是给你一个无偏估计，来评价最终所选取的网络的性能。但**如果不需要无偏的估计的话，没有测试集也许也没有问题**。所以当只有开发集而没有测试集的时候，所做的就是用训练集尝试不同的模型结构，然后用开发集去评估它们，根据结果进一步迭代，并尝试得到一个好的模型，因为模型拟合了开发集中的数据，所以开发集不能给无偏的估计。

#### Bias / Variance

- Bias，偏差，描述偏离度。
- Variance，方差，描述集中度。

{% asset_img bias_and_variance.jpg bias and variance  %}

坐标中的⭕和×表示训练集。

左侧是用一条直线来区分样本数据，用逻辑回归可能画出图上的这条直线，这和训练数据的拟合度并不高，这样的分类我们称之为**高偏差**。或者换一种说法，称为**欠拟合**。

相对的，右侧的曲线，如果使用一个极为复杂的分类器，或许可以像图上画的这样完美区分训练数据，但看上去也并不是一个非常好的分类算法 这个**高方差**的分类器，我们也称之为**过拟合** 。

中间的一条曲线是比较合适的。

{% asset_img bias_and_variance_example.jpg bias and variance example  %}

训练集的误差，至少可以知道算法是否可以很好的拟合训练集数据，然后总结出是否属于高偏差问题。然后通过观察同一个算法，在开发集上的误差了多少，可以知道这个算法是否有高方差问题。这样就能判断训练集上的算法是否在开发集上同样适用。上述结果都基于贝叶斯误差非常低，并且训练集和开发集都来自与同一个分布。

高偏差高方差的例子如下：

{% asset_img high_bias_and_high_variance.jpg high bias and high variance  %}

通过观察算法，在训练集和开发集的误差来诊断，它是否有高偏差或者高方差的问题，或许两者都有，或许都没有，基于算法遇到高偏差或高方差问题的不同情况，可以尝试不同的方法来进行改进。

#### Basic Recipe for Machine Learning

机器学习的基本准则如下图所示：

{% asset_img basic_recipe_for_machine_learning.jpg basic recipe for machine learning  %}


图中，找到一个新的神经网络结构，这个办法可能有效，也可能无效。把它写在括号里，是因为它是一种需要你亲自尝试的方法，也许最终能使它有效，也许不能。

但在当前这个深度学习和大数据的时代,只要能不断扩大所训练的网络的规模,只要能不断获得更多数据,虽然这两点都不是永远成立的,但如果这两点是可能的,那扩大网络几乎总是能够,减小偏差而不增大方差。只要用恰当的方式正则化的话，而获得更多数据几乎总是能够，减小方差而不增大偏差。

所以归根结底，有了这两步以后，再加上能够选取不同的网络来训练，以及获取更多数据的能力，就有了能够且只单独削减偏差或者能够并且单独削减方差，同时不会过多影响另一个指标的能力。这就是诸多原因中的一个，能够解释为何深度学习在监督学习中如此有用，以及为何在深度学习中，偏差与方差的权衡要不明显得多。这样你就不需小心地平衡两者，而是因为有了更多选择，可以单独削减偏差或单独削减方差，而不会同时增加方差或偏差。而且事实上当有了一个良好地正则化的网络时，训练一个更大的网络几乎从来没有坏处，当训练的神经网络太大时主要的代价只是计算时间。

### Regularizing your neural network

#### Regularization(正则化)

##### 逻辑回归的正则化

$$
\min_{w,b} J(w,b)
$$
正则化前：
$$
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}[\mathcal{L}(\hat{y}^{(i)}, y^{(i)})]
$$

$L_2$正则化后：
$$
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}[\mathcal{L}(\hat{y}^{(i)}, y^{(i)})] + \frac{\lambda}{2m} \parallel w \parallel_2^2
$$

其中，$\lambda$是正则化参数。

为什么只对参数w进行正则化呢? 为什么不把b的相关项也加进去呢？实际上可以这样做,但通常会把它省略掉。因为参数$w$往往是一个非常高维的参数矢量,尤其是在发生高方差问题的情况下,可能w有非常多的参数。而b只是单个数字，几乎所有的参数都集中在w中，而不是b中。即使你加上了最后这一项，实际上也不会起到太大的作用，因为b只是大量参数中的一个参数，在实践中通常就不费力气去包含它了。

$ L_2$正则化：
$$\frac{\lambda}{2m} \parallel w \parallel_2^2 = \frac{\lambda}{2m}\sum_{i=1}^{m}{w_j^2} =\frac{\lambda}{2m} w^Tw$$

$ L_2$正则化是参数矢量w的欧几里得范数的平方。

$ L_1 $正则化：
$$\frac{\lambda}{2m} \parallel w \parallel_1 = \frac{\lambda}{2m} \sum_{i=1}^{m}{|w_j|} $$

如果使用$ L_1 $正则化，最终$w$会变的稀疏（sparse），也就是包含很多0. 有些人认为这有助于压缩模型，因为有一部分参数是0，只需较少的内存来存储模型。然而在实践中发现，通过L1正则化让模型变得稀疏，带来的收效甚微。所以在压缩模型的目标上，它的作用不大，在训练网络时，L2正则化使用得频繁得多。 

> 注：$L_1$范数，表示向量中每个元素绝对值的和：$$ \parallel x \parallel_1 =  \sum_{i=1}^{m}{|x_i|} $$
$L_2$范数,也称为欧几里得距离：$$ \parallel x \parallel_2 =  \sqrt{\sum_{i=1}^{m}{x_i^2}} $$
L2范数越小，可以使得x的每个元素都很小，接近于0。在回归里面，有人把有它的回归叫“岭回归”（Ridge Regression），有人也叫它**权值衰减（Weight Decay）**。越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象。

##### 神经网络的正则化

正则化前：
$$
J(w^{[1]},b^{[1]},\cdots,w^{[l]},b^{[l]}) = -\frac{1}{m}\sum_{i=1}^{m}[\mathcal{L}(\hat{y}^{(i)}, y^{(i)})]
$$

正则化后：
$$
J(w^{[1]},b^{[1]},\cdots,w^{[l]},b^{[l]}) = -\frac{1}{m}\sum_{i=1}^{m}[\mathcal{L}(\hat{y}^{(i)}, y^{(i)})] + \frac{\lambda}{2m} \sum_{i=1}^{l} \parallel w^{[l]} \parallel_F^2
$$

其中，
$$
 \parallel w^{[l]} \parallel_F^2 = \sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^2
$$

这里矩阵范数的平方定义为,对于i和j,对矩阵中每一个元素的平方求和.如果想为这个求和加上索引，这个求和是i从1到n[l-1]，j从1到n[l]，因为w是一个n[l-1]列、n[l]行的矩阵,这些是第l-1层和第l层的隐藏单元数量单元数量或这个矩阵的范数，称为矩阵的**弗罗贝尼乌斯范数**。

$\lambda$是正则化参数。

$$ dw^{[l]} = from \,\,backpropagation + \frac{\lambda}{m} w^{[l]}$$

$$
\begin{aligned}
w^{[l]} & =w^{[l]} - \alpha dw^{[l]} \\
& = w^{[l]} - \alpha[from \,\,backpropagation + \frac{\lambda}{m} w^{[l]}
& = w^{[l]} - \alpha dw^{[l]} \\
& = (1- \frac{\lambda}{m})w^{[l]} - \alpha[from \,\,backpropagation]
\end{aligned}
$$

从上面的公式中可以看到，$w^{[l]}$项前面乘了一个小于1的数，也就是权重会减小，所以这个范数也被称为“权重衰减（weight decay）”。

#### Why regularization reduces overfitting?

**为什么正则化能够防止过拟合呢? 为什么它有助于减少方差问题?**

左边是高偏差，右边是高方差，中间的是恰好的情况。

{% asset_img bias_and_variance.jpg bias and variance  %}

关于这个问题的一个直观理解就是，如果你把正则项λ设置的很大，权重矩阵W就会被设置为非常接近0的值。因此这个直观理解就是：把很多隐藏单元的权重，设置的太接近0了而导致，一些隐藏单元的影响被消除了。如果是这种情况，那么就会使这个大大简化的神经网络变成一个很小的神经网络。事实上，这种情况与逻辑回归单元很像，但很可能是网络的深度更大了，因此这就会使，这个过拟合网络带到更加接近左边高偏差的状态。但是λ存在一个中间值，能够得到一个更加接近中间这个刚刚好的状态。


{% asset_img intuitive_understanding_of_regularizing.jpg intuitive understanding of regularizing  %}

直觉上认为上图中蓝色被覆盖的部分，这些隐藏单元的影响被完全消除了，其实并不完全正确。实际上网络仍在使用所有的隐藏单元，但每个隐藏单元的影响变得非常小了。但最终得到的这个简单的网络，看起来就像一个不容易过拟合的小型的网络。

现在通过另一个例子直观理解一下，为什么正则化可以帮助防止过拟合，在这个例子中，假设使用的，tanh激活函数。使用$g(z)$表示$tanh(z)$，因此这种情况下，可以发现只要Z的值很小，比如Z只涉及很小范围的参数，tanh曲线中中间的一小部分，那么其实是在使用tanh函数的线性的条件部分。只有Z的值被允许取到更大的值或者像这种小一点的值的时候，激活函数才开始展现出它的非线性的能力。因此直觉就是，如果λ值，即正则化参数被设置的很大的话，那么激活函数的参数实际上会变小，因为代价函数的参数会被不能过大，并且如果W很小那么由于，Z等于W这项再加上b，但如果W非常非常小，那么Z也会非常小。特别是如果Z的值相对都很小时，就在曲线的中间部分范围内取值的话，那么g(z)函数就会接近于线性函数，因此，每一层都几乎是线性的，就像线性回归一样。如果每层都是线性的，那么整个网络就是线性网络。因此即使一个很深的神经网络，如果使用线性激活函数，最终也只能计算线性的函数，因此就不能拟合那些很复杂的决策函数，也不过度拟合那些，数据集的非线性决策平面。

总结一下，如果正则化变得非常大，而参数W很小，那么Z就会相对很小。此时先暂时忽略b的影响，Z会相对变小，即Z只在小范围内取值，那么激活函数如果是tanh的话，这个激活函数就会呈现相对线性，那么整个神经网络就只能计算一些，离线性函数很近的值，也就是相对比较简单的函数，而不能计算，很复杂的非线性函数，因此就不大容易过拟合了。

#### Dropout Regularization(随机失活正则化（丢弃法）)

假设你训练下图所示的神经网络并发现过拟合现象，可以使用随机失活技术来处理它。使用随机失活技术，遍历这个网络的每一层，并且为丢弃(drop)网络中的某个节点置一个概率值（比如50%），即对于网络中的每一层，使这个节点有50%的几率被保留,50%的的几率被丢弃。最后得到的是一个小得多的、被简化了很多的网络。然后再做反向传播训练。

{% asset_img dropout_regularizaiton.jpg dropout regularizaiton  %}

##### dropout的实现：

有几种方法可以实现随机失活算法，最常用的一种是反向随机失活(inverted dropout) 。

Inverted dropout:反向随机失活的实现：（以3层网络为例）

```
layer l = 3

keep-prob = 0.8  # 保留网络中某一节点的概率值，0.8意味着layer中的节点有20%的概率被丢弃

d3 = np.random.rand(a3.shape[0],a3.shape[1] ) < keep-prob) ? 1 : 0

a3 = np.multiply(a3,d3)  # 把某些元素置为0，表示此节点被隐藏

a3 /= keep-prob 
```

最后一步的作用是"保证下一层 $Z^{[4]} = W^{[4]}  a^{[3]} + b^{[4]}$ 的期望值不会降低。

测试阶段不使用dropout。

#### Understanding Dropout

**Dropout为什么有效？**
**Can't rely on any one feature,so have to spread out weights.**

不同的层，可以设置不同的keep-prop。

**如果觉得某一层比其他层更容易发生过拟合，给这一层设置更低的保留率（keep-prop）**。这样的缺点是在交叉验证 (网格) 搜索时，会有更多的超参数 (运行会更费时)。另一个选择就是对一些层使用dropout (留存率相同)，而另一些不使用。这样的话，就只有一个超参数了。

需要记住dropout是一种正则化技术,目的是防止过拟合。所以，**除非算法已经过拟合了，否则是不会考虑使用dropout的。**

dropout的另一个缺点是让代价函数J，变得不那么明确。因为每一次迭代，都有一些神经元随机失活，所以检验梯度下降算法表现的时候，会发现很难确定代价函数是否已经定义的足够好 (随着迭代 值不断变小)。通常这个时候可以关闭dropout，把留存率设为1。然后再运行代码并确保代价函数J 是单调递减的，最后再打开dropout并期待使用dropout的时候没有引入别的错误。 

#### Other regularization methods

其他的一些正则化的方法：

- Data augmentation（数据扩增）:
    - {% asset_img data_augmentation.jpg Data augmentation  %}
    - 一只猫，那它水平翻转之后还是一只猫，随机放大图片的一部分，这很可能仍然是一只猫。对于字符识别，可以通过给数字加上随机的旋转和变形。 
- Early stoping(早终止法)
    - {% asset_img early_stoping.jpg early stoping  %}
    - 在某次次迭代附近 ，神经网络表现得最好，然后把神经网络的训练过程停住，并且选取这个(最小)开发集误差所对应的值。
    - Early stopiing有个缺点。可以把机器学习过程看作几个不同的步骤，其中之一是：需要一个算法，能够最优化成本函数J。Early stopping的主要缺点就是,它把这两个任务结合了，所以无法分开解决这两个问题。因为提早停止了梯度下降，意味着打断了优化成本函数J的过程。

### Setting up your optimization problem

#### Normalizing inputs

- 1. 均值归一：
$$ \mu = \frac{1}{m} \sum_{i=1}^m x^{(i)} \,\,\,\,\,\,\,\,\,  x= x - \mu $$
- 2. 方差归一化： 
$$ \sigma ^ {2} = \frac{1}{m} \sum_{i=1}^m (x^{(i)})^2  \,\,\,\,  x=   \frac{x}{\sigma ^2} $$

{% asset_img normalizing_training_sets.jpg Normalizing training sets  %}

**为什么Normalizing有效？**

{% asset_img why_normalize_inputs.jpg Why normalize inputs  %}

为什么要对输入特征进行归一化呢? 像上图右上方这样的代价函数，如果使用了未归一化的输入特征，代价函数看起来就会像一个压扁了的碗。 如果把这个函数的等值线画出来，就会有一个像这样的扁长的函数。而如果将特征进行归一化后，代价函数通常就会看起来更对称。

如果对左图的那种代价函数使用梯度下降法，那可能必须使用非常小的学习率 。如果等值线更趋近于圆形，那无论从哪儿开始，梯度下降法几乎都能直接朝向最小值而去，可以在梯度下降中采用更长的步长，而无需像左图那样来回摇摆缓慢挪动。

当特征的范围有非常大的不同时，譬如一个特征是1到1000，而另一个是0到1，那就会着实地削弱优化算法的表现了。但只要将它们变换，使均值皆为0，方差皆为1，就能保证所有的特征尺度都相近，这通常也能让帮助学习算法运行得更快。所以**如果输入特征的尺度非常不同，比如可能有些特征取值范围是0到1，有些是1到1000，那对输入进行归一化就很重要**。而如果输入特征本来尺度就相近，那么这一步就不那么重要。不过因为归一化的步骤几乎从来没有任何害处，所以一般总是进行归一化。 

#### Vanishing / Exploding gradients

当训练神经网络时会遇到一个问题，尤其是当训练层数非常多的神经网络时，这个问题就是**梯度消失和梯度爆炸**。它的意思是当在训练一个深度神经网络的时候，损失函数的导数，有时会变得非常大，或者非常小甚至是呈指数级减小。

{% asset_img l_layer_network.jpg  l layer network  %}

假设：$g(z)=z,b=0$

$$
\hat{y} =w^{[l]}w^{[l-1]}w^{[l-2]}\cdots w^{[2]}w^{[1]}x 
$$

假设： 

$$
w^{[l]} = \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \\ \end{bmatrix} 
$$

则 $\hat{y} = 1.5^{L}x $，如果L很大，则$\hat{y}$呈指数级增长。

反之，假设： 

$$
w^{[l]} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \\ \end{bmatrix} 
$$

则 $\hat{y} = 0.5^{L}x $，如果L很大，则$\hat{y}$会很小。

这就是梯度爆炸或者消失产生的原因。

在这么深的神经网络里，如果激活函数或梯度作为L的函数指数级的增加或减少，这些值会变得非常大或非常小，这会让训练变得非常困难。尤其是如果梯度比L要小指数级别， 梯度下降会很用很小很小步的走，梯度下降会用很长的时间才能有任何学习。

解决梯度爆炸或消失的方法是：谨慎选择初始化权重方法。

#### Weight Initialization for Deep Networks

{% asset_img single_neuron_example.jpg  single neuron example  %}

如上图的单个的神经元会输入4个特征，从x1到x4，然后用a=g(z)激活，最后得到y。

$$ z = w_1*x_1+w_2*x_2+···+w_n*x_n \,\,\,\,\, b=0$$

为了不让z项太大或者太小,那么n项的数值越大，就会希望$W_i$的值越小。因为z是$w_i*x_i$的加和。因此，如果加和了很多这样的项，就会希望每一项就尽可能地小。一个合理的做法是，让变量wi等于1/n，这里的n是指输入一个神经元的特征数。

具体做法为：

$$
W^{[l]} = np.random.rand(shape) * np.sqrt(\frac{2}{n^{[l-1]}})
$$

> 注：这个做法是针对激活函数为$Rule$的情况。

虽然这样不能完全解决问题，但它降低了梯度消失和梯度爆炸问题的程度。因为这种做法通过设置权重矩阵W，使得W不会比1大很多，也不会比1小很多。因此梯度不会过快地膨胀或者消失。

其他的激活函数的做法：

- tanh函数（Xavier初始化方法）：
$$ W^{[l]} = np.random.rand(shape) * np.sqrt(\frac{1}{n^{[l-1]}}) $$
- Rule也可以尝试如下方法：
$$ W^{[l]} = np.random.rand(shape) * np.sqrt(\frac{2}{n^{[l]}+ n^{[l-1]}}) $$


#### Gradient checking

原理是如下公式：

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$

梯度检查的流程：

- 1.Take $W^{[1]},b^{[1]},\cdots,W^{[L]},b^{[L]}$ and reshape into a big vector $\theta$
- 2.Take $dW^{[1]},db^{[1]},\cdots,dW^{[L]},db^{[L]}$ and reshape into a big vector $d\theta$
- 3.Check: Is $d\theta$ the gradient of $J(\theta)$?
    - First compute "gradapprox" using the formula above (1) and a small value of $\varepsilon$. Here are the Steps to follow:
         1. $\theta^{+} = \theta + \varepsilon$
         2. $\theta^{-} = \theta - \varepsilon$
         3. $J^{+} = J(\theta^{+})$
         4. $J^{-} = J(\theta^{-})$
         5. $gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon}$
    - Then compute the gradient using backward propagation, and store the result in a variable "grad"
    - Finally, compute the relative difference between "gradapprox" and the "grad" using the following formula:
    $$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} $$
    You will need 3 Steps to compute this formula:
        - 1'. compute the numerator using np.linalg.norm(...)
        - 2'. compute the denominator. You will need to call np.linalg.norm(...) twice.
        - 3'. divide them.
    - If this difference is small (say less than $10^{-7}$), you can be quite confident that you have computed your gradient correctly. Otherwise, there may be a mistake in the gradient computation. 
    
#### Gradient Checking Implementation Notes

- Don’t use in training – only to debug
- If algorithm fails grad check, look at components to try to identify bug.
- Remember regularization.
- Doesn’t work with dropout.
- Run at random initialization; perhaps again after some training.


## 编程练习

### Initialization

Welcome to the first assignment of "Improving Deep Neural Networks". 

Training your neural network requires specifying an initial value of the weights. A well chosen initialization method will help learning.  

If you completed the previous course of this specialization, you probably followed our instructions for weight initialization, and it has worked out so far. But how do you choose the initialization for a new neural network? In this notebook, you will see how different initializations lead to different results. 

A well chosen initialization can:
- Speed up the convergence of gradient descent
- Increase the odds of gradient descent converging to a lower training (and generalization) error 

To get started, run the following cell to load the packages and the planar dataset you will try to classify.

```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
```

You would like a classifier to separate the blue dots from the red dots.

#### 1 - Neural Network model 

You will use a 3-layer neural network (already implemented for you). Here are the initialization methods you will experiment with:  
- *Zeros initialization* --  setting `initialization = "zeros"` in the input argument.
- *Random initialization* -- setting `initialization = "random"` in the input argument. This initializes the weights to large random values.  
- *He initialization* -- setting `initialization = "he"` in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015. 

**Instructions**: Please quickly read over the code below, and run it. In the next part you will implement the three initialization methods that this `model()` calls.

```python
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

#### 2 - Zero initialization

There are two types of parameters to initialize in a neural network:
- the weight matrices $(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
- the bias vectors $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$

**Exercise**: Implement the following function to initialize all parameters to zeros. You'll see later that this does not work well since it fails to "break symmetry", but lets try it anyway and see what happens. Use np.zeros((..,..)) with the correct shapes.

```python
# GRADED FUNCTION: initialize_parameters_zeros 

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###
    return parameters
```

Run the following code to train your model on 15,000 iterations using zeros initialization.

```python
parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

```
Cost after iteration 0: 0.6931471805599453
Cost after iteration 1000: 0.6931471805599453
Cost after iteration 2000: 0.6931471805599453
Cost after iteration 3000: 0.6931471805599453
Cost after iteration 4000: 0.6931471805599453
Cost after iteration 5000: 0.6931471805599453
Cost after iteration 6000: 0.6931471805599453
Cost after iteration 7000: 0.6931471805599453
Cost after iteration 8000: 0.6931471805599453
Cost after iteration 9000: 0.6931471805599453
Cost after iteration 10000: 0.6931471805599455
Cost after iteration 11000: 0.6931471805599453
Cost after iteration 12000: 0.6931471805599453
Cost after iteration 13000: 0.6931471805599453
Cost after iteration 14000: 0.6931471805599453
```
{% asset_img output_of_init_zerosk.jpg %}
```
On the train set:
Accuracy: 0.5
On the test set:
Accuracy: 0.5
```

The performance is really bad, and the cost does not really decrease, and the algorithm performs no better than random guessing. Why? Lets look at the details of the predictions and the decision boundary:

```python
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))
```

outputs:

```
predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0]]
predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
```


```python
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
{% asset_img output_of_init_zerosk.jpg %}


The model is predicting 0 for every example. 

In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with $n^{[l]}=1$ for every layer, and the network is no more powerful than a linear classifier such as logistic regression. 

<font color='blue'>
**What you should remember**:
- The weights $W^{[l]}$ should be initialized randomly to break symmetry. 
- It is however okay to initialize the biases $b^{[l]}$ to zeros. Symmetry is still broken so long as $W^{[l]}$ is initialized randomly. 
</font>

#### 3 - Random initialization

To break symmetry, lets intialize the weights randomly. Following random initialization, each neuron can then proceed to learn a different function of its inputs. In this exercise, you will see what happens if the weights are intialized randomly, but to very large values. 

**Exercise**: Implement the following function to initialize your weights to large random values (scaled by \*10) and your biases to zeros. Use `np.random.randn(..,..) * 10` for weights and `np.zeros((.., ..))` for biases. We are using a fixed `np.random.seed(..)` to make sure your "random" weights  match ours, so don't worry if running several times your code gives you always the same initial values for the parameters. 

```python
# GRADED FUNCTION: initialize_parameters_random

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10 
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###

    return parameters
```

```python
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

```
W1 = [[ 17.88628473   4.36509851   0.96497468]
 [-18.63492703  -2.77388203  -3.54758979]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[-0.82741481 -6.27000677]]
b2 = [[ 0.]]
```

Run the following code to train your model on 15,000 iterations using random initialization.

```python
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

**outputs:**

```
Cost after iteration 0: inf
Cost after iteration 1000: 0.6237287551108738
Cost after iteration 2000: 0.5981106708339466
Cost after iteration 3000: 0.5638353726276827
Cost after iteration 4000: 0.550152614449184
Cost after iteration 5000: 0.5444235275228304
Cost after iteration 6000: 0.5374184054630083
Cost after iteration 7000: 0.47357131493578297
Cost after iteration 8000: 0.39775634899580387
Cost after iteration 9000: 0.3934632865981078
Cost after iteration 10000: 0.39202525076484457
Cost after iteration 11000: 0.38921493051297673
Cost after iteration 12000: 0.38614221789840486
Cost after iteration 13000: 0.38497849983013926
Cost after iteration 14000: 0.38278397192120406
```
{% asset_img output_of_init_with_random.jpg %}
```
On the train set:
Accuracy: 0.83
On the test set:
Accuracy: 0.86
```

If you see "inf" as the cost after the iteration 0, this is because of numerical roundoff; a more numerically sophisticated implementation would fix this. But this isn't worth worrying about for our purposes. 

Anyway, it looks like you have broken symmetry, and this gives better results. than before. The model is no longer outputting all 0s. 

```python
print (predictions_train)
print (predictions_test)
```

***outputs:**
```
[[1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1 1
  1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 0 0 0
  0 0 1 0 1 0 1 1 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0 1 0 1
  1 0 0 1 0 0 1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0 0 0 1 0
  1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 0 1
  0 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 0 1 0 1 1 0 1 1
  0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 0 0 0 1 1 0 1 1 1 1 0 1 1 0 1
  1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 1
  1 1 1 0]]
[[1 1 1 1 0 1 0 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 0 1 0
  1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1
  1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0]]
```

```python
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

{% asset_img model_with_large_random_initialization.jpg %}


**Observations**:
- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when $\log(a^{[3]}) = \log(0)$, the loss goes to infinity.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm. 
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.

<font color='blue'>
**In summary**:
- Initializing weights to very large random values does not work well. 
- Hopefully intializing with small random values does better. The important question is: how small should be these random values be? Lets find out in the next part! 
</font>

#### 4 - He initialization

Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights $W^{[l]}$ of `sqrt(1./layers_dims[l-1])` where He initialization would use `sqrt(2./layers_dims[l-1])`.)

**Exercise**: Implement the following function to initialize your parameters with He initialization.

**Hint**: This function is similar to the previous `initialize_parameters_random(...)`. The only difference is that instead of multiplying `np.random.randn(..,..)` by 10, you will multiply it by $\sqrt{\frac{2}{\text{dimension of the previous layer}}}$, which is what He initialization recommends for layers with a ReLU activation. 

```python
#GRADED FUNCTION: initialize_parameters_he

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2. / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###
        
    return parameters
```

```python
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

```
W1 = [[ 1.78862847  0.43650985]
 [ 0.09649747 -1.8634927 ]
 [-0.2773882  -0.35475898]
 [-0.08274148 -0.62700068]]
b1 = [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
b2 = [[ 0.]]
```

```python
parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```
**outputs:**

```
Cost after iteration 0: 0.8830537463419761
Cost after iteration 1000: 0.6879825919728063
Cost after iteration 2000: 0.6751286264523371
Cost after iteration 3000: 0.6526117768893807
Cost after iteration 4000: 0.6082958970572938
Cost after iteration 5000: 0.5304944491717495
Cost after iteration 6000: 0.4138645817071794
Cost after iteration 7000: 0.3117803464844441
Cost after iteration 8000: 0.23696215330322562
Cost after iteration 9000: 0.18597287209206836
Cost after iteration 10000: 0.1501555628037182
Cost after iteration 11000: 0.12325079292273548
Cost after iteration 12000: 0.09917746546525937
Cost after iteration 13000: 0.0845705595402428
Cost after iteration 14000: 0.07357895962677366
```

{% asset_img output_of_init_with_he_init.jpg %}

```
On the train set:
Accuracy: 0.993333333333
On the test set:
Accuracy: 0.96
```

```python
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

{% asset_img Model_with_He_initialization.jpg %}

**Observations**:
- The model with He initialization separates the blue and the red dots very well in a small number of iterations.


#### 5 - Conclusions

You have seen three different types of initializations. For the same number of iterations and same hyperparameters the comparison is:

<table> 
    <tr>
        <td>
        **Model**
        </td>
        <td>
        **Train accuracy**
        </td>
        <td>
        **Problem/Comment**
        </td>
    </tr>
        <td>
        3-layer NN with zeros initialization
        </td>
        <td>
        50%
        </td>
        <td>
        fails to break symmetry
        </td>
    <tr>
        <td>
        3-layer NN with large random initialization
        </td>
        <td>
        83%
        </td>
        <td>
        too large weights 
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with He initialization
        </td>
        <td>
        99%
        </td>
        <td>
        recommended method
        </td>
    </tr>
</table> 

<font color='blue'>
**What you should remember from this notebook**:
- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations. 
</font>

### Regularization

Welcome to the second assignment of this week. Deep Learning models have so much flexibility and capacity that **overfitting can be a serious problem**, if the training dataset is not big enough. Sure it does well on the training set, but the learned network **doesn't generalize to new examples** that it has never seen!

**You will learn to:** Use regularization in your deep learning models.

Let's first import the packages you are going to use.

```python
import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

**Problem Statement**: You have just been hired as an AI expert by the French Football Corporation. They would like you to recommend positions where France's goal keeper should kick the ball so that the French team's players can then hit it with their head. 

{% asset_img field_kiank.png Figure 1 : Football field.The goal keeper kicks the ball in the air, the players of each team are fighting to hit the ball with their head %}


They give you the following 2D dataset from France's past 10 games.

```python
train_X, train_Y, test_X, test_Y = load_2D_dataset()
```

{% asset_img football_dataset.jpg %}

Each dot corresponds to a position on the football field where a football player has hit the ball with his/her head after the French goal keeper has shot the ball from the left side of the football field.
- If the dot is blue, it means the French player managed to hit the ball with his/her head
- If the dot is red, it means the other team's player hit the ball with their head

**Your goal**: Use a deep learning model to find the positions on the field where the goalkeeper should kick the ball.

**Analysis of the dataset**: This dataset is a little noisy, but it looks like a diagonal line separating the upper left half (blue) from the lower right half (red) would work well. 

You will first try a non-regularized model. Then you'll learn how to regularize it and decide which model you will choose to solve the French Football Corporation's problem. 

#### 1 - Non-regularized model

You will use the following neural network (already implemented for you below). This model can be used:
- in *regularization mode* -- by setting the `lambd` input to a non-zero value. We use "`lambd`" instead of "`lambda`" because "`lambda`" is a reserved keyword in Python. 
- in *dropout mode* -- by setting the `keep_prob` to a value less than one

You will first try the model without any regularization. Then, you will implement:
- *L2 regularization* -- functions: "`compute_cost_with_regularization()`" and "`backward_propagation_with_regularization()`"
- *Dropout* -- functions: "`forward_propagation_with_dropout()`" and "`backward_propagation_with_dropout()`"

In each part, you will run this model with the correct inputs so that it calls the functions you've implemented. Take a look at the code below to familiarize yourself with the model.

```python
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

Let's train the model without any regularization, and observe the accuracy on the train/test sets.

```python
parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

```
Cost after iteration 0: 0.6557412523481002
Cost after iteration 10000: 0.16329987525724216
Cost after iteration 20000: 0.13851642423255986
```
{% asset_img non_regularized_model_output.jpg %}
```
On the training set:
Accuracy: 0.947867298578
On the test set:
Accuracy: 0.915
```

The train accuracy is 94.8% while the test accuracy is 91.5%. This is the **baseline model** (you will observe the impact of regularization on this model). Run the following code to plot the decision boundary of your model.

```python
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

{% asset_img model_without_regularization.jpg %}

The non-regularized model is obviously overfitting the training set. It is fitting the noisy points! Lets now look at two techniques to reduce overfitting.

#### 2 - L2 Regularization

The standard way to avoid overfitting is called **L2 regularization**. It consists of appropriately modifying your cost function, from:
$$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} $$
To:
$$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} $$

Let's modify your cost and observe the consequences.

**Exercise**: Implement `compute_cost_with_regularization()` which computes the cost given by formula (2). To calculate $\sum\limits_k\sum\limits_j W_{k,j}^{[l]2}$  , use :
```python
np.sum(np.square(Wl))
```
Note that you have to do this for $W^{[1]}$, $W^{[2]}$ and $W^{[3]}$, then sum the three terms and multiply by $ \frac{1}{m} \frac{\lambda}{2} $.

```python
# GRADED FUNCTION: compute_cost_with_regularization

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = lambd / 2 / m * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    ### END CODER HERE ###
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
```

```python
A3, Y_assess, parameters = compute_cost_with_regularization_test_case()

print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))
```

    cost = 1.78648594516

Of course, because you changed the cost, you have to change backward propagation as well! All the gradients have to be computed with respect to this new cost. 

**Exercise**: Implement the changes needed in backward propagation to take into account regularization. The changes only concern dW1, dW2 and dW3. For each, you have to add the regularization term's gradient ($\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$).

```python
# GRADED FUNCTION: backward_propagation_with_regularization

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd / m * W3
    ### END CODE HERE ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1./m * np.dot(dZ2, A1.T) +  lambd / m * W2
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1./m * np.dot(dZ1, X.T) +  lambd / m * W1
    ### END CODE HERE ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

```python
X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
print ("dW1 = "+ str(grads["dW1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("dW3 = "+ str(grads["dW3"]))
```

**Output**:

```
dW1 = [[-0.25604646  0.12298827 -0.28297129]
 [-0.17706303  0.34536094 -0.4410571 ]]
dW2 = [[ 0.79276486  0.85133918]
 [-0.0957219  -0.01720463]
 [-0.13100772 -0.03750433]]
dW3 = [[-1.77691347 -0.11832879 -0.09397446]]
```

Let's now run the model with L2 regularization $(\lambda = 0.7)$. The `model()` function will call: 
- `compute_cost_with_regularization` instead of `compute_cost`
- `backward_propagation_with_regularization` instead of `backward_propagation`

```python
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

**Output**:

```
Cost after iteration 0: 0.6974484493131264
Cost after iteration 10000: 0.2684918873282239
Cost after iteration 20000: 0.2680916337127301
```
{% asset_img l2_regularization_model_output.jp.jpg %}
```
On the train set:
Accuracy: 0.938388625592
On the test set:
Accuracy: 0.93
```
Congrats, the test set accuracy increased to 93%. You have saved the French football team!

You are not overfitting the training data anymore. Let's plot the decision boundary.

```python
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

{% asset_img model_with_l2-regularization.jpg %}


**Observations**:
- The value of $\lambda$ is a hyperparameter that you can tune using a dev set.
- L2 regularization makes your decision boundary smoother. If $\lambda$ is too large, it is also possible to "oversmooth", resulting in a model with high bias.

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes. 

<font color='blue'>
**What you should remember** -- the implications of L2-regularization on:
- The cost computation:
    - A regularization term is added to the cost
- The backpropagation function:
    - There are extra terms in the gradients with respect to weight matrices
- Weights end up smaller ("weight decay"): 
    - Weights are pushed to smaller values.
</font>

#### 3 - Dropout

Finally, **dropout** is a widely used regularization technique that is specific to deep learning. 
**It randomly shuts down some neurons in each iteration.** Watch these two videos to see what this means!

<!--
To understand drop-out, consider this conversation with a friend:
- Friend: "Why do you need all these neurons to train your network and classify images?". 
- You: "Because each neuron contains a weight and can learn specific features/details/shape of an image. The more neurons I have, the more featurse my model learns!"
- Friend: "I see, but are you sure that your neurons are learning different features and not all the same features?"
- You: "Good point... Neurons in the same layer actually don't talk to each other. It should be definitly possible that they learn the same image features/shapes/forms/details... which would be redundant. There should be a solution."
!--> 


{% asset_img dropout1_kiank.gif  Figure 2 : Drop-out on the second hidden layer. %}

At each iteration, you shut down (= set to zero) each neuron of a layer with probability $1 - keep\_prob$ or keep it with probability $keep\_prob$ (50% here). The dropped neurons don't contribute to the training in both the forward and backward propagations of the iteration. 


{% asset_img dropout2_kiank.gif  Figure 3 : Drop-out on the first and third hidden layers. %}

$1^{st}$ layer: we shut down on average 40% of the neurons.  $3^{rd}$ layer: we shut down on average 20% of the neurons. 

When you shut some neurons down, you actually modify your model. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time. 

##### 3.1 - Forward propagation with dropout

**Exercise**: Implement the forward propagation with dropout. You are using a 3 layer neural network, and will add dropout to the first and second hidden layers. We will not apply dropout to the input layer or output layer. 

**Instructions**:
You would like to shut down some neurons in the first and second layers. To do that, you are going to carry out 4 Steps:
1. In lecture, we dicussed creating a variable $d^{[1]}$ with the same shape as $a^{[1]}$ using `np.random.rand()` to randomly get numbers between 0 and 1. Here, you will use a vectorized implementation, so create a random matrix $D^{[1]} = [d^{[1](1)} d^{[1](2)} ... d^{[1](m)}] $ of the same dimension as $A^{[1]}$.
2. Set each entry of $D^{[1]}$ to be 0 with probability (`1-keep_prob`) or 1 with probability (`keep_prob`), by thresholding values in $D^{[1]}$ appropriately. Hint: to set all the entries of a matrix X to 0 (if entry is less than 0.5) or 1 (if entry is more than 0.5) you would do: `X = (X < 0.5)`. Note that 0 and 1 are respectively equivalent to False and True.
3. Set $A^{[1]}$ to $A^{[1]} * D^{[1]}$. (You are shutting down some neurons). You can think of $D^{[1]}$ as a mask, so that when it is multiplied with another matrix, it shuts down some of the values.
4. Divide $A^{[1]}$ by `keep_prob`. By doing this you are assuring that the result of the cost will still have the same expected value as without drop-out. (This technique is also called inverted dropout.)

```python
#GRADED FUNCTION: forward_propagation_with_dropout

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.rand(A1.shape[0],A1.shape[1])                                         # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                                        # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1                                         # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob                                         # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])                                         # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                                           # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2                                         # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                                         # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache
```

```python

X_assess, parameters = forward_propagation_with_dropout_test_case()

A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
print ("A3 = " + str(A3))
```

**Output**: 

    A3 = [[ 0.36974721  0.00305176  0.04565099  0.49683389  0.36974721]]

#####3.2 - Backward propagation with dropout

**Exercise**: Implement the backward propagation with dropout. As before, you are training a 3 layer network. Add dropout to the first and second hidden layers, using the masks $D^{[1]}$ and $D^{[2]}$ stored in the cache. 

**Instruction**:
Backpropagation with dropout is actually quite easy. You will have to carry out 2 Steps:
1. You had previously shut down some neurons during forward propagation, by applying a mask $D^{[1]}$ to `A1`. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask $D^{[1]}$ to `dA1`. 
2. During forward propagation, you had divided `A1` by `keep_prob`. In backpropagation, you'll therefore have to divide `dA1` by `keep_prob` again (the calculus interpretation is that if $A^{[1]}$ is scaled by `keep_prob`, then its derivative $dA^{[1]}$ is also scaled by the same `keep_prob`).


```python
# GRADED FUNCTION: backward_propagation_with_dropout

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

```python
X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)

print ("dA1 = " + str(gradients["dA1"]))
print ("dA2 = " + str(gradients["dA2"]))
```

**Output**: 

```
dA1 = [[ 0.36544439  0.         -0.00188233  0.         -0.17408748]
 [ 0.65515713  0.         -0.00337459  0.         -0.        ]]
dA2 = [[ 0.58180856  0.         -0.00299679  0.         -0.27715731]
 [ 0.          0.53159854 -0.          0.53159854 -0.34089673]
 [ 0.          0.         -0.00292733  0.         -0.        ]]
```

Let's now run the model with dropout (`keep_prob = 0.86`). It means at every iteration you shut down each neurons of layer 1 and 2 with 14% probability. The function `model()` will now call:
- `forward_propagation_with_dropout` instead of `forward_propagation`.
- `backward_propagation_with_dropout` instead of `backward_propagation`.

```python
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

**Output**

```
Cost after iteration 0: 0.6543912405149825
Cost after iteration 10000: 0.06101698657490559
Cost after iteration 20000: 0.060582435798513114
```
{% asset_img dropout_model_output.jpg %}
```
On the train set:
Accuracy: 0.928909952607
On the test set:
Accuracy: 0.95
```
Dropout works great! The test accuracy has increased again (to 95%)! Your model is not overfitting the training set and does a great job on the test set. The French football team will be forever grateful to you! 

Run the code below to plot the decision boundary.

```pytho

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```
{% asset_img model_with_dropout.jpg %}

**Note**:
- A **common mistake** when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training. 
- Deep learning frameworks like [tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout), [PaddlePaddle](http://doc.paddlepaddle.org/release_doc/0.9.0/doc/ui/api/trainer_config_helpers/attrs.html), [keras](https://keras.io/layers/core/#dropout) or [caffe](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html) come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.

<font color='blue'>
**What you should remember about dropout:**
- Dropout is a regularization technique.
- You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
- Apply dropout both during forward and backward propagation.
- During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.  
</font>

#### 4 - Conclusions

**Here are the results of our three models**: 

<table> 
    <tr>
        <td>
        **model**
        </td>
        <td>
        **train accuracy**
        </td>
        <td>
        **test accuracy**
        </td>
    </tr>
        <td>
        3-layer NN without regularization
        </td>
        <td>
        95%
        </td>
        <td>
        91.5%
        </td>
    <tr>
        <td>
        3-layer NN with L2-regularization
        </td>
        <td>
        94%
        </td>
        <td>
        93%
        </td>
    </tr>
    <tr>
        <td>
        3-layer NN with dropout
        </td>
        <td>
        93%
        </td>
        <td>
        95%
        </td>
    </tr>
</table> 

Note that regularization hurts training set performance! This is because it limits the ability of the network to overfit to the training set. But since it ultimately gives better test accuracy, it is helping your system. 

Congratulations for finishing this assignment! And also for revolutionizing French football. :-) 

<font color='blue'>
**What we want you to remember from this notebook**:
- Regularization will help you reduce overfitting.
- Regularization will drive your weights to lower values.
- L2 regularization and Dropout are two very effective regularization techniques.
</font>

### Gradient Checking

Welcome to the final assignment for this week! In this assignment you will learn to implement and use gradient checking. 

You are part of a team working to make mobile payments available globally, and are asked to build a deep learning model to detect fraud--whenever someone makes a payment, you want to see if the payment might be fraudulent, such as if the user's account has been taken over by a hacker. 

But backpropagation is quite challenging to implement, and sometimes has bugs. Because this is a mission-critical application, your company's CEO wants to be really certain that your implementation of backpropagation is correct. Your CEO says, "Give me a proof that your backpropagation is actually working!" To give this reassurance, you are going to use "gradient checking".

Let's do it!

```python
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
```

#### 1) How does gradient checking work?

Backpropagation computes the gradients $\frac{\partial J}{\partial \theta}$, where $\theta$ denotes the parameters of the model. $J$ is computed using forward propagation and your loss function.

Because forward propagation is relatively easy to implement, you're confident you got that right, and so you're almost  100% sure that you're computing the cost $J$ correctly. Thus, you can use your code for computing $J$ to verify the code for computing $\frac{\partial J}{\partial \theta}$. 

Let's look back at the definition of a derivative (or gradient):
$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$

If you're not familiar with the "$\displaystyle \lim_{\varepsilon \to 0}$" notation, it's just a way of saying "when $\varepsilon$ is really really small."

We know the following:

- $\frac{\partial J}{\partial \theta}$ is what you want to make sure you're computing correctly. 
- You can compute $J(\theta + \varepsilon)$ and $J(\theta - \varepsilon)$ (in the case that $\theta$ is a real number), since you're confident your implementation for $J$ is correct. 

Lets use equation (1) and a small value for $\varepsilon$ to convince your CEO that your code for computing  $\frac{\partial J}{\partial \theta}$ is correct!

#### 2) 1-dimensional gradient checking

Consider a 1D linear function $J(\theta) = \theta x$. The model contains only a single real-valued parameter $\theta$, and takes $x$ as input.

You will implement code to compute $J(.)$ and its derivative $\frac{\partial J}{\partial \theta}$. You will then use gradient checking to make sure your derivative computation for $J$ is correct. 

{% asset_img 1Dgrad_kiank.png  Figure 1: 1D linear model  %}

The diagram above shows the key computation steps: First start with $x$, then evaluate the function $J(x)$ ("forward propagation"). Then compute the derivative $\frac{\partial J}{\partial \theta}$ ("backward propagation"). 

**Exercise**: implement "forward propagation" and "backward propagation" for this simple function. I.e., compute both $J(.)$ ("forward propagation") and its derivative with respect to $\theta$ ("backward propagation"), in two separate functions. 

```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    
    ### START CODE HERE ### (approx. 1 line)
    J = x * theta
    ### END CODE HERE ###
    
    return J
```

**Exercise**: Now, implement the backward propagation step (derivative computation) of Figure 1. That is, compute the derivative of $J(\theta) = \theta x$ with respect to $\theta$. To save you from doing the calculus, you should get $dtheta = \frac { \partial J }{ \partial \theta} = x$.


```python
# GRADED FUNCTION: backward_propagation

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    
    ### START CODE HERE ### (approx. 1 line)
    dtheta = x
    ### END CODE HERE ###
    
    return dtheta
```


**Exercise**: To show that the `backward_propagation()` function is correctly computing the gradient $\frac{\partial J}{\partial \theta}$, let's implement gradient checking.

**Instructions**:
- First compute "gradapprox" using the formula above (1) and a small value of $\varepsilon$. Here are the Steps to follow:
    1. $\theta^{+} = \theta + \varepsilon$
    2. $\theta^{-} = \theta - \varepsilon$
    3. $J^{+} = J(\theta^{+})$
    4. $J^{-} = J(\theta^{-})$
    5. $gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon}$
- Then compute the gradient using backward propagation, and store the result in a variable "grad"
- Finally, compute the relative difference between "gradapprox" and the "grad" using the following formula:
$$ difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} $$
You will need 3 Steps to compute this formula:
   - 1'. compute the numerator using np.linalg.norm(...)
   - 2'. compute the denominator. You will need to call np.linalg.norm(...) twice.
   - 3'. divide them.
- If this difference is small (say less than $10^{-7}$), you can be quite confident that you have computed your gradient correctly. Otherwise, there may be a mistake in the gradient computation. 


```python
#GRADED FUNCTION: gradient_check

def gradient_check(x, theta, epsilon = 1e-7):
    """
    Implement the backward propagation presented in Figure 1.
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    ### START CODE HERE ### (approx. 5 lines)
    thetaplus = theta + epsilon                               # Step 1
    thetaminus = theta - epsilon                              # Step 2
    J_plus = forward_propagation(x, thetaplus)                                  # Step 3
    J_minus = forward_propagation(x, thetaminus)                                    # Step 4
    gradapprox = (J_plus - J_minus) / (2 * epsilon)                              # Step 5
    ### END CODE HERE ###
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    ### START CODE HERE ### (approx. 1 line)
    grad = backward_propagation(x, theta)
    ### END CODE HERE ###
    
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                                # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                             # Step 2'
    difference = numerator / denominator                              # Step 3'
    ### END CODE HERE ###
    
    if difference < 1e-7:
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    
    return difference
```

```python
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))
```

**Expected Output**:

    The gradient is correct!
    difference
         2.9193358103083e-10


Congrats, the difference is smaller than the $10^{-7}$ threshold. So you can have high confidence that you've correctly computed the gradient in `backward_propagation()`. 

Now, in the more general case, your cost function $J$ has more than a single 1D input. When you are training a neural network, $\theta$ actually consists of multiple matrices $W^{[l]}$ and biases $b^{[l]}$! It is important to know how to do a gradient check with higher-dimensional inputs. Let's do it!

#### 3) N-dimensional gradient checking

The following figure describes the forward and backward propagation of your fraud detection model.

{% asset_img NDgrad_kiank.png  Figure 2: deep neural network[LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID]  %}


Let's look at your implementations for forward propagation and backward propagation. 

```python
def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    
    Returns:
    cost -- the cost function (logistic cost for one example)
    """
    
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache
```

Now, run backward propagation.


```python
def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()
    
    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

You obtained some results on the fraud detection test set but you are not 100% sure of your model. Nobody's perfect! Let's implement gradient checking to verify if your gradients are correct.

**How does gradient checking work?**.

As in 1) and 2), you want to compare "gradapprox" to the gradient computed by backpropagation. The formula is still:

$$ \frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} $$

However, $\theta$ is not a scalar anymore. It is a dictionary called "parameters". We implemented a function "`dictionary_to_vector()`" for you. It converts the "parameters" dictionary into a vector called "values", obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them.

The inverse function is "`vector_to_dictionary`" which outputs back the "parameters" dictionary.

{% asset_img dictionary_to_vector.png  Figure 3: dictionary_to_vector() and vector_to_dictionary().You will need these functions in gradient_check_n()  %}


We have also converted the "gradients" dictionary into a vector "grad" using gradients_to_vector(). You don't need to worry about that.

**Exercise**: Implement gradient_check_n().

**Instructions**: Here is pseudo-code that will help you implement the gradient check.

For each i in num_parameters:
- To compute `J_plus[i]`:
    1. Set $\theta^{+}$ to `np.copy(parameters_values)`
    2. Set $\theta^{+}_i$ to $\theta^{+}_i + \varepsilon$
    3. Calculate $J^{+}_i$ using to `forward_propagation_n(x, y, vector_to_dictionary(`$\theta^{+}$ `))`.     
- To compute `J_minus[i]`: do the same thing with $\theta^{-}$
- Compute $gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon}$

Thus, you get a vector gradapprox, where gradapprox[i] is an approximation of the gradient with respect to `parameter_values[i]`. You can now compare this gradapprox vector to the gradients vector from backpropagation. Just like for the 1D case (Steps 1', 2', 3'), compute: 
$$ difference = \frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 } $$


```python
# GRADED FUNCTION: gradient_check_n

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus =  np.copy(parameters_values)                                      # Step 1
        thetaplus[i][0] =  thetaplus[i][0] + epsilon                               # Step 2
        J_plus[i], _ =  forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))                                   # Step 3
        ### END CODE HERE ###
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)                                      # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                # Step 2        
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))                                   # Step 3
        ### END CODE HERE ###
        
        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        ### END CODE HERE ###
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                                           # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                                         # Step 2'
    difference = numerator / denominator                                          # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
```


```python
X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)
```

**Expected output**:

    There is a mistake in the backward propagation!
        difference = 0.285093156781

It seems that there were errors in the `backward_propagation_n` code we gave you! Good that you've implemented the gradient check. Go back to `backward_propagation` and try to find/correct the errors *(Hint: check dW2 and db1)*. Rerun the gradient check when you think you've fixed it. Remember you'll need to re-execute the cell defining `backward_propagation_n()` if you modify the code. 

Can you get gradient check to declare your derivative computation correct? Even though this part of the assignment isn't graded, we strongly urge you to try to find the bug and re-run gradient check until you're convinced backprop is now correctly implemented. 

**Note** 
- Gradient Checking is slow! Approximating the gradient with $\frac{\partial J}{\partial \theta} \approx  \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$ is computationally costly. For this reason, we don't run gradient checking at every iteration during training. Just a few times to check if the gradient is correct. 
- Gradient Checking, at least as we've presented it, doesn't work with dropout. You would usually run the gradient check algorithm without dropout to make sure your backprop is correct, then add dropout. 

Congrats, you can be confident that your deep learning model for fraud detection is working correctly! You can even use this to convince your CEO. :) 

<font color='blue'>

**What you should remember from this notebook**:
- Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
- Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process. 

</font>
