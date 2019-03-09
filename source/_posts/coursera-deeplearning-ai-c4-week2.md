---
title: coursera-deeplearning-ai-c4-week2
date: 2018-10-25 07:16:37
tags:
---

## 课程笔记

本周课程主要讲了集中常见的卷积神经网络，比如ResNet，同时讲解了如何使用keras构建神经网络。要点如下：

- 几篇重要的卷积神经网络论文
- 深度神经网络中维度的变化
- Residual Network
- Keras

**学习目标**

- Understand multiple foundational papers of convolutional neural networks
- Analyze the dimensionality reduction of a volume in a very deep network
- Understand and Implement a Residual network
- Build a deep neural network using Keras
- Implement a skip-connection in your network

### Case studies

#### Why look at case studies?

课程大纲：

- Classic networks:
    - LeNet-5
    - AlexNet
    - VGG
- ResNet
- Inception

#### Classic Networks

**Lenet-5**
{% asset_img  lenet_5.jpg LeNet-5 %}


**AlexNet**
{% asset_img  alexnet.jpg AlexNet %}

**VGG-16**
{% asset_img  vgg_16.jpg VGG-16 %}


#### ResNets

太深的神经网络训练起来很难，因为有”梯度消失和爆炸“这类问题。 本节课学习 ”跳跃连接（skip connection）“，它能从一层中得到激活 并把它递给下一层。在更深的神经网络层中使用它，就可以训练网络层很深很深的残差网络(ResNet)。 有时甚至可以超过100层。

残差网络（Residual Network）是指包含残差块(Residual Block)的神经网络。

残差块的结构如下图所示，输入$a^{[l]}$不仅对其直接连接的节点$a^{[l+1]}$有影响，还对下一个节点$a^{[l+2]}$有影响。

{% asset_img  residual_block.jpg Residual Block %}

图中，正常的路径被称为"main path"，上面一条路径被称为"short cut"或者"skip connection"（跳跃连接）。

在深度神经网络中使用残差块，可以解决深层网络中梯度爆炸或消失的问题。下图中的两个曲线分别是正常神经网络和残差神经网络训练误差随着和网络层数的关系。

对正常网络，理论上曲线应该是递减下降的，实际上却是当网络层数达到一定值时，训练误差会有所上升。使用残差神经网络则不会有这种问题。

{% asset_img  residual_network.jpg Residual Network %}

#### Why ResNets Work

**残差块比较容易学习恒等函数。**

由于跳跃连接很容易得到a[l+2]等于a[l]。 这意味着将这两层加入到神经网络中，与上面没有这两层的网络相比，并不会非常影响神经网络的能力， 因为对于它来说学习恒等函数非常容易， 只需要复制a[l]到a[l+2]，即便它中间有额外两层。 所以这就是为什么添加这额外的两层， 这个残差块到大型神经网络, 中间或者尾部并不会影响神经网络的表现。 

残差块网络有效的主要原因是,这些额外层学习起恒等函数非常简单， 几乎总能保证它不会影响总体的表现， 甚至许多时候幸运的话可以提升网络的表现。 至少有个合理的底线， 不会影响其表现。

{% asset_img  why_do_redidual_networks_work.jpg Why do residual Networks work? %}

resnet的应用：

{% asset_img  resnet_com_plain.jpg ResNet %}

这里使用的卷积大部分是3*3 same的，是为了保持a[l+2]和a[l]的尺寸相同。

#### Networks in Networks and 1x1 Convolutions

1 * 1的卷积的主要用途是：改变channel数。

H和W可以通过pooling和k\*k卷积（k>1）来改变，而改变channel可以通过1\*1卷积来改变。

{% asset_img  one_by_one_conv.jpg 1*1 Conv %}


 1\*1卷积本质上是一个完全连接的神经网络。也称为1\*1卷积 但有时也被称为网中网(Network in Network)。在inception network中广泛应用。1\*1卷积是一个非常不平凡的操作，用它可以缩小输入的通道数，或不改变它，甚至增加它。
下载

#### Inception Network Motivation

为卷积网络设计某一层时，可能需要选择， 是用一个1x1的卷积核 或者3x3，或5x5 或者还想用一个池化层？inception网络是说为什么不全用呢？这使得网络架构更加复杂， 但它的效果也变得更好。

下面是一个Inception层的示例，在一层，1x1，3x3，5x5的卷积核和pooling全部都用上了。每个操作都得到一个结果，把这些结果拼装起来就是最终的结果。

{% asset_img motivation_for_inception_network.jpg %}

但是这里会出现一个关于inception network的问题： 计算成本问题。

如下图所示的计算，使用5*5的卷积核，产生的计算量是120M。这个计算量是相当大了。

{% asset_img the_problem_of_computational_cost.jpg %}

而使用下图中的计算方式，先使用1\*1卷积核，在使用5\*5卷积核，得到的结果相同，但是计算量却只有12.4M，只有上面的十分之一。

通过这种方式，可以减少计算量。

{% asset_img using_one_by_one_conv.jpg Using 1*1 Conv %}

#### Inception Network

上一节讲到了Inception层，由于存在计算量过大的问题，进行了优化。把Inception层优化后，就得到下图的**Inception Module**。

Inception Module充分利用1\*1卷积，既可以减少计算量，又可以使得网络的层更深。

{% asset_img inception_module.jpg Inception module %}

把上面的Inception Module进行组合，即得到**Inception Network**。

{% asset_img inception_network.png Inception Network %}

有趣的是，这个网络是由Google提出的，又被称为**GoogleNet**。叫做InceptionNetwork的原因是因为诺兰大神的电影《盗梦空间》（Inception）。下图是网络上很流行的一个表情包（meme），出自Inception。所以作者就把网络的名字取作Inception。

{% asset_img deeper-meme.jpg We Need to do deeper %}

**补充**：

Inception Network有4个版本：

- **Inception V1**——构建了1x1、3x3、5x5的 conv 和3x3的 pooling 的分支网络，同时使用 MLPConv 和全局平均池化，扩宽卷积层网络宽度，增加了网络对尺度的适应性；
- **Inception V2**——提出了 Batch Normalization，代替 Dropout 和 LRN，其正则化的效果让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高，同时学习 VGG 使用两个3´3的卷积核代替5´5的卷积核，在降低参数量同时提高网络学习能力；
- **Inception V3**——引入了 Factorization，将一个较大的二维卷积拆成两个较小的一维卷积，比如将3´3卷积拆成1´3卷积和3´1卷积，一方面节约了大量参数，加速运算并减轻了过拟合，同时增加了一层非线性扩展模型表达能力，除了在 Inception Module 中使用分支，还在分支中使用了分支（Network In Network In Network）；
- **Inception V4**——研究了 Inception Module 结合 Residual Connection，结合 ResNet 可以极大地加速训练，同时极大提升性能，在构建 Inception-ResNet 网络同时，还设计了一个更深更优化的 Inception v4 模型，能达到相媲美的性能

其中，V2的论文中提出了著名的 Batch Normalization（以下简称BN）方法。BN 是一个非常有效的正则化方法，可以让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以得到大幅提高。BN 在用于神经网络某层时，会对每一个 mini-batch 数据的内部进行标准化（normalization）处理，使输出规范化到 N(0,1) 的正态分布，减少了 Internal Covariate Shift（内部神经元分布的改变）。

BN 的论文指出，传统的深度神经网络在训练时，每一层的输入的分布都在变化，导致训练变得困难，只能使用一个很小的学习速率解决这个问题。而对每一层使用 BN 之后，我们就可以有效地解决这个问题，学习速率可以增大很多倍，达到之前的准确率所需要的迭代次数只有1/14，训练时间大大缩短。而达到之前的准确率后，可以继续训练，并最终取得远超于 Inception V1 模型的性能—— top-5 错误率 4.8%，已经优于人眼水平。因为 BN 某种意义上还起到了正则化的作用，所以可以减少或者取消 Dropout 和 LRN，简化网络结构。

参考： [https://blog.csdn.net/langb2014/article/details/52787095](https://blog.csdn.net/langb2014/article/details/52787095)

### Practical advices for using ConvNets

#### Using Open-Source Implementation

讲了如何从github下载项目。

#### Transfer Learning

有时候我们想实现一个新的计算机视觉应用，而不想从零开始训练权重，比方从随机初始化开始（训练），实现更快的方式通常是，下载已经训练好权重的网络结构，把这个作为预训练迁移到新的任务上。

训练需要耗费大量的时间和计算资源。从头开始训练肯定是不划算的。比如我有一批训练集，包含15种类型的车，当然训练集的数量有效，只有2000张图片，我想训练一个网络识别不同类型的车。显然，如果使用一个网络从头开始训练，受限于训练集的大小，训练得到的模型效果一般。

这时，我们就可以使用**迁移学习（Transfer Learning）**来训练网络。基本上计算机视觉相关网络都是首先进行特征提取，已经训练好的模型，对大部分图片进行特征提取都是适用的，所以我们可以使用这些已有的参数，对网络进行小的改动，即可实现迁移学习。

比如，下载一个Resnet-50的网络，它的输出是1000种分类，而我们想要的结果只有15种分类。

可以通过如下两种方式进行实现：

第一种是，冻结前面的一些层，即冻结相应参数，加上一个新的softmax层，只训练新增的softmax层有关的参数即可。

{% asset_img transfer_learning_freeze_layer.jpg Transfer Learning:Freeze layer %}

冻结的层数和训练集的大小有关，如果训练集较大，可以冻结前面少部分的层，如果训练集小，可以多冻结一些。

第二种是，可以把我们训练集中的图片在网络中进行推理，把softmax之前一层的结果保持到磁盘，这个结果作为新的网络的输入，我们只需要训练一个包含softmax层的网络即可。

相比于上一种方法的好处是，由于前面的层被冻结，参数不变，所以softmax之前一层的结果是不会变的，我们把这个结果保存下来，不需要每次都进行这样的计算了，可以节省计算量和时间。

{% asset_img transfer_learning_save_to_disk.jpg Transfer Learning:save intermediate result to disk %}

#### Data Augmentation

数据增强可以增强训练集，对训练有收益。

常用的数据增强方法：

- Mirroring: 镜像翻转
- Random Cropping： 随机裁剪
- Rotation: 旋转
- Shearing： 剪切
- Local Warping：局部变形

第二种常见的数据增强的方法是：

- Color Shifting: 色彩变化，分别在GRB不同的channel中加入不同的干扰。通过引入颜色干扰或是色彩变化, 使得学习算法在应对图像色彩变化时健壮性更好.

{% asset_img color_shifting.jpg Color Shifting %}

有一种色彩变化的方法是**PCA**，在AlexNet的论文中有讲述。

在训练中也可以通过一些方法实现分布式的计算，提升训练效率。

{% asset_img implementing_distortions_during_training.jpg %}

#### State of Computer Vision

```
Little data <------------------------------------------> Lots of data
              /\             /\             /\
More hand-    ||             ||             ||         Simpler algorithms
enginnering   ||             ||             ||        Less hand-enginnering
              ||             ||             ||      
            Objection      Image           Speech
            Detection     Recognition   Recognition  
```
Two sources of knowledge:

- Labeled data
- Hand engineered features/network architecture/other components

Tips for doing well on benchmarks/winning competitions

- Ensembling
    - Train several networks independently and average theie outputs
- Multi-crop at test itme
    - Run classifier on multiple versions of test images and average results

User open source code:

- Use srchitectures of networks published in the literature
- Use open source implementions if possible
- Use pretrained models and fine-tune on your dataset

## 编程练习

### Keras Tutorial - The Happy House (not graded)


coding: utf-8

### Keras tutorial - the Happy House

Welcome to the first assignment of week 2. In this assignment, you will:
1. Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK. 
2. See how you can in a couple of hours build a deep learning algorithm.

Why are we using Keras? Keras was developed to enable deep learning engineers to build and experiment with different models very quickly. Just as TensorFlow is a higher-level framework than Python, Keras is an even higher-level framework and provides additional abstractions. Being able to go from idea to result with the least possible delay is key to finding good models. However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models that you can implement in TensorFlow but not (without more difficulty) in Keras. That being said, Keras will work fine for many common models. 

In this exercise, you'll work on the "Happy House" problem, which we'll explain below. Let's load the required packages and solve the problem of the Happy House!

```python
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

get_ipython().run_line_magic('matplotlib', 'inline')
```


**Note**: As you can see, we've imported a lot of functions from Keras. You can use them easily just by calling them directly in the notebook. Ex: `X = Input(...)` or `X = ZeroPadding2D(...)`.

#### 1 - The Happy House 

For your next vacation, you decided to spend a week with five of your friends from school. It is a very convenient house with many things to do nearby. But the most important benefit is that everybody has commited to be happy when they are in the house. So anyone wanting to enter the house must prove their current state of happiness.

{% asset_img happy-house.jpg Figure 1: the Happy House %}


As a deep learning expert, to make sure the "Happy" rule is strictly applied, you are going to build an algorithm which that uses pictures from the front door camera to check if the person is happy or not. The door should open only if the person is happy. 

You have gathered pictures of your friends and yourself, taken by the front-door camera. The dataset is labbeled. 

{% asset_img house-members.png %}

Run the following code to normalize the dataset and learn about its shapes.

```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```


**Details of the "Happy" dataset**:
- Images are of shape (64,64,3)
- Training: 600 pictures
- Test: 150 pictures

It is now time to solve the "Happy" Challenge.

#### 2 - Building a model in Keras

Keras is very good for rapid prototyping. In just a short time you will be able to build a model that achieves outstanding results.

Here is an example of a model in Keras:

```python
def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model
```

Note that Keras uses a different convention with variable names than we've previously used with numpy and TensorFlow. In particular, rather than creating and assigning a new variable on each step of forward propagation such as `X`, `Z1`, `A1`, `Z2`, `A2`, etc. for the computations for the different layers, in Keras code each line above just reassigns `X` to a new value using `X = ...`. In other words, during each step of forward propagation, we are just writing the latest value in the commputation into the same variable `X`. The only exception was `X_input`, which we kept separate and did not overwrite, since we needed it at the end to create the Keras model instance (`model = Model(inputs = X_input, ...)` above). 

**Exercise**: Implement a `HappyModel()`. This assignment is more open-ended than most. We suggest that you start by implementing a model using the architecture we suggest, and run through the rest of this assignment using that as your initial model. But after that, come back and take initiative to try out other model architectures. For example, you might take inspiration from the model above, but then vary the network architecture and hyperparameters however you wish. You can also use other functions such as `AveragePooling2D()`, `GlobalMaxPooling2D()`, `Dropout()`. 

**Note**: You have to be careful with your data's shapes. Use what you've learned in the videos to make sure your convolutional, pooling and fully-connected layers are adapted to the volumes you're applying it to.

```python
# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    ### END CODE HERE ###
    
    return model
```


You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
1. Create the model by calling the function above
2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`
3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`
4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`

If you want to know more about `model.compile()`, `model.fit()`, `model.evaluate()` and their arguments, refer to the official [Keras documentation](https://keras.io/models/model/).

**Exercise**: Implement step 1, i.e. create the model.

```python
### START CODE HERE ### (1 line)
happyModel = HappyModel((64,64,3))
### END CODE HERE ###
```


**Exercise**: Implement step 2, i.e. compile the model to configure the learning process. Choose the 3 arguments of `compile()` wisely. Hint: the Happy Challenge is a binary classification problem.

```python
### START CODE HERE ### (1 line)
happyModel.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
### END CODE HERE ###
```

**Exercise**: Implement step 3, i.e. train the model. Choose the number of epochs and the batch size.

```python
### START CODE HERE ### (1 line)
happyModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=20)
### END CODE HERE ###
```


Note that if you run `fit()` again, the `model` will continue to train with the parameters it has already learnt instead of reinitializing them.

**Exercise**: Implement step 4, i.e. test/evaluate the model.

```python
### START CODE HERE ### (1 line)
preds = happyModel.evaluate(x=X_test, y=Y_test,)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
```


If your `happyModel()` function worked, you should have observed much better than random-guessing (50%) accuracy on the train and test sets.

To give you a point of comparison, our model gets around **95% test accuracy in 40 epochs** (and 99% train accuracy) with a mini batch size of 16 and "adam" optimizer. But our model gets decent accuracy after just 2-5 epochs, so if you're comparing different models you can also train a variety of models on just a few epochs and see how they compare. 

If you have not yet achieved a very good accuracy (let's say more than 80%), here're some things you can play around with to try to achieve it:

- Try using blocks of CONV->BATCHNORM->RELU such as:

```python
X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)
```
until your height and width dimensions are quite low and your number of channels quite large (≈32 for example). You are encoding useful information in a volume with a lot of channels. You can then flatten the volume and use a fully-connected layer.
- You can use MAXPOOL after such blocks. It will help you lower the dimension in height and width.
- Change your optimizer. We find Adam works well. 
- If the model is struggling to run and you get memory issues, lower your batch_size (12 is usually a good compromise)
- Run on more epochs, until you see the train accuracy plateauing. 

Even if you have achieved a good accuracy, please feel free to keep playing with your model to try to get even better results. 

**Note**: If you perform hyperparameter tuning on your model, the test set actually becomes a dev set, and your model might end up overfitting to the test (dev) set. But just for the purpose of this assignment, we won't worry about that here.


#### 3 - Conclusion

Congratulations, you have solved the Happy House challenge! 

Now, you just need to link this model to the front-door camera of your house. We unfortunately won't go into the details of how to do that here. 

<font color='blue'>
**What we would like you to remember from this assignment:**

- Keras is a tool we recommend for rapid prototyping. It allows you to quickly try out different model architectures. Are there any applications of deep learning to your daily life that you'd like to implement using Keras? 
- Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set. Create->Compile->Fit/Train->Evaluate/Test.
</font> 

## 4 - Test with your own image (Optional)

Congratulations on finishing this assignment. You can now take a picture of your face and see if you could enter the Happy House. To do that:
    1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
    2. Add your image to this Jupyter Notebook's directory, in the "images" folder
    3. Write your image's name in the following code
    4. Run the code and check if the algorithm is right (0 is unhappy, 1 is happy)!
    
The training/test sets were quite similar; for example, all the pictures were taken against the same background (since a front door camera is always mounted in the same position). This makes the problem easier, but a model trained on this data may or may not work on your own data. But feel free to give it a try! 

```python
### START CODE HERE ###
for i in range(1,5):
    img_path = 'images/'+str(i) + '.jpg'
    ### END CODE HERE ###
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(happyModel.predict(x))
```


#### 5 - Other useful functions in Keras (Optional)

Two other basic features of Keras that you'll find useful are:
- `model.summary()`: prints the details of your layers in a table with the sizes of its inputs/outputs
- `plot_model()`: plots your graph in a nice layout. You can even save it as ".png" using SVG() if you'd like to share it on social media ;). It is saved in "File" then "Open..." in the upper bar of the notebook.

Run the following code.

```python
happyModel.summary()
```


```python
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
```

### Residual Networks

Welcome to the second assignment of this week! You will learn how to build very deep convolutional networks, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by [He et al.](https://arxiv.org/pdf/1512.03385.pdf), allow you to train much deeper networks than were previously practically feasible.

**In this assignment, you will:**
- Implement the basic building blocks of ResNets. 
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification. 

This assignment will be done in Keras. 

Before jumping into the problem, let's run the cell below to load the required packages.


```python
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
```
    

#### 1 - The problem of very deep neural networks

Last week, you built your first convolutional neural network. In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers.

The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow. More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values). 

During training, you might therefore see the magnitude (or norm) of the gradient for the earlier layers descrease to zero very rapidly as training proceeds: 

{% asset_img vanishing_grad_kiank.png  Figure 1:Vanishing gradient. The speed of learning decreases very rapidly for the early layers as the network trains %}

You are now going to solve this problem by building a Residual Network!

#### 2 - Building a Residual Network

In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:  

{% asset_img skip_connection_kiank.png Figure 2 : A ResNet block showing a skip-connection %}

The image on the left shows the "main path" through the network. The image on the right adds a shortcut to the main path. By stacking these ResNet blocks on top of each other, you can form a very deep network. 

We also saw in lecture that having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that you can stack on additional ResNet blocks with little risk of harming training set performance. (There is also some evidence that the ease of learning an identity function--even more than skip connections helping with vanishing gradients--accounts for ResNets' remarkable performance.)

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. You are going to implement both of them. 

##### 2.1 - The identity block

The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say $a^{[l]}$) has the same dimension as the output activation (say $a^{[l+2]}$). To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:

{% asset_img idblock2_kiank.png Figure 3: Identity block.Skip connection skips over 2 layers. %}

The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step. Don't worry about this being complicated to implement--you'll see that BatchNorm is just one line of code in Keras! 

In this exercise, you'll actually implement a slightly more powerful version of this identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers. It looks like this: 

{% asset_img idblock3_kiank.png %}

Here're the individual steps.

First component of main path: 
- The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be `conv_name_base + '2a'`. Use 0 as the seed for the random initialization. 
- The first BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2a'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Second component of main path:
- The second CONV2D has $F_2$ filters of shape $(f,f)$ and a stride of (1,1). Its padding is "same" and its name should be `conv_name_base + '2b'`. Use 0 as the seed for the random initialization. 
- The second BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2b'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Third component of main path:
- The third CONV2D has $F_3$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be `conv_name_base + '2c'`. Use 0 as the seed for the random initialization. 
- The third BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2c'`. Note that there is no ReLU activation function in this component. 

Final step: 
- The shortcut and the input are added together.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

**Exercise**: Implement the ResNet identity block. We have implemented the first component of the main path. Please read over this carefully to make sure you understand what it is doing. You should implement the rest. 
- To implement the Conv2D step: [See reference](https://keras.io/layers/convolutional/#conv2d)
- To implement BatchNorm: [See reference](https://faroit.github.io/keras-docs/1.2.2/layers/normalization/) (axis: Integer, the axis that should be normalized (typically the channels axis))
- For the activation, use:  `Activation('relu')(X)`
- To add the value passed forward by the shortcut: [See reference](https://keras.io/layers/merge/#add)


```python
# GRADED FUNCTION: identity_block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
```


```python
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
```

##### 2.2 - The convolutional block

You've implemented the ResNet identity block. Next, the ResNet "convolutional block" is the other type of block. You can use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path: 

{% asset_img convblock_kiank.png Figure 4 : Convolutional block %}

The CONV2D layer in the shortcut path is used to resize the input $x$ to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. (This plays a similar role as the matrix $W_s$ discussed in lecture.) For example, to reduce the activation dimensions's height and width by a factor of 2, you can use a 1x1 convolution with a stride of 2. The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step. 

The details of the convolutional block are as follows. 

First component of main path:
- The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be `conv_name_base + '2a'`. 
- The first BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2a'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Second component of main path:
- The second CONV2D has $F_2$ filters of (f,f) and a stride of (1,1). Its padding is "same" and it's name should be `conv_name_base + '2b'`.
- The second BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2b'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Third component of main path:
- The third CONV2D has $F_3$ filters of (1,1) and a stride of (1,1). Its padding is "valid" and it's name should be `conv_name_base + '2c'`.
- The third BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '2c'`. Note that there is no ReLU activation function in this component. 

Shortcut path:
- The CONV2D has $F_3$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be `conv_name_base + '1'`.
- The BatchNorm is normalizing the channels axis.  Its name should be `bn_name_base + '1'`. 

Final step: 
- The shortcut and the main path values are added together.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 
    
**Exercise**: Implement the convolutional block. We have implemented the first component of the main path; you should implement the rest. As before, always use 0 as the seed for the random initialization, to ensure consistency with our grader.
- [Conv Hint](https://keras.io/layers/convolutional/#conv2d)
- [BatchNorm Hint](https://keras.io/layers/normalization/#batchnormalization) (axis: Integer, the axis that should be normalized (typically the features axis))
- For the activation, use:  `Activation('relu')(X)`
- [Addition Hint](https://keras.io/layers/merge/#add)


```python
# GRADED FUNCTION: convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f,f), strides=(1,1), name = conv_name_base + '2b', padding='same' ,kernel_initializer= glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base +'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1,1), strides=(1,1), name=conv_name_base+'2c', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1,1), strides=(s,s), name=conv_name_base+'1', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
```


```python
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
```


#### 3 - Building your first ResNet model (50 layers)

You now have the necessary blocks to build a very deep ResNet. The following figure describes in detail the architecture of this neural network. "ID BLOCK" in the diagram stands for "Identity block," and "ID BLOCK x3" means you should stack 3 identity blocks together.

{% asset_img resnet_kiank.png Figure 5 : ResNet-50 model %}

The details of this ResNet-50 model are:
- Zero-padding pads the input with a pad of (3,3)
- Stage 1:
    - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
    - BatchNorm is applied to the channels axis of the input.
    - MaxPooling uses a (3,3) window and a (2,2) stride.
- Stage 2:
    - The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
    - The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
- Stage 3:
    - The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    - The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
- Stage 4:
    - The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    - The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
- Stage 5:
    - The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    - The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
- The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
- The flatten doesn't have any hyperparameters or name.
- The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be `'fc' + str(classes)`.

**Exercise**: Implement the ResNet with 50 layers described in the figure above. We have implemented Stages 1 and 2. Please implement the rest. (The syntax for implementing Stages 3-5 should be quite similar to that of Stage 2.) Make sure you follow the naming convention in the text above. 

You'll need to use this function: 
- Average pooling [see reference](https://keras.io/layers/pooling/#averagepooling2d)

Here're some other functions we used in the code below:
- Conv2D: [See reference](https://keras.io/layers/convolutional/#conv2d)
- BatchNorm: [See reference](https://keras.io/layers/normalization/#batchnormalization) (axis: Integer, the axis that should be normalized (typically the features axis))
- Zero padding: [See reference](https://keras.io/layers/convolutional/#zeropadding2d)
- Max pooling: [See reference](https://keras.io/layers/pooling/#maxpooling2d)
- Fully conected layer: [See reference](https://keras.io/layers/core/#dense)
- Addition: [See reference](https://keras.io/layers/merge/#add)


```python
# GRADED FUNCTION: ResNet50

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters =  [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3,  [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3,  [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name="avg_pool")(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
```

Run the following code to build the model's graph. If your implementation is not correct you will know it by checking your accuracy when running `model.fit(...)` below.


```python
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
```

As seen in the Keras Tutorial Notebook, prior training a model, you need to configure the learning process by compiling the model.


```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

The model is now ready to be trained. The only thing you need is a dataset.

Let's load the SIGNS Dataset.

{% asset_img signs_data_kiank.png Figure 6 : SIGNS dataset %}


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

Run the following cell to train your model on 2 epochs with a batch size of 32. On a CPU it should take you around 5min per epoch. 


```python
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)
```

```
    Epoch 1/2
    1080/1080 [==============================] - 280s - loss: 3.1437 - acc: 0.2546   
    Epoch 2/2
    1080/1080 [==============================] - 272s - loss: 2.1371 - acc: 0.3426   
```

Let's see how this model (trained on only two epochs) performs on the test set.


```python
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
```

```
    120/120 [==============================] - 10s    
    Loss = 2.6892288208
    Test Accuracy = 0.166666666667
```

For the purpose of this assignment, we've asked you to train the model only for two epochs. You can see that it achieves poor performances. Please go ahead and submit your assignment; to check correctness, the online grader will run your code only for a small number of epochs as well.

After you have finished this official (graded) part of this assignment, you can also optionally train the ResNet for more iterations, if you want. We get a lot better performance when we train for ~20 epochs, but this will take more than an hour when training on a CPU. 

Using a GPU, we've trained our own ResNet50 model's weights on the SIGNS dataset. You can load and run our trained model on the test set in the cells below. It may take ≈1min to load the model.


```python
model = load_model('ResNet50.h5') 
```


```python
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
```

```
    120/120 [==============================] - 11s    
    Loss = 0.530178320408
    Test Accuracy = 0.866666662693
```

ResNet50 is a powerful model for image classification when it is trained for an adequate number of iterations. We hope you can use what you've learnt and apply it to your own classification problem to perform state-of-the-art accuracy.

Congratulations on finishing this assignment! You've now implemented a state-of-the-art image classification system! 

#### 4 - Test on your own image (Optional/Ungraded)

If you wish, you can also take a picture of your own hand and see the output of the model. To do this:
    1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
    2. Add your image to this Jupyter Notebook's directory, in the "images" folder
    3. Write your image's name in the following code
    4. Run the code and check if the algorithm is right! 


```python
img_path = 'images/2.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
my_image = scipy.misc.imread(img_path)
imshow(my_image)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))
```

{% asset_img output_35_1.png %}


```python
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_1 (InputLayer)             (None, 64, 64, 3)     0                                            
    ____________________________________________________________________________________________________
    zero_padding2d_1 (ZeroPadding2D) (None, 70, 70, 3)     0           input_1[0][0]                    
    ____________________________________________________________________________________________________
    conv1 (Conv2D)                   (None, 32, 32, 64)    9472        zero_padding2d_1[0][0]           
    ____________________________________________________________________________________________________
    bn_conv1 (BatchNormalization)    (None, 32, 32, 64)    256         conv1[0][0]                      
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 32, 32, 64)    0           bn_conv1[0][0]                   
    ____________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)   (None, 15, 15, 64)    0           activation_4[0][0]               
    ____________________________________________________________________________________________________
    res2a_branch2a (Conv2D)          (None, 15, 15, 64)    4160        max_pooling2d_1[0][0]            
    ____________________________________________________________________________________________________
    bn2a_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2a_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_5 (Activation)        (None, 15, 15, 64)    0           bn2a_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res2a_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_5[0][0]               
    ____________________________________________________________________________________________________
    bn2a_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2a_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_6 (Activation)        (None, 15, 15, 64)    0           bn2a_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res2a_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_6[0][0]               
    ____________________________________________________________________________________________________
    res2a_branch1 (Conv2D)           (None, 15, 15, 256)   16640       max_pooling2d_1[0][0]            
    ____________________________________________________________________________________________________
    bn2a_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2a_branch2c[0][0]             
    ____________________________________________________________________________________________________
    bn2a_branch1 (BatchNormalization (None, 15, 15, 256)   1024        res2a_branch1[0][0]              
    ____________________________________________________________________________________________________
    add_2 (Add)                      (None, 15, 15, 256)   0           bn2a_branch2c[0][0]              
                                                                       bn2a_branch1[0][0]               
    ____________________________________________________________________________________________________
    activation_7 (Activation)        (None, 15, 15, 256)   0           add_2[0][0]                      
    ____________________________________________________________________________________________________
    res2b_branch2a (Conv2D)          (None, 15, 15, 64)    16448       activation_7[0][0]               
    ____________________________________________________________________________________________________
    bn2b_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2b_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_8 (Activation)        (None, 15, 15, 64)    0           bn2b_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res2b_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_8[0][0]               
    ____________________________________________________________________________________________________
    bn2b_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2b_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_9 (Activation)        (None, 15, 15, 64)    0           bn2b_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res2b_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_9[0][0]               
    ____________________________________________________________________________________________________
    bn2b_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2b_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_3 (Add)                      (None, 15, 15, 256)   0           bn2b_branch2c[0][0]              
                                                                       activation_7[0][0]               
    ____________________________________________________________________________________________________
    activation_10 (Activation)       (None, 15, 15, 256)   0           add_3[0][0]                      
    ____________________________________________________________________________________________________
    res2c_branch2a (Conv2D)          (None, 15, 15, 64)    16448       activation_10[0][0]              
    ____________________________________________________________________________________________________
    bn2c_branch2a (BatchNormalizatio (None, 15, 15, 64)    256         res2c_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_11 (Activation)       (None, 15, 15, 64)    0           bn2c_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res2c_branch2b (Conv2D)          (None, 15, 15, 64)    36928       activation_11[0][0]              
    ____________________________________________________________________________________________________
    bn2c_branch2b (BatchNormalizatio (None, 15, 15, 64)    256         res2c_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_12 (Activation)       (None, 15, 15, 64)    0           bn2c_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res2c_branch2c (Conv2D)          (None, 15, 15, 256)   16640       activation_12[0][0]              
    ____________________________________________________________________________________________________
    bn2c_branch2c (BatchNormalizatio (None, 15, 15, 256)   1024        res2c_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_4 (Add)                      (None, 15, 15, 256)   0           bn2c_branch2c[0][0]              
                                                                       activation_10[0][0]              
    ____________________________________________________________________________________________________
    activation_13 (Activation)       (None, 15, 15, 256)   0           add_4[0][0]                      
    ____________________________________________________________________________________________________
    res3a_branch2a (Conv2D)          (None, 8, 8, 128)     32896       activation_13[0][0]              
    ____________________________________________________________________________________________________
    bn3a_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3a_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_14 (Activation)       (None, 8, 8, 128)     0           bn3a_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res3a_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_14[0][0]              
    ____________________________________________________________________________________________________
    bn3a_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3a_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_15 (Activation)       (None, 8, 8, 128)     0           bn3a_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res3a_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_15[0][0]              
    ____________________________________________________________________________________________________
    res3a_branch1 (Conv2D)           (None, 8, 8, 512)     131584      activation_13[0][0]              
    ____________________________________________________________________________________________________
    bn3a_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3a_branch2c[0][0]             
    ____________________________________________________________________________________________________
    bn3a_branch1 (BatchNormalization (None, 8, 8, 512)     2048        res3a_branch1[0][0]              
    ____________________________________________________________________________________________________
    add_5 (Add)                      (None, 8, 8, 512)     0           bn3a_branch2c[0][0]              
                                                                       bn3a_branch1[0][0]               
    ____________________________________________________________________________________________________
    activation_16 (Activation)       (None, 8, 8, 512)     0           add_5[0][0]                      
    ____________________________________________________________________________________________________
    res3b_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_16[0][0]              
    ____________________________________________________________________________________________________
    bn3b_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3b_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_17 (Activation)       (None, 8, 8, 128)     0           bn3b_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res3b_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_17[0][0]              
    ____________________________________________________________________________________________________
    bn3b_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3b_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_18 (Activation)       (None, 8, 8, 128)     0           bn3b_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res3b_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_18[0][0]              
    ____________________________________________________________________________________________________
    bn3b_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3b_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_6 (Add)                      (None, 8, 8, 512)     0           bn3b_branch2c[0][0]              
                                                                       activation_16[0][0]              
    ____________________________________________________________________________________________________
    activation_19 (Activation)       (None, 8, 8, 512)     0           add_6[0][0]                      
    ____________________________________________________________________________________________________
    res3c_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_19[0][0]              
    ____________________________________________________________________________________________________
    bn3c_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3c_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_20 (Activation)       (None, 8, 8, 128)     0           bn3c_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res3c_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_20[0][0]              
    ____________________________________________________________________________________________________
    bn3c_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3c_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_21 (Activation)       (None, 8, 8, 128)     0           bn3c_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res3c_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_21[0][0]              
    ____________________________________________________________________________________________________
    bn3c_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3c_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_7 (Add)                      (None, 8, 8, 512)     0           bn3c_branch2c[0][0]              
                                                                       activation_19[0][0]              
    ____________________________________________________________________________________________________
    activation_22 (Activation)       (None, 8, 8, 512)     0           add_7[0][0]                      
    ____________________________________________________________________________________________________
    res3d_branch2a (Conv2D)          (None, 8, 8, 128)     65664       activation_22[0][0]              
    ____________________________________________________________________________________________________
    bn3d_branch2a (BatchNormalizatio (None, 8, 8, 128)     512         res3d_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_23 (Activation)       (None, 8, 8, 128)     0           bn3d_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res3d_branch2b (Conv2D)          (None, 8, 8, 128)     147584      activation_23[0][0]              
    ____________________________________________________________________________________________________
    bn3d_branch2b (BatchNormalizatio (None, 8, 8, 128)     512         res3d_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_24 (Activation)       (None, 8, 8, 128)     0           bn3d_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res3d_branch2c (Conv2D)          (None, 8, 8, 512)     66048       activation_24[0][0]              
    ____________________________________________________________________________________________________
    bn3d_branch2c (BatchNormalizatio (None, 8, 8, 512)     2048        res3d_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_8 (Add)                      (None, 8, 8, 512)     0           bn3d_branch2c[0][0]              
                                                                       activation_22[0][0]              
    ____________________________________________________________________________________________________
    activation_25 (Activation)       (None, 8, 8, 512)     0           add_8[0][0]                      
    ____________________________________________________________________________________________________
    res4a_branch2a (Conv2D)          (None, 4, 4, 256)     131328      activation_25[0][0]              
    ____________________________________________________________________________________________________
    bn4a_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4a_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_26 (Activation)       (None, 4, 4, 256)     0           bn4a_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res4a_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_26[0][0]              
    ____________________________________________________________________________________________________
    bn4a_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4a_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_27 (Activation)       (None, 4, 4, 256)     0           bn4a_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res4a_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_27[0][0]              
    ____________________________________________________________________________________________________
    res4a_branch1 (Conv2D)           (None, 4, 4, 1024)    525312      activation_25[0][0]              
    ____________________________________________________________________________________________________
    bn4a_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4a_branch2c[0][0]             
    ____________________________________________________________________________________________________
    bn4a_branch1 (BatchNormalization (None, 4, 4, 1024)    4096        res4a_branch1[0][0]              
    ____________________________________________________________________________________________________
    add_9 (Add)                      (None, 4, 4, 1024)    0           bn4a_branch2c[0][0]              
                                                                       bn4a_branch1[0][0]               
    ____________________________________________________________________________________________________
    activation_28 (Activation)       (None, 4, 4, 1024)    0           add_9[0][0]                      
    ____________________________________________________________________________________________________
    res4b_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_28[0][0]              
    ____________________________________________________________________________________________________
    bn4b_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4b_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_29 (Activation)       (None, 4, 4, 256)     0           bn4b_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res4b_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_29[0][0]              
    ____________________________________________________________________________________________________
    bn4b_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4b_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_30 (Activation)       (None, 4, 4, 256)     0           bn4b_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res4b_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_30[0][0]              
    ____________________________________________________________________________________________________
    bn4b_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4b_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_10 (Add)                     (None, 4, 4, 1024)    0           bn4b_branch2c[0][0]              
                                                                       activation_28[0][0]              
    ____________________________________________________________________________________________________
    activation_31 (Activation)       (None, 4, 4, 1024)    0           add_10[0][0]                     
    ____________________________________________________________________________________________________
    res4c_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_31[0][0]              
    ____________________________________________________________________________________________________
    bn4c_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4c_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_32 (Activation)       (None, 4, 4, 256)     0           bn4c_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res4c_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_32[0][0]              
    ____________________________________________________________________________________________________
    bn4c_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4c_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_33 (Activation)       (None, 4, 4, 256)     0           bn4c_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res4c_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_33[0][0]              
    ____________________________________________________________________________________________________
    bn4c_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4c_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_11 (Add)                     (None, 4, 4, 1024)    0           bn4c_branch2c[0][0]              
                                                                       activation_31[0][0]              
    ____________________________________________________________________________________________________
    activation_34 (Activation)       (None, 4, 4, 1024)    0           add_11[0][0]                     
    ____________________________________________________________________________________________________
    res4d_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_34[0][0]              
    ____________________________________________________________________________________________________
    bn4d_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4d_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_35 (Activation)       (None, 4, 4, 256)     0           bn4d_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res4d_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_35[0][0]              
    ____________________________________________________________________________________________________
    bn4d_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4d_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_36 (Activation)       (None, 4, 4, 256)     0           bn4d_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res4d_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_36[0][0]              
    ____________________________________________________________________________________________________
    bn4d_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4d_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_12 (Add)                     (None, 4, 4, 1024)    0           bn4d_branch2c[0][0]              
                                                                       activation_34[0][0]              
    ____________________________________________________________________________________________________
    activation_37 (Activation)       (None, 4, 4, 1024)    0           add_12[0][0]                     
    ____________________________________________________________________________________________________
    res4e_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_37[0][0]              
    ____________________________________________________________________________________________________
    bn4e_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4e_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_38 (Activation)       (None, 4, 4, 256)     0           bn4e_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res4e_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_38[0][0]              
    ____________________________________________________________________________________________________
    bn4e_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4e_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_39 (Activation)       (None, 4, 4, 256)     0           bn4e_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res4e_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_39[0][0]              
    ____________________________________________________________________________________________________
    bn4e_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4e_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_13 (Add)                     (None, 4, 4, 1024)    0           bn4e_branch2c[0][0]              
                                                                       activation_37[0][0]              
    ____________________________________________________________________________________________________
    activation_40 (Activation)       (None, 4, 4, 1024)    0           add_13[0][0]                     
    ____________________________________________________________________________________________________
    res4f_branch2a (Conv2D)          (None, 4, 4, 256)     262400      activation_40[0][0]              
    ____________________________________________________________________________________________________
    bn4f_branch2a (BatchNormalizatio (None, 4, 4, 256)     1024        res4f_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_41 (Activation)       (None, 4, 4, 256)     0           bn4f_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res4f_branch2b (Conv2D)          (None, 4, 4, 256)     590080      activation_41[0][0]              
    ____________________________________________________________________________________________________
    bn4f_branch2b (BatchNormalizatio (None, 4, 4, 256)     1024        res4f_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_42 (Activation)       (None, 4, 4, 256)     0           bn4f_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res4f_branch2c (Conv2D)          (None, 4, 4, 1024)    263168      activation_42[0][0]              
    ____________________________________________________________________________________________________
    bn4f_branch2c (BatchNormalizatio (None, 4, 4, 1024)    4096        res4f_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_14 (Add)                     (None, 4, 4, 1024)    0           bn4f_branch2c[0][0]              
                                                                       activation_40[0][0]              
    ____________________________________________________________________________________________________
    activation_43 (Activation)       (None, 4, 4, 1024)    0           add_14[0][0]                     
    ____________________________________________________________________________________________________
    res5a_branch2a (Conv2D)          (None, 2, 2, 512)     524800      activation_43[0][0]              
    ____________________________________________________________________________________________________
    bn5a_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5a_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_44 (Activation)       (None, 2, 2, 512)     0           bn5a_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res5a_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_44[0][0]              
    ____________________________________________________________________________________________________
    bn5a_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5a_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_45 (Activation)       (None, 2, 2, 512)     0           bn5a_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res5a_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_45[0][0]              
    ____________________________________________________________________________________________________
    res5a_branch1 (Conv2D)           (None, 2, 2, 2048)    2099200     activation_43[0][0]              
    ____________________________________________________________________________________________________
    bn5a_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5a_branch2c[0][0]             
    ____________________________________________________________________________________________________
    bn5a_branch1 (BatchNormalization (None, 2, 2, 2048)    8192        res5a_branch1[0][0]              
    ____________________________________________________________________________________________________
    add_15 (Add)                     (None, 2, 2, 2048)    0           bn5a_branch2c[0][0]              
                                                                       bn5a_branch1[0][0]               
    ____________________________________________________________________________________________________
    activation_46 (Activation)       (None, 2, 2, 2048)    0           add_15[0][0]                     
    ____________________________________________________________________________________________________
    res5b_branch2a (Conv2D)          (None, 2, 2, 512)     1049088     activation_46[0][0]              
    ____________________________________________________________________________________________________
    bn5b_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5b_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_47 (Activation)       (None, 2, 2, 512)     0           bn5b_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res5b_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_47[0][0]              
    ____________________________________________________________________________________________________
    bn5b_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5b_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_48 (Activation)       (None, 2, 2, 512)     0           bn5b_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res5b_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_48[0][0]              
    ____________________________________________________________________________________________________
    bn5b_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5b_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_16 (Add)                     (None, 2, 2, 2048)    0           bn5b_branch2c[0][0]              
                                                                       activation_46[0][0]              
    ____________________________________________________________________________________________________
    activation_49 (Activation)       (None, 2, 2, 2048)    0           add_16[0][0]                     
    ____________________________________________________________________________________________________
    res5c_branch2a (Conv2D)          (None, 2, 2, 512)     1049088     activation_49[0][0]              
    ____________________________________________________________________________________________________
    bn5c_branch2a (BatchNormalizatio (None, 2, 2, 512)     2048        res5c_branch2a[0][0]             
    ____________________________________________________________________________________________________
    activation_50 (Activation)       (None, 2, 2, 512)     0           bn5c_branch2a[0][0]              
    ____________________________________________________________________________________________________
    res5c_branch2b (Conv2D)          (None, 2, 2, 512)     2359808     activation_50[0][0]              
    ____________________________________________________________________________________________________
    bn5c_branch2b (BatchNormalizatio (None, 2, 2, 512)     2048        res5c_branch2b[0][0]             
    ____________________________________________________________________________________________________
    activation_51 (Activation)       (None, 2, 2, 512)     0           bn5c_branch2b[0][0]              
    ____________________________________________________________________________________________________
    res5c_branch2c (Conv2D)          (None, 2, 2, 2048)    1050624     activation_51[0][0]              
    ____________________________________________________________________________________________________
    bn5c_branch2c (BatchNormalizatio (None, 2, 2, 2048)    8192        res5c_branch2c[0][0]             
    ____________________________________________________________________________________________________
    add_17 (Add)                     (None, 2, 2, 2048)    0           bn5c_branch2c[0][0]              
                                                                       activation_49[0][0]              
    ____________________________________________________________________________________________________
    activation_52 (Activation)       (None, 2, 2, 2048)    0           add_17[0][0]                     
    ____________________________________________________________________________________________________
    avg_pool (AveragePooling2D)      (None, 1, 1, 2048)    0           activation_52[0][0]              
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 2048)          0           avg_pool[0][0]                   
    ____________________________________________________________________________________________________
    fc6 (Dense)                      (None, 6)             12294       flatten_1[0][0]                  
    ====================================================================================================
    Total params: 23,600,006
    Trainable params: 23,546,886
    Non-trainable params: 53,120
    ____________________________________________________________________________________________________
    

Finally, run the code below to visualize your ResNet50. You can also download a .png picture of your model by going to "File -> Open...-> model.png".


```python
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

{% asset_img output_38_0.svg %}

**What you should remember:**

- Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.  
- The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function. 
- There are two main type of blocks: The identity block and the convolutional block. 
- Very deep Residual Networks are built by stacking these blocks together.

You can also print a summary of your model by running the following code.

##### References 

This notebook presents the ResNet algorithm due to He et al. (2015). The implementation here also took significant inspiration and follows the structure given in the github repository of Francois Chollet: 

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
- Francois Chollet's github repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
