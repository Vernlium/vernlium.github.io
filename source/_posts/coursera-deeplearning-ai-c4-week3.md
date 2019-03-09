---
title: coursera-deeplearning-ai-c4-week3
mathjax: true
date: 2018-10-30 07:16:32
tags: [deeplearning.ai]
---

## 课程笔记

本周课程主要介绍当前计算机视觉领域最热门的领域：目标检测（Object detection），介绍目标检测的相关概念和基础算法，以及常用的一些网络，包括：

- 目标检测的概念
- 标记点检测（Landmark detection）
- 滑动窗口(Sliding windows)
- 非最大抑制值（non-max suppression）算法
- IOU（intersection over union，交并比）算法
- YOLO网络
- 锚框（Anchor box）

**学习目标**

- Understand the challenges of Object Localization, Object Detection and Landmark Finding
- Understand and implement non-max suppression
- Understand and implement intersection over union
- Understand how we label a dataset for an object detection application
- Remember the vocabulary of object detection (landmark, anchor, bounding box, grid, ...)

### Detection algorithms

#### Object Localization

分类并定位: 不仅仅要识别出, 并且算法也负责生成一个边框，标出物体的位置。

对象检测， 如何检测在一张图片里的多个对象，并且能够把它们全部检测出来并且定位它们。 

对于图像分类 和分类并定位问题，通常只有一个对象。 相比之下，在对象检测问题中，可能会有很多对象。

{% asset_img localization_and_detection.jpg Localization and Detection  %}

#### Landmark Detection

landmark检测是指对图片中的物体的特征点进行检测，比如人脸，标记出人脸上的特征点，如眼角、嘴角、鼻子、下巴等。或者人体特征，如手臂、腿、胸、头等。

#### Object Detection

目标检测，在深度学习没有兴起之前的做法是，比如要检测车，先训练一个小网络，检测一副小图片中是否包含车。

然后在要检测的图片中，通过下图滑动窗口的方式，在大图片中有个小窗口一直滑动，然后使用网络判断小窗口中是否有车，从而找到车的位置。为了检测的更准确，需要大小不同的几个窗口来进行滑动。

{% asset_img  sliding_windows_detection.jpg Sliding Windows Detection %}

这种方法的缺点是，如果窗口多且滑动步长小，虽然检测精确，但是计算量很大；如果窗口少且滑动步长大，则检测精度不够。不能很好的解决目标检测的问题。

下面通过卷积的方式解决此问题。

#### Convolutional Implementation of Sliding Windows

首先讲一下，如何把全连接层转换为卷积层进行计算。为下面的用卷积实现滑动窗口做铺垫。

下图中，把上面的全连接层替换为下面的卷积层，后面的全连接全部用卷积替换，可以得到同样的计算结果。

{% asset_img  turing_fc_layer_into_convolutional_layers.jpg Turning FC layer into convolutional layers %}

最上面一行，一个14\*14的窗口，计算出一个结果，中间一行，输入是16\*16，可以划分4个14\*14的窗口，经过计算后得到4个结果。这样就可以实现类似滑动窗口的效果，而且大部分计算都是共享的，可以节省大部分的计算量。

{% asset_img convolution_implementation_of_sliding_windows.jpg Convolution implementation of sliding windows %}

同样，最后一行的输入更大，可以划分更多的小窗口，得到更多的结果。

#### Bounding Box Predictions（边界框预测）

通过前面讲到的滑动窗口目标检测，有时候无法得到精确的预测，因为窗口的大小和位置是固定，而目标出现的位置比较随机。

YOLO(You Only Look Once)算法可以解决此问题，而且速度很快。

yolo算法的做法是，把输入的图像分割为3×3的窗口（这儿为了讲解方便，使用3×3，实际应用中，是19×19或者其他的划分）。每个窗口对应一个label，label的形式如下图所示，$p_c$表示窗口中是否含有目标，然后下面4个值是目标的位置信息，后面3个是对应目标的种类（这儿只检测三种类型的目标，汽车/行人/摩托车）。比如，下图中，9个小窗口，第4个和第6个窗口中包含汽车，则$p_c = 1$，其他窗口不含目标，$p_c = 0$。

训练集中，每个示例都是一副图片，对应的label和下图中类似。通过卷积和池化等操作，把输入映射到对应的label上。比如下图的示例中，输入是$100×100×3$的图片，对应的label是$3×3×8$的结果。

{% asset_img yolo_algorithm.jpg YOLO algorithm %}

算法的优势在于，该神经网络可以精确地输出边界框。在实际操作中，相比这里使用的相对较少的3乘3的网格，可能会使用更精细的网格，可能是19乘19的网格，所以最终会得到19乘19乘8的结果。由于使用了更精细的网格，这会减少同一个网格元中有多个目标物的可能性。值得提醒的是，将目标物分配到网格元中的方式，即先找到目标物的中心点，再根据中心点的位置将它分配到包含该中心点的网格中。所以对于每一个目标物，即使它跨越了多个网格，它也只会被分配给这九个网格元中的一个，或者说这3乘3个网格元中的一个，或者19乘19个网格元中的一个。而使用19乘19的网格的算法，两个目标物的中心，出现在同一个网格元中的概率会稍小。

所以值得注意的有两点：

- 1.这种会直接输出边界框的坐标位置，以及它允许神经网络输出，任意长宽比的边界框，同时输出的坐标位置也更为精确，而不会受限于滑动窗口的步长 
- 2.这个算法是通过卷积实现的，不需要在3乘3的网格上执行这个算法9次。这个算法是一整个卷积实现的，只需要用一个卷积网络，所需的计算被大量地共享，这是一个非常有效率的算法。事实上，YOLO算法的一个好处，也是它一直很流行的原因是它是通过卷积实现的，实际上的运行起来非常快，它甚至可以运用在实时的目标识别上。

值得注意的是，YOLO算法中，目标的位置的bx,by,bh,bw是以每个小网格的右上点为参考点。如下图所示。

{% asset_img specify_the_bounding_boxes.jpg %}


#### Intersection Over Union（交并比）

交并比（Intersection Over Union，简称IOU）是用来判断目标检测算法是否有效的算法。它既可以用来评价你的目标检测算法，也可以用于，往目标检测算法中加入其他特征部分，来进一步改善它。

交并比或者说是IoU函数做的是计算两个边界框的交集除以并集的比率。按惯例，或者说计算机视觉领域的原则，如果IoU大于0.5，结果就会被判断为正确的，如果预测的和真实的边界框完美重合了 IoU就会是1。如果你想更严格一些，可以把准确的标准提高，仅当IoU大于等于0.6，或者别的数值。**IoU越高，边界框就越准确**。

{% asset_img iou.jpg IOU %}

IoU创立的初衷，作为一个评估的方法，来判断目标定位算法是否准确。但更普遍地说，IoU是，两个边界框重叠程度的一个度量值。当有两个框时，可以计算交集，计算并集，然后求两个面积的比值。因此这也是一个方法，来衡量两个框的相近程度。

当讨论非最大值抑制(non-max suppression)时，会用到IoU或者说是交并比。非最大值抑制（non-max suppression） 这个工具可以用来让YOLO的结果变得更理想。

#### Non-max Suppression

目前所学到的目标检测的问题之一是，算法或许会对同一目标有多次检测。非极大值抑制是一种确保算法只对每个对象得到一个检测结果的方法。

非极大值抑制是如何工作的？

首先，是看一看每个检测结果的相关概率。候选者Pc为一个检测到的概率。首先它取其中最大那个, 如下图中左边的例子是0.9, 意味着 "这是我最自信的检测结果了, 那么让我们标明它, 认为我在这里找到一辆车." 

做完这一步, 非极大值抑制再看，所有的剩下的方框以及所有和刚输入的那个结果有着着多重叠的, 有着高IOU值的方形区域得到的产出值将被抑制。就是那二个概率为0,6和0.7的方框。这二个和亮蓝色的方框重叠最多。所以这些将要做抑制。

{% asset_img  non_max_suppression_example.jpg Non-max suppression example %}

非极大值意思是将要输出有着最大可能性的分类判断，而抑制那些非最大可能性的邻近的方框。因此叫做非极大值抑制。


过一下这个算法的细节。为了运用非极大值抑制： 

- 首先要做的是丢掉所有预测值Pc小于或等于 某个门限的边界框, 例如0.6.
- 接下来, 如果还有剩下的边界框,还没有被去掉或处理的, 将重复地选出有着最大概率最大Pc值的边界框, 将它作为一个预测结果.
- 然后，要丢掉其他剩余边界框, 即那些不被认为是预测结果的, 并且之前也没有被去掉的框。因此丢弃任何“剩余的，同上一步的计算有着着多重叠的, 高IOU值的边框”。
- 需要重复这个过程, 直到每个边界框不是被输出为预测结果, 就是被丢弃掉, 由于它们有着很大的重复或很高的IOU值。和刚刚检测输出的目标检测来作为你检测到的目标相比。

{% asset_img  non_max_suppression_algorithm.jpg Non-max suppression algorithm %}

#### Anchor Boxes

前面的例子中，每个网格都是检测单一目标，但实际应用中，很可能一个网格含有多个目标。下面讲到的Anchor Boxes（锚框）可以解决此问题。

如下图所示，示例图片中，一个网格包含两个目标物体，通过两个锚框，分别对应两个目标。对应的label的大小是3×3×16，分别对应2个锚框检测的结果。

{% asset_img overlapping_objects.jpg Overlapping Objects %}

在此之前，每个目标在训练集中，分配在包含这个目标中心点的网格中。有了锚框算法后，每个目标在训练集中，这个目标分配的网格不仅要包含这个目标中心点，还要和对应的锚框有最高的IOU值。

{% asset_img  anchor_box_algorithm.jpg Anchor box algorithm %}

下图是一个示例，对于车这个目标，和锚框2的IOU值更高，所以对应到锚框2的位置，人和锚框1的IOU值更高，对应到锚框1的位置。

{% asset_img anchor_box_example.jpg Anchor box example %}

还有一些细节，有两个锚框，但是万一同一格子裡有三个物件呢？此算法对这个例子没办法处理好，希望这不会发生，不过如果真有其事，此算法并没有好法子处理。对这种情形，会写某种挑选的机制。 还有，万一有两个物件在同一个格子，但是两个都和同一个锚框的形状一样呢？ 同样地，此算法也无法处理好这情况。如果发生这种情况，可以做一些挑选的机制 (tiebreaking)。希望资料集不会出现这种例子，希望不会常发生。 但实际上，这很少发生。特别是当用19×19，而不只是 3×3 格子，在 361 个格子有两个物件的中心点在同一个格子，的确有这机会，但概率很小。 

最后，怎么选择锚框呢？大家曾经是手动挑选，可能设计五或十个锚框，让他们有各种不同形状，看起来能涵盖想侦测的物件种类。 而更进阶的版本，是利用 K-means 算法 (K-平均)： 把两种想侦测的物件的形状集合起来，然后利用k-平均算法，挑选出一些锚框，最具代表性的、让他们能展现出想侦测的、多种各个类别的物件。 不过这种自动选择锚框的方法比较进阶。如果手动挑选各式各样的形状，能够扩展出想侦测的物件形状，想找高的、 瘦的、胖的宽的... 应该也能表现不错。

#### YOLO Algorithm

把上面学到的东西，组合回 YOLO 算法。

首先是训练。下图中给出了训练集中label的样式。网格划分是3×3，有两个锚框，检测的目标物体是3种类型，那么label的大小是3×3×2×8。

{% asset_img yolo_training.jpg Training %}

然后是进行预测，预测的结果同样是3×3×2×8的大小。

{% asset_img making_predictions.jpg Making predictions %}

下图是一个预测后的结果，通过去除低概率的预测值，再通过非最大值抑制算法，得到最终的结果。

{% asset_img  outputting_the_non-max_supressed_outputs.jpg Outputting the non-max supressed outputs %}

#### (Optional) Region Proposals

Region Proposal(候选区域)在计算机视觉中也非常流行。R-CNN 算法中用到了 Region Proposal,它的意思是，伴随区域的卷积网络或者伴随区域的CNN。这个算法所做的是，它会尝试选取，仅仅是少许有意义的区域，用来进行目标检测。所以，相较于运行每一个移动窗口，可以用选取少许窗口的方式来代替。所用的运行区域候选的方法，是通过运行所谓的分割算法来实现的。

{% asset_img  region_proposal.jpg Region proposal %}

下图是R-CNN算法及其进阶版本的介绍。

{% asset_img  rcnn_algorithms.jpg R-CNN algorithm %}

## 编程练习

### Autonomous driving - Car detection

Welcome to your week 3 programming assignment. You will learn about object detection using the very powerful YOLO model. Many of the ideas in this notebook are described in the two YOLO papers: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640) and Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242). 

**You will learn to**:
- Use object detection on a car detection dataset
- Deal with bounding boxes

Run the following cell to load the packages and dependencies that are going to be useful for your journey!

```python 

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

get_ipython().magic('matplotlib inline')
```

**Important Note**: As you can see, we import Keras's backend as K. This means that to use a Keras function in this notebook, you will need to write: `K.function(...)`.

#### 1 - Problem Statement

You are working on a self-driving car. As a critical component of this project, you'd like to first build a car detection system. To collect data, you've mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds while you drive around. 

{% asset_img  road_video_compressed2.gif Pictures taken from a car-mounted camera while driving around Silicon Valley.  We would like to especially thank [drive.ai](https://www.drive.ai/) for providing this dataset! Drive.ai is a company building the brains of self-driving vehicles. %}


{% asset_img  driveai.png  %}

You've gathered all these images into a folder and have labelled them by drawing bounding boxes around every car you found. Here's an example of what your bounding boxes look like.

{% asset_img  box_label.png Figure 1: Definition of a box  %}


If you have 80 classes that you want YOLO to recognize, you can represent the class label $c$ either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1 and the rest of which are 0. The video lectures had used the latter representation; in this notebook, we will use both representations, depending on which is more convenient for a particular step.  

In this exercise, you will learn how YOLO works, then apply it to car detection. Because the YOLO model is very computationally expensive to train, we will load pre-trained weights for you to use. 

#### 2 - YOLO

YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

##### 2.1 - Model details

First things to know:
- The **input** is a batch of images of shape (m, 608, 608, 3)
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers $(p_c, b_x, b_y, b_h, b_w, c)$ as explained above. If you expand $c$ into an 80-dimensional vector, each bounding box is then represented by 85 numbers. 

We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

Lets look in greater detail at what this encoding represents. 

{% asset_img  architecture.png Figure 2:Encoding architecture for YOLO %}


If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

{% asset_img  flatten.png Figure 3: Flattening the last two last dimensions %}


Now, for each box (of each cell) we will compute the following elementwise product and extract a probability that the box contains a certain class.

{% asset_img  probability_extraction.png Figure 4:Find the class detected by each box %}


Here's one way to visualize what YOLO is predicting on an image:
- For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across both the 5 anchor boxes and across different classes). 
- Color that grid cell according to what object that grid cell considers the most likely.

Doing this results in this picture: 

{% asset_img  proba_map.png Figure 5 : Each of the 19x19 grid cells colored according to which class has the largest predicted probability in that cell. %}


Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm. 


Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:  

{% asset_img  anchor_map.png Figure 6 : Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image (one forward pass through the network)! Different colors denote different classes. %}


In the figure above, we plotted only boxes that the model had assigned a high probability to, but this is still too many boxes. You'd like to filter the algorithm's output down to a much smaller number of detected objects. To do so, you'll use non-max suppression. Specifically, you'll carry out these steps: 
- Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class)
- Select only one box when several boxes overlap with each other and detect the same object.



##### 2.2 - Filtering with a threshold on class scores

You are going to apply a first filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold. 

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It'll be convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:  
- `box_confidence`: tensor of shape $(19 \times 19, 5, 1)$ containing $p_c$ (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
- `boxes`: tensor of shape $(19 \times 19, 5, 4)$ containing $(b_x, b_y, b_h, b_w)$ for each of the 5 boxes per cell.
- `box_class_probs`: tensor of shape $(19 \times 19, 5, 80)$ containing the detection probabilities $(c_1, c_2, ... c_{80})$ for each of the 80 classes for each of the 5 boxes per cell.

**Exercise**: Implement `yolo_filter_boxes()`.
1. Compute box scores by doing the elementwise product as described in Figure 4. The following code may help you choose the right operator: 
```python
a = np.random.randn(19*19, 5, 1)
b = np.random.randn(19*19, 5, 80)
c = a * b # shape of c will be (19*19, 5, 80)
```
2. For each box, find:
    - the index of the class with the maximum box score ([Hint](https://keras.io/backend/#argmax)) (Be careful with what axis you choose; consider using axis=-1)
    - the corresponding box score ([Hint](https://keras.io/backend/#max)) (Be careful with what axis you choose; consider using axis=-1)
3. Create a mask by using a threshold. As a reminder: `([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4)` returns: `[False, True, False, False, True]`. The mask should be True for the boxes you want to keep. 
4. Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. You should be left with just the subset of boxes you want to keep. ([Hint](https://www.tensorflow.org/api_docs/python/tf/boolean_mask))

Reminder: to call a Keras function, you should use `K.function(...)`.

```python 

#GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    ### END CODE HERE ###
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask = box_class_scores >= threshold
    ### END CODE HERE ###
    
    # Step 4: Apply the mask to scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### END CODE HERE ###
    
    return scores, boxes, classes
```

##### 2.3 - Non-max suppression ###

Even after filtering by thresholding over the classes scores, you still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS). 

{% asset_img non-max-suppression.png Figure 7 : In this example, the model has predicted 3 cars, but it's actually 3 predictions of the same car. Running non-max suppression (NMS) will select only the most accurate (highest probabiliy) one of the 3 boxes.  %}

Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.

{% asset_img  iou.png  Figure 8 : Definition of "Intersection over Union %}

**Exercise**: Implement iou(). Some hints:
- In this exercise only, we define a box using its two corners (upper left and lower right): `(x1, y1, x2, y2)` rather than the midpoint and height/width.
- To calculate the area of a rectangle you need to multiply its height `(y2 - y1)` by its width `(x2 - x1)`.
- You'll also need to find the coordinates `(xi1, yi1, xi2, yi2)` of the intersection of two boxes. Remember that:
    - xi1 = maximum of the x1 coordinates of the two boxes
    - yi1 = maximum of the y1 coordinates of the two boxes
    - xi2 = minimum of the x2 coordinates of the two boxes
    - yi2 = minimum of the y2 coordinates of the two boxes
- In order to compute the intersection area, you need to make sure the height and width of the intersection are positive, otherwise the intersection area should be zero. Use `max(height, 0)` and `max(width, 0)`.

In this code, we use the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) the lower-right corner. 

```python 

#GRADED FUNCTION: iou

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = (yi2-yi1)*(xi2-xi1) if (max(yi2-yi1,0) > 0 and max(xi2-xi1,0) > 0) else 0
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] -box1[1]) * (box1[2] -box1[0])
    box2_area = (box2[3] -box2[1]) * (box2[2] -box2[0])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = 1.0 * inter_area / union_area
    ### END CODE HERE ###
    
    return iou
```

You are now ready to implement non-max suppression. The key steps are: 
1. Select the box that has the highest score.
2. Compute its overlap with all other boxes, and remove boxes that overlap it more than `iou_threshold`.
3. Go back to step 1 and iterate until there's no more boxes with a lower score than the current selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

**Exercise**: Implement yolo_non_max_suppression() using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so you don't actually need to use your `iou()` implementation):
- [tf.image.non_max_suppression()](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
- [K.gather()](https://www.tensorflow.org/api_docs/python/tf/gather)

```python 

#GRADED FUNCTION: yolo_non_max_suppression

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes, scores,max_boxes,iou_threshold)
    ### END CODE HERE ###
    
    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.gather(scores,nms_indices)
    boxes = tf.gather(boxes,nms_indices)
    classes = tf.gather(classes,nms_indices)
    ### END CODE HERE ###
    
    return scores, boxes, classes
```

##### 2.4 Wrapping up the filtering

It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions you've just implemented. 

**Exercise**: Implement `yolo_eval()` which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which we have provided): 

```python
boxes = yolo_boxes_to_corners(box_xy, box_wh) 
```
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of `yolo_filter_boxes`
```python
boxes = scale_boxes(boxes, image_shape)
```
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image--for example, the car detection dataset had 720x1280 images--this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.  

Don't worry about these two functions; we'll show you where they need to be called.  

```python 

#GRADED FUNCTION: yolo_eval

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    ### START CODE HERE ### 
    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes,max_boxes,iou_threshold)
    
    ### END CODE HERE ###
    
    return scores, boxes, classes
```

```python 

with tf.Session() as test_b:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))
```

<font color='blue'>
**Summary for YOLO**:
- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
    - Each cell in a 19x19 grid over the input image gives 425 numbers. 
    - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
    - 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and and 80 is the number of classes we'd like to detect
- You then select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output. 

#### 3 - Test YOLO pretrained model on images

In this part, you are going to use a pretrained model and test it on the car detection dataset. As usual, you start by **creating a session to start your graph**. Run the following cell.

```python 

sess = K.get_session()
```

##### 3.1 - Defining classes, anchors and image shape.

Recall that we are trying to detect 80 classes, and are using 5 anchor boxes. We have gathered the information about the 80 classes and 5 boxes in two files "coco_classes.txt" and "yolo_anchors.txt". Let's load these quantities into the model by running the next cell. 

The car detection dataset has 720x1280 images, which we've pre-processed into 608x608 images. 

```python 

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    
```

##### 3.2 - Loading a pretrained model

Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. You are going to load an existing pretrained Keras YOLO model stored in "yolo.h5". (These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this notebook. Technically, these are the parameters from the "YOLOv2" model, but we will more simply refer to it as "YOLO" in this notebook.) Run the cell below to load the model from this file.

```python 

yolo_model = load_model("model_data/yolo.h5")
```

This loads the weights of a trained YOLO model. Here's a summary of the layers your model contains.

```python 

yolo_model.summary()
```

**Note**: On some computers, you may see a warning message from Keras. Don't worry about it if you do--it is fine.

**Reminder**: this model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85) as explained in Figure (2).

##### 3.3 - Convert output of the model to usable bounding box tensors

The output of `yolo_model` is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. The following cell does that for you.

```python 

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
```

You added `yolo_outputs` to your graph. This set of 4 tensors is ready to be used as input by your `yolo_eval` function.

##### 3.4 - Filtering boxes

`yolo_outputs` gave you all the predicted boxes of `yolo_model` in the correct format. You're now ready to perform filtering and select only the best boxes. Lets now call `yolo_eval`, which you had previously implemented, to do this. 

```python 

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
```

##### 3.5 - Run the graph on an image

Let the fun begin. You have created a (`sess`) graph that can be summarized as follows:

1. <font color='purple'> yolo_model.input </font> is given to `yolo_model`. The model is used to compute the output <font color='purple'> yolo_model.output </font>
2. <font color='purple'> yolo_model.output </font> is processed by `yolo_head`. It gives you <font color='purple'> yolo_outputs </font>
3. <font color='purple'> yolo_outputs </font> goes through a filtering function, `yolo_eval`. It outputs your predictions: <font color='purple'> scores, boxes, classes </font>

**Exercise**: Implement predict() which runs the graph to test YOLO on an image.
You will need to run a TensorFlow session, to have it compute `scores, boxes, classes`.

The code below also uses the following function:
```python
image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
```
which outputs:
- image: a python (PIL) representation of your image used for drawing boxes. You won't need to use it.
- image_data: a numpy-array representing the image. This will be the input to the CNN.

**Important note**: when a model uses BatchNorm (as is the case in YOLO), you will need to pass an additional placeholder in the feed_dict {K.learning_phase(): 0}.

```python 

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    ### START CODE HERE ### (≈ 1 line)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data , K.learning_phase(): 0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes
```

Run the following cell on the "test.jpg" image to verify that your function is correct.

```python 

out_scores, out_boxes, out_classes = predict(sess, "test.jpg")
```

**Expected Output**:

<table>
    <tr>
        <td>
            **Found 7 boxes for test.jpg**
        </td>
    </tr>
    <tr>
        <td>
            **car**
        </td>
        <td>
           0.60 (925, 285) (1045, 374)
        </td>
    </tr>
    <tr>
        <td>
            **car**
        </td>
        <td>
           0.66 (706, 279) (786, 350)
        </td>
    </tr>
    <tr>
        <td>
            **bus**
        </td>
        <td>
           0.67 (5, 266) (220, 407)
        </td>
    </tr>
    <tr>
        <td>
            **car**
        </td>
        <td>
           0.70 (947, 324) (1280, 705)
        </td>
    </tr>
    <tr>
        <td>
            **car**
        </td>
        <td>
           0.74 (159, 303) (346, 440)
        </td>
    </tr>
    <tr>
        <td>
            **car**
        </td>
        <td>
           0.80 (761, 282) (942, 412)
        </td>
    </tr>
    <tr>
        <td>
            **car**
        </td>
        <td>
           0.89 (367, 300) (745, 648)
        </td>
    </tr>
</table>

The model you've just run is actually able to detect 80 different classes listed in "coco_classes.txt". To test the model on your own images:
    1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
    2. Add your image to this Jupyter Notebook's directory, in the "images" folder
    3. Write your image's name in the cell above code
    4. Run the code and see the output of the algorithm!

If you were to run your session in a for loop over all your images. Here's what you would get:

{% asset_img pred_video_compressed2.gif Predictions of the YOLO model on pictures taken from a camera while driving around the Silicon Valley <br> Thanks [drive.ai](https://www.drive.ai/) for providing this dataset! %}

<font color='blue'>
**What you should remember**:
- YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume. 
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppression. Specifically: 
    - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise. 

**References**: The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's github repository. The pretrained weights used in this exercise came from the official YOLO website. 
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (2015)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (2016)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/) 

**Car detection dataset**:
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">The Drive.ai Sample Dataset</span> (provided by drive.ai) is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. We are especially grateful to Brody Huval, Chih Hu and Rahul Patel for collecting and providing this dataset. 
