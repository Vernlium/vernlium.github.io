---
title: coursera-deeplearning-course_list
date: 2018-07-05 07:19:13
tags:
---

## course1:Neural Networks and Deep Learning

### c1_week1: Introduction to deep learning

Be able to explain the major trends driving the rise of deep learning, and understand where and how it is applied today.

#### 学习目标

- Understand the major trends driving the rise of deep learning.
- Be able to explain how deep learning is applied to supervised learning.
- Understand what are the major categories of models (such as CNNs and RNNs), and when they should be applied.
- Be able to recognize the basics of when deep learning will (or will not) work well.

#### Welcome to the Deep Learning Specialization

##### 课程视频Welcome

#### Introduction to Deep Learning

##### 课程视频What is a neural network?
##### 课程视频Supervised Learning with Neural Networks
##### 课程视频Why is Deep Learning taking off?
##### 课程视频About this Course
##### 阅读材料Frequently Asked Questions
##### 课程视频Course Resources
##### 阅读材料How to use Discussion Forums

### c1_week2:Neural Networks Basics

Learn to set up a machine learning problem with a neural network mindset. Learn to use vectorization to speed up your models.

#### 学习目标

- Build a logistic regression model, structured as a shallow neural network
- Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent.
- Implement computationally efficient, highly vectorized, versions of models.
- Understand how to compute derivatives for logistic regression, using a backpropagation mindset.
- Become familiar with Python and Numpy
- Work with iPython Notebooks
- Be able to implement vectorization across multiple training examples

#### Logistic Regression as a Neural Network

##### 课程视频Binary Classification
##### 课程视频Logistic Regression
##### 课程视频Logistic Regression Cost Function
##### 课程视频Gradient Descent
##### 课程视频Derivatives
##### 课程视频More Derivative Examples
##### 课程视频Computation graph
##### 课程视频Derivatives with a Computation Graph
##### 课程视频Logistic Regression Gradient Descent
##### 课程视频Gradient Descent on m Examples

#### Python and Vectorization

##### 课程视频Vectorization
##### 课程视频More Vectorization Examples
##### 课程视频Vectorizing Logistic Regression
##### 课程视频Vectorizing Logistic Regression's Gradient Output
##### 课程视频Broadcasting in Python
##### 课程视频A note on python/numpy vectors
##### 课程视频Quick tour of Jupyter/iPython Notebooks
##### 课程视频Explanation of logistic regression cost function (optional)


#### Programming Assignments

##### 编程作业:Python Basics with numpy (optional)
##### 编程作业: Logistic Regression with a Neural Network mindset

### c1_week3: Shallow neural networks

Learn to build a neural network with one hidden layer, using forward propagation and backpropagation.

#### 学习目标

- Understand hidden units and hidden layers
- Be able to apply a variety of activation functions in a neural network.
- Build your first forward and backward propagation with a hidden layer
- Apply random initialization to your neural network
- Become fluent with Deep Learning notations and Neural Network Representations
- Build and train a neural network with one hidden layer.


#### Shallow Neural Network

##### 课程视频Neural Networks Overview
##### 课程视频Neural Network Representation
##### 课程视频Computing a Neural Network's Output
##### 课程视频Vectorizing across multiple examples
##### 课程视频Explanation for Vectorized Implementation
##### 课程视频Activation functions
##### 课程视频Why do you need non-linear activation functions?
##### 课程视频Derivatives of activation functions
##### 课程视频Gradient descent for Neural Networks
##### 课程视频Backpropagation intuition (optional)
##### 课程视频Random Initialization

#### Programming Assignment

##### 编程作业: Planar data classification with a hidden layer

### c1_week4: Deep Neural Networks

Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

#### 学习目标

- See deep neural networks as successive blocks put one after each other
- Build and train a deep L-layer Neural Network
- Analyze matrix and vector dimensions to check neural network implementations.
- Understand how to use a cache to pass information from forward propagation to back propagation.
- Understand the role of hyperparameters in deep learning

#### Deep Neural Network

##### 课程视频Deep L-layer neural network
##### 课程视频Forward Propagation in a Deep Network
##### 课程视频Getting your matrix dimensions right
##### 课程视频Why deep representations?
##### 课程视频Building blocks of deep neural networks
##### 课程视频Forward and Backward Propagation
##### 课程视频Parameters vs Hyperparameters
##### 课程视频What does this have to do with the brain?

#### Programming Assignments

##### 编程作业: Building your deep neural network: Step by Step
##### 编程作业: Deep Neural Network Application

## course2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

### c2_week1: Practical aspects of Deep Learning

#### 学习目标

- Recall that different types of initializations lead to different results
- Recognize the importance of initialization in complex neural networks.
- Recognize the difference between train/dev/test sets
- Diagnose the bias and variance issues in your model
- Learn when and how to use regularization methods such as dropout or L2 regularization.
- Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
- Use gradient checking to verify the correctness of your backpropagation implementation

#### Setting up your Machine Learning Application

##### 课程视频Train / Dev / Test sets
##### 课程视频Bias / Variance
##### 课程视频Basic Recipe for Machine Learning

#### Regularizing your neural network

##### 课程视频Regularization
##### 课程视频Why regularization reduces overfitting?
##### 课程视频Dropout Regularization
##### 课程视频Understanding Dropout
##### 课程视频Other regularization methods

#### Setting up your optimization problem

##### 课程视频Normalizing inputs
##### 课程视频Vanishing / Exploding gradients
##### 课程视频Weight Initialization for Deep Networks
##### 课程视频Numerical approximation of gradients
##### 课程视频Gradient checking
##### 课程视频Gradient Checking Implementation Notes

#### Programming assignments

##### 编程作业: Initialization
##### 编程作业: Regularization
##### 编程作业: Gradient Checking

### c2_week2: Optimization algorithms

#### 学习目标

- Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate the convergence and improve the optimization
- Know the benefits of learning rate decay and apply it to your optimization

#### Optimization algorithms

##### 课程视频Mini-batch gradient descent
##### 课程视频Understanding mini-batch gradient descent
##### 课程视频Exponentially weighted averages
##### 课程视频Understanding exponentially weighted averages
##### 课程视频Bias correction in exponentially weighted averages
##### 课程视频Gradient descent with momentum
##### 课程视频RMSprop
##### 课程视频Adam optimization algorithm
##### 课程视频Learning rate decay
##### 课程视频The problem of local optima

#### Programming assignment

##### 编程作业: Optimization

### c2_week3: Hyperparameter tuning, Batch Normalization and Programming Frameworks

#### 学习目标

- Master the process of hyperparameter tuning

#### Hyperparameter tuning

##### 课程视频Tuning process
##### 课程视频Using an appropriate scale to pick hyperparameters
##### 课程视频Hyperparameters tuning in practice: Pandas vs. Caviar

#### Batch Normalization

##### 课程视频Normalizing activations in a network
##### 课程视频Fitting Batch Norm into a neural network
##### 课程视频Why does Batch Norm work?
##### 课程视频Batch Norm at test time

#### Multi-class classification

##### 课程视频Softmax Regression
##### 课程视频Training a softmax classifier

#### Introduction to programming frameworks

##### 课程视频Deep learning frameworks
##### 课程视频TensorFlow

#### Programming assignment

##### 编程作业: Tensorflow

## course3: Structuring Machine Learning Projects

### c3_week1: ML Strategy (1)

#### 学习目标

- Understand why Machine Learning strategy is important
- Apply satisficing and optimizing metrics to set up your goal for ML projects
- Choose a correct train/dev/test split of your dataset
- Understand how to define human-level performance
- Use human-level perform to define your key priorities in ML projects
- Take the correct ML Strategic decision based on observations of performances and dataset

#### Introduction to ML Strategy

##### 课程视频Why ML Strategy
##### 课程视频Orthogonalization

#### Setting up your goal

##### 课程视频Single number evaluation metric
##### 课程视频Satisficing and Optimizing metric
##### 课程视频Train/dev/test distributions
##### 课程视频Size of the dev and test sets
##### 课程视频When to change dev/test sets and metrics

#### Comparing to human-level performance

##### 课程视频Why human-level performance?
##### 课程视频Avoidable bias
##### 课程视频Understanding human-level performance
##### 课程视频Surpassing human-level performance
##### 课程视频Improving your model performance

#### Machine Learning flight simulator

##### 阅读材料Machine Learning flight simulator
##### 测验: Bird recognition in the city of Peacetopia (case study)

### c3_week2: ML Strategy (2)

#### 学习目标

- Understand what multi-task learning and transfer learning are
- Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets

#### Error Analysis

##### 课程视频Carrying out error analysis
##### 课程视频Cleaning up incorrectly labeled data
##### 课程视频Build your first system quickly, then iterate

#### Mismatched training and dev/test set

##### 课程视频Training and testing on different distributions
##### 课程视频Bias and Variance with mismatched data distributions
##### 课程视频Addressing data mismatch

#### Learning from multiple tasks

##### 课程视频Transfer learning
##### 课程视频Multi-task learning

#### End-to-end deep learning

##### 课程视频What is end-to-end deep learning?
##### 课程视频Whether to use end-to-end deep learning

#### Machine Learning flight simulator

##### 测验: Autonomous driving (case study)

## course4: Convolutional Neural Networks

### c4_week1: Foundations of Convolutional Neural Networks

Learn to implement the foundational layers of CNNs (pooling, convolutions) and to stack them properly in a deep network to solve multi-class image classification problems.

#### 学习目标
- Understand the convolution operation
- Understand the pooling operation
- Remember the vocabulary used in convolutional neural network (padding, stride, filter, ...)
- Build a convolutional neural network for image multi-class classification

#### Convolutional Neural Networks

##### 课程视频Computer Vision
##### 课程视频Edge Detection Example
##### 课程视频More Edge Detection
##### 课程视频Padding
##### 课程视频Strided Convolutions
##### 课程视频Convolutions Over Volume
##### 课程视频One Layer of a Convolutional Network
##### 课程视频Simple Convolutional Network Example
##### 课程视频Pooling Layers
##### 课程视频CNN Example
##### 课程视频Why Convolutions?

#### Programming assignments

##### 编程作业: Convolutional Model: step by step
##### 编程作业: Convolutional model: application
 
### c4_week2:  Deep convolutional models: case studies

Learn about the practical tricks and methods used in deep CNNs straight from the research papers.

#### 学习目标

- Understand multiple foundational papers of convolutional neural networks
- Analyze the dimensionality reduction of a volume in a very deep network
- Understand and Implement a Residual network
- Build a deep neural network using Keras
- Implement a skip-connection in your network
- Clone a repository from github and use transfer learning

#### Case studies

##### 课程视频Why look at case studies?
##### 课程视频Classic Networks
##### 课程视频ResNets
##### 课程视频Why ResNets Work
##### 课程视频Networks in Networks and 1x1 Convolutions
##### 课程视频Inception Network Motivation
##### 课程视频Inception Network

#### Practical advices for using ConvNets

##### 课程视频Using Open-Source Implementation
##### 课程视频Transfer Learning
##### 课程视频Data Augmentation
##### 课程视频State of Computer Vision

#### Programming assignments

##### 编程作业: Keras Tutorial - The Happy House (not graded)
##### 编程作业: Residual Networks

### c4_week3: Object detection

Learn how to apply your knowledge of CNNs to one of the toughest but hottest field of computer vision: Object detection.

#### 学习目标

- Understand the challenges of Object Localization, Object Detection and Landmark Finding
- Understand and implement non-max suppression
- Understand and implement intersection over union
- Understand how we label a dataset for an object detection application
- Remember the vocabulary of object detection (landmark, anchor, bounding box, grid, ...)

#### Detection algorithms

##### 课程视频Object Localization
##### 课程视频Landmark Detection
##### 课程视频Object Detection
##### 课程视频Convolutional Implementation of Sliding Windows
##### 课程视频Bounding Box Predictions
##### 课程视频Intersection Over Union
##### 课程视频Non-max Suppression
##### 课程视频Anchor Boxes
##### 课程视频YOLO Algorithm
##### 课程视频(Optional) Region Proposals

#### Programming assignments

##### 编程作业: Car detection with YOLOv2

### c4_week4: Special applications: Face recognition & Neural style transfer

Discover how CNNs can be applied to multiple fields, including art generation and face recognition. Implement your own algorithm to generate art and recognize faces!

#### Face Recognition

##### 课程视频What is face recognition?
##### 课程视频One Shot Learning
##### 课程视频Siamese Network
##### 课程视频Triplet Loss
##### 课程视频Face Verification and Binary Classification

#### Neural Style Transfer

##### 课程视频What is neural style transfer?
##### 课程视频What are deep ConvNets learning?
##### 课程视频Cost Function
##### 课程视频Content Cost Function
##### 课程视频Style Cost Function
##### 课程视频1D and 3D Generalizations

#### Programming assignments

##### 编程作业: Art generation with Neural Style Transfer
##### 编程作业: Face Recognition for the Happy House


## course5: Sequence Models

### c5_week1: Recurrent Neural Networks

Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

#### Recurrent Neural Networks

##### 课程视频Why sequence models
##### 课程视频Notation
##### 课程视频Recurrent Neural Network Model
##### 课程视频Backpropagation through time
##### 课程视频Different types of RNNs
##### 课程视频Language model and sequence generation
##### 课程视频Sampling novel sequences
##### 课程视频Vanishing gradients with RNNs
##### 课程视频Gated Recurrent Unit (GRU)
##### 课程视频Long Short Term Memory (LSTM)
##### 课程视频Bidirectional RNN
##### 课程视频Deep RNNs

#### Programming assignments

##### 编程作业: Building a recurrent neural network - step by step
##### 编程作业: Dinosaur Island - Character-Level Language Modeling
##### 编程作业: Jazz improvisation with LSTM

### c5_week2: Natural Language Processing & Word Embeddings

Natural language processing with deep learning is an important combination. Using word vector representations and embedding layers you can train recurrent neural networks with outstanding performances in a wide variety of industries. Examples of applications are sentiment analysis, named entity recognition and machine translation.

#### Introduction to Word Embeddings

##### 课程视频Word Representation
##### 课程视频Using word embeddings
##### 课程视频Properties of word embeddings
##### 课程视频Embedding matrix

#### Learning Word Embeddings: Word2vec & GloVe

##### 课程视频Learning word embeddings
##### 课程视频Word2Vec
##### 课程视频Negative Sampling
##### 课程视频GloVe word vectors

#### Applications using Word Embeddings

##### 课程视频Sentiment Classification
##### 课程视频Debiasing word embeddings

#### Programming assignments

##### 编程作业: Operations on word vectors - Debiasing
##### 编程作业: Emojify

### c5_week3: Sequence models & Attention mechanism

Sequence models can be augmented using an attention mechanism. This algorithm will help your model understand where it should focus its attention given a sequence of inputs. This week, you will also learn about speech recognition and how to deal with audio data.

#### Various sequence to sequence architectures

##### 课程视频Basic Models
##### 课程视频Picking the most likely sentence
##### 课程视频Beam Search
##### 课程视频Refinements to Beam Search
##### 课程视频Error analysis in beam search
##### 课程视频Bleu Score (optional)
##### 课程视频Attention Model Intuition
##### 课程视频Attention Model

#### Speech recognition - Audio data

##### 课程视频Speech recognition
##### 课程视频Trigger Word Detection

#### Conclusion

##### 课程视频Conclusion and thank you

#### Programming assignments

##### 编程作业: Neural Machine Translation with Attention
##### 编程作业: Trigger word detection
