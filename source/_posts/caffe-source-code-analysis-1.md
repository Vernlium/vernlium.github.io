---
title: caffe_source_code_analysis_1
mathjax: true
date: 2018-11-03 14:22:52
tags:
---


## Caffe源码解析

Caffe作为一个出现较早、使用较广的深度学习框架，其代码是值得深入学习和研究的。

下面从数据结构、执行流程、算子等几个维度来分析一下Caffe的源码。

### 数据结构

深度学习的构成元素无非是网络、算子、各种参数等。网络由算子按照一定顺序组合，输入数据经过网络计算，计算时需要各种学习得到的参数，最终得到输出。输入数据和各种参数都可以看作是数据。其实也就是一些数据，在网络中流动（包括前向和后向传播，推理时只有前向传播，训练时包括前向和后向传播），网络是由一个个的算子组成。每个算子都有一个输入和输出，网络中上一个算子输出可以作为下一个算子的输入。

Caffe中的数据结构基本上与这三种元素对应，最重要的是三个：

- Blob: 数据管理
- Layer: 构成深度学习网络的基础单元
- Net: 定义深度学习网络模型

除此之外，还有其他的数据结构，如：

- Loss: 定义损失函数
- Solver: 定义各种优化算法

#### Blob

Blob包装了在Caffe中被传播和处理的实际数据，同时也提供了数据在CPU和GPU上进行同步的能力。从数学上来讲，Blob就是一个N维的数组。

Caffe中所有数据的存储和流动使用的都是Blob，它提供了对数据进行管理的统一内存接口。保存是数据有：输入的图像数据、模型的参数、后向传播中的梯度等。

对于输入图像数据，传统的blob维度是：$N * C * H * W$。Blob在布局上是行优先的，因此在最后/最右边的维度变化是最快的。Blob常见的维度是4维，但也不完全都是4维，比如进行网络中的全连接层计算时，调用InnerProductLayer层使用的就是2维的Blob。

##### 实现细节

Blob类中包含如下成员变量：

```c++
template <typename Dtype>
class Blob {
 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob
}  // namespace caffe
```

> 注：这里主要时为了介绍概念，只看数据结构，不看具体逻辑。类的方法后面用到时再看。

Blob类中包含了`data_`和`diff_`两个字段，分别表示Blob存储的两块内存：数据和梯度。

shape_中存储的是数据的shape信息。

值得注意的是，真实的数据可能存储在CPU或GPU上，有两种不同的方式可以访问它们：

- The const way: 不会改变数据
- The mutable way: 会改变数据

```c++
const Dtype* cpu_data() const;
Dtype* mutable_cpu_data();
//similarly for gpu and diff
```

这样设计的目的是，当Blob使用SyncedMem类在CPU和GPU之间进行数据同步时，可以隐藏同步的细节和最小化数据传输。一个常用的规则是：

> always use the const call if you do not want to change the values, and never store the pointers in your own object. 

#### Layer

Layer是模型的本质和计算的基本单元。

下图描述了Layer和Blob的关系，`bottom blob`作为layer的输入，`top blob`作为layer的输出。

{% asset_img  layer.jpg Layer and Blob %}

每个Layer定义了3个重要的计算：`setup,forword,backward`.

- Setup: 模型初始化后，初始化每个层及其连接关系
- Forward: 从bottom中获取输入数据，计算输出，并发送给top blob
- Backward: 从top中获取梯度w.r.t，计算梯度w.r.t，并发送给bottom。对有参数的layer，计算其参数的梯度w.r.t，并在其内部存储。

通过Layer类的定义也可以看到这三个函数：

```c++

namespace caffe {
template <typename Dtype>
class Layer {
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        CheckBlobCounts(bottom, top);
        LayerSetUp(bottom, top);
        Reshape(bottom, top);
        SetLossWeights(top);
      }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

 protected:
  LayerParameter layer_param_;
  Phase phase_;
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<bool> param_propagate_down_;

  vector<Dtype> loss_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}
};  // class Layer
}  // namespace caffe
```

Layer类的成员变量中，`layer_param_`保存的是从caffe.prototxt文件中解析得到的layer的各种参数。layer在Setup中进行初始化时就要从中获取参数。

可以看到Layer的Setup函数已经实现，其中调用了`LayerSetUp`和`Reshape`两个函数，这两个函数都是虚函数，需要子类实现。不同的Layer，参数时不同的，初始化也不同，所以必须由子类进行实现，而Setup的流程时一样的。

同时，Forward和Backward函数都有两个实现，分别是使用CPU和GPU。如果没有实现GPU的版本，layer会使用CPU版本的实现，这样做可能会导致额外的数据传输成本。

Layer类有两个重要的作用：

- Forward pass: 获取输入并计算输出
- Backward pass: 获取输出的梯度，并计算相关参数的梯度，传递给输入，通过反向传播依次传递到前面的layer。

通过这种模块化的设计，开发者可以自定义layer，只需要按照这种模式实现Setup，forward,backward即可。

#### Net

Net由计算图中的连接着的一组layer组成。计算图是一个有向无环图。

一个典型的Net一般是由一个data layer开始，由一个计算目标（比如分类等）的loss layer结束。

Net由prototxt文件进行定义，prototxt中定义一组layer和layer之间的连接关系。

比如一个逻辑回归的模型通过如下的文件定义：

```
name: "LogReg"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "input_leveldb"
    batch_size: 64
  }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
}
```

它定义的Net，可以表示为：

{% asset_img  logreg.jpg Logistic Regression Classifier %}

通过调用Net::Init()对模型进行初始化。初始化中只要做了两件事：

- 通过创建Blob和Layer，把整个Net的有向无环图构建出来 
- 调用Layer的SetUp()函数

同时也会做一些其他的bookkeeping操作，比如验证整个网络结构的正确性等。

值得注意的是，网络是与设备无关的。网络运行在CPU或者GPU上，通过定义在 `Caffe::mode()`中的开关控制。通过`Caffe::set_mode()`进行设置。

##### 实现细节

```c++
namespace caffe {
template <typename Dtype>
class Net {
 protected:
  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  // Callbacks
  vector<Callback*> before_forward_;
  vector<Callback*> after_forward_;
  vector<Callback*> before_backward_;
  vector<Callback*> after_backward_;

}  // namespace caffe

```

其中：

- vector<shared_ptr<Layer<Dtype> > > layers_ : 定义了Net的所有层
- vector<string> layer_names_ : 每一层的name
- map<string, int> layer_names_index_ : 层和name之间的对应关系
- vector<bool> layer_need_backward_ : 层是否需要进行后向计算
- vector<shared_ptr<Blob<Dtype> > > blobs_ : 保存层之间的中间结果的blob
- vector<string> blob_names_ : blob的名字
- map<string, int> blob_names_index_ ： blob和name之间的对应关系

##### Model format

模型由prototxt文件（plaintext protocol buffer schema）进行定义，学习好的模型通过序列化保存到.caffemodel（binary protocol buffer）二进制文件中

caffe使用google的Protocol buffer进行定义模型和存储模型，可以使得序列化后的模型文件较小，而且易于扩展。

#### Solver

Solver定义模型优化的算法，即在后向计算时如何更新使得loss更小。

Caffe中的Solver有：

- Stochastic Gradient Descent (type: "SGD"),
- AdaDelta (type: "AdaDelta"),
- Adaptive Gradient (type: "AdaGrad"),
- Adam (type: "Adam"),
- Nesterov’s Accelerated Gradient (type: "Nesterov") and
- RMSprop (type: "RMSProp")

每一次的迭代中，做了如下动作：

- 1.调用网络的forward方法计算输出和损失
- 2.调用网络的backward计算梯度
- 3.根据solver的方法进行梯度更新
- 4.根据学习率、历史纪录和方法更新solver的状态

### 小结

本文总结了caffe中的相关数据结构，下一篇将对caffe继续分析，主要从流程上看caffe的运行机制。