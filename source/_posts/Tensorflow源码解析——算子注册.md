---
title: Tensorflow源码解析——算子注册
date: 2019-07-01 06:38:44
tags: tensorflow 源码分析
---

### 什么是op

op和kernel是TF框架中最重要的两个概念，如果一定要做一个类比的话，可以认为op相当于函数声明，kernel相当于函数实现。举个例子，对于矩阵相乘，可以声明一个op叫做MatMul，指明它的名称，输入，输出，参数，以及对参数的限制等。op只是告诉我们，这个操作的目的是什么，op内部有哪些可定制的东西，但不会提供具体实现。op在某种设备上的具体实现方法，是由kernel决定的。TF的计算图由节点构成，而每个节点对应了一个op，在构建计算图时，我们只知道不同节点对应的操作是什么，而不知道运行时这个操作是怎样实现的。也就是说，op是编译期概念，而kernel是运行期概念。

那为什么要把操作和它的实现分离呢？是为了实现TF代码的可移植性。我们可以把TF构建的计算图想象为Java的字节码，而计算图在执行的时候，需要考虑可用的设备资源，相当于我们在运行Java字节码的时候，需要考虑当前所在的操作系统，选择合适的字节码实现。因为TF的目标是在多设备上运行，但我们在编码的时候，是无法预先知道某一个操作具体是在哪种设备上运行的，因此，将op和它的实现分离，可以让我们在设计计算图的时候，更专注于它的结构，而不是具体实现。当我们构建完成一个计算图之后，在一个包含GPU的设备上，它可以利用对应操作在GPU上的kernel，充分利用GPU的高计算性能，在一个仅包含CPU的设备上，它也可以利用对应操作在CPU上的kenrel，完成计算功能。这就提高了TF代码在不同设备之间的可移植性。

### 注册方式

下面是tensorflow代码中注册`Argmax`算子的代码：

```c++
REGISTER_OP("ArgMax")
    .Input("input: T")
    .Input("dimension: Tidx")
    .Output("output: output_type")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("output_type: {int32, int64} = DT_INT64")
    .SetShapeFn(ArgOpShape);
```

通过`REGISTER_OP`宏进行算子注册，注册的内容有:

- Input:算子的输入
- Output:算子的输出
- Attr:算子的属性，比如Argmax算子，有个属性是axis，在哪根轴上求最大值的下标
- ShapeFn:用于shape推断

下面分析这个算子是如何被注册进去的。

### OpDef

OpDef的定义在`tensorflow\core\framework\op_def.proto`中

```c++
message OpDef {
  // Op names starting with an underscore are reserved for internal use.
  // Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9_]*".
  string name = 1;

  // For describing inputs and outputs.
  message ArgDef {
    // Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
    string name = 1;

    // Human readable description.
    string description = 2;

    DataType type = 3;
    string type_attr = 4;    // if specified, attr must have type "type"
    string number_attr = 5;  // if specified, attr must have type "int"
    // If specified, attr must have type "list(type)", and none of
    // type, type_attr, and number_attr may be specified.
    string type_list_attr = 6;

    bool is_ref = 16;
  };
```

OpDef中最核心的数据成员是操作名称、输入、输出、参数。

对于其中的几个难理解的点，作出说明：

- `ArgDef`中的3-6四个字段，是为了描述·输入或输出的类型。当输入或输出是一个张量时，type或type_attr被设置为这个张量的数据类型，当输入或输出是一个由相同数据类型的张量构成的序列时，number_attr被设置为int对应的标识，当输入或输出是一个由张量构成的列表时，type_list_attr被设置为list（type）对应的标识；
- `AttrDef`中的`has_minimum`字段，表明这个属性是否有最小值，如果数据类型是int,那么minimum就是允许的最小值，如果数据类型是列表，那么minimum就是列表的最短长度，is_aggregate这个字段，表明当前的操作是否是可聚集的。一个可聚集的操作是，能接受任意数量相同类型和形状的输入，并且保持输出与每个输入的类型和形状相同，这个字段对于操作的优化非常重要，如果一个操作是可聚集的，并且其输入来自多个不同的设备，那么我们就可以把聚集优化成一个树形的操作，先在设备内部对输入做聚集，最后在操作所在的设备集中，这样可以提高效率。这种优化对于分布式的机器学习模型训练非常有帮助，Spark ML中的TreeAggregate就实现了这样的优化。
- is_stateful这个字段，表明当前的op是否带有状态的，什么操op会带有状态呢？比如Variable;

通过protoc工具用proto文件生成.h文件。命令为：

```shell
./protoc \ 
-I=/home/anan/tensorflow1.12/tensorflow-1.12.0/ \
--cpp_out=/home/anan/tensorflow1.12/tensorflow-1.12.0/tensorflow/core/framework/
/home/z00354782/tensorflow_1.12/tensorflow-
1.12.0/tensorflow/core/framework/op_def.proto
```

从中找到OpDef的定义：

```c++
class OpDef : public::google::protobuf::Message {
private:
    ::google::protobuf::RepeatedPtrField<::tensorflow::OpDef_ArgDef> input_arg_;
    ::google::protobuf::RepeatedPtrField<::tensorflow::OpDef_ArgDef> output_arg_;
    ::google::protobuf::RepeatedPtrField<::tensorflow::OpDef_ArgDef> attr_;
    ::google::protobuf::internal::ArenaStringPtr name_;
    ::google::protobuf::internal::ArenaStringPtr summary_;
    ::google::protobuf::internal::ArenaStringPtr description_;
    bool is_commutative_;
    bool is_aggregate_;
    bool is_stateful_;
    bool allows_uninitialized_input_;
}
```

为了方便进行OpDef的构建，TF还设计了`OpDefBuilder`类，它的私有数据成员如下：

```c++
// Builder class passed to the REGISTER_OP() macro.
class OpDefBuilder {
 public:
  // ...

 private:
  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  std::vector<string> control_outputs_;
  string doc_;
  std::vector<string> errors_;
};
```

可以看到，除了`errors_`字段外，其他内容几乎就是把OpDef的结构原封不动的搬了过来。

在`op_def_builder.h`中还有一个新的结构，`OpRegistrationData`，他的结构如下：

```c++
struct OpRegistrationData {
 public:
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def) : op_def(def) {}
  OpRegistrationData(const OpDef& def, const OpShapeInferenceFn& fn,
                     bool is_function = false)
      : op_def(def), shape_inference_fn(fn), is_function_op(is_function) {}

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
  bool is_function_op = false;
};
```

在这个结构中，除了屋面熟知的`OpDef`之外，还包含一个`OpShapeInferenceFn`结构，他的定义如下：

```c++
typedef std::function<Status(shape_inference::InferenceContext* c)>
    OpShapeInferenceFn;
```

这个结构的定义中，涉及到了我们后面要讲到的形状推断的内容，这里我们只需要知道，OpShapeInferenceFn是一个帮助操作根据输入形状对输出形状进行推断的函数即可。

### Op注册

上面的例子中使用`REGISTER_OP`宏进行Op注册，看一下这个宏的定义：

```c++
#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \
              name)>(name)
```

> 注：`__COUNTER__`宏表示自动计数，最终的定义是`register_op0`、`register_op1`、`register_op2`依次往后排。

```c++
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op0   = \
    ::tensorflow::register_op::OpDefBuilderWrapper<true>("Argmax") \
                            .Input("input: T")
                            .Input("dimension: Tidx")
                            .Output("output: output_type")
                            .Attr("T: numbertype")
                            .Attr("Tidx: {int32, int64} = DT_INT32")
                            .Attr("output_type: {int32, int64} = DT_INT64")
                            .SetShapeFn(ArgOpShape);
```

也就是说，生成一个`OpDefBuilderWrapper`对象，并链式调用它的`Input`、`Output`、`Attr`等方法。

`OpDefBuilderWrapper`的定义为：

```c++

// Template specialization that forwards all calls to the contained builder.
template <>
class OpDefBuilderWrapper<true> {
 public:
  explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}
  OpDefBuilderWrapper<true>& Attr(string spec) {
    builder_.Attr(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper<true>& Input(string spec) {
    builder_.Input(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper<true>& Output(string spec) {
    builder_.Output(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper<true>& SetIsCommutative() {
    builder_.SetIsCommutative();
    return *this;
  }
  OpDefBuilderWrapper<true>& SetIsAggregate() {
    builder_.SetIsAggregate();
    return *this;
  }
  OpDefBuilderWrapper<true>& SetIsStateful() {
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper<true>& SetAllowsUninitializedInput() {
    builder_.SetAllowsUninitializedInput();
    return *this;
  }
  OpDefBuilderWrapper<true>& Deprecated(int version, string explanation) {
    builder_.Deprecated(version, std::move(explanation));
    return *this;
  }
  OpDefBuilderWrapper<true>& Doc(string text) {
    builder_.Doc(std::move(text));
    return *this;
  }
  OpDefBuilderWrapper<true>& SetShapeFn(
      Status (*fn)(shape_inference::InferenceContext*)) {
    builder_.SetShapeFn(fn);
    return *this;
  }
  const ::tensorflow::OpDefBuilder& builder() const { return builder_; }

 private:
  mutable ::tensorflow::OpDefBuilder builder_;
};
```
通过链式调用，把Input、Output、Attr等描述保存到`OpDefBuiIder`的attrs_、inputs_、outputs_属性中。例如，Input的处理为：

```c++
OpDefBuilder& OpDefBuilder::Input(string spec) {
  inputs_.push_back(std::move(spec));
  return *this;
}
```

`OpDefBuilderWrapper`是`OpDefBuilder`的包装器，其成员包含一个`OpDefBuilder`的对象，它的API都是设置型的，且都返回对象本身，提供 链式的方式进行属性设置。值得注意的是，这个类名后面跟着一个true，它的含义等会再看。

最终把`OpDefBuilderWrapper`类型的对象用于构造`OpDefBuilderReceiver`。

`OpDefBuilderReceiver`定义为：

```c++
struct OpDefBuilderReceiver {
  // To call OpRegistry::Global()->Register(...), used by the
  // REGISTER_OP macro below.
  // Note: These are implicitly converting constructors.
  OpDefBuilderReceiver(
      const OpDefBuilderWrapper<true>& wrapper);  // NOLINT(runtime/explicit)
  constexpr OpDefBuilderReceiver(const OpDefBuilderWrapper<false>&) {
  }  // NOLINT(runtime/explicit)
};
}  // namespace register_op
```

`OpDefBuilderReceiver`的构造函数的实现为:

```c++
OpDefBuilderReceiver::OpDefBuilderReceiver(
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);
      });
}
```

相当于是`OpDefBuilderWrapper`构造时，以`OpDefBuilderWrapper`为参数，在构造函数中调用`OpRegistry::Global()->Register(...)`。

也就是说，`REGISTER_OP`绕了一圈，先用`OpDefBuilderWrapper`对操作进行封装，然后把它作为参数传递给`OpDefBuilderReceiver`的构造函数，而在这个构造函数中，完成了对算子的注册。

真正的注册过程就是`OpRegistry`的`Register`方法中完成的，下面具体看一下注册类的实现。

### 注册类

为了方便对操作进行统一管理，TF提出了OP注册器的概念。这个OP注册器的作用，是为各种OP提供一个统一的管理接囗。

操作注册类的继承结构如下：

![image](https://user-images.githubusercontent.com/11350907/58377221-e46e1480-7fad-11e9-86fc-0d37c91104ca.png)

其中，`OpRegistryInterface`是一个接口类，它提供了注册类最基础的查找功能：

```c++
// Users that want to look up an OpDef by type name should take an
// OpRegistryInterface.  Functions accepting a
// (const) OpRegistryInterface* may call LookUp() from multiple threads.
class OpRegistryInterface {
 public:
  virtual ~OpRegistryInterface();

  // Returns an error status and sets *op_reg_data to nullptr if no OpDef is
  // registered under that name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  virtual Status LookUp(const string& op_type_name,
                        const OpRegistrationData** op_reg_data) const = 0;

  // Shorthand for calling LookUp to get the OpDef.
  Status LookUpOpDef(const string& op_type_name, const OpDef** op_def) const;
};
```

`OpRegistry`类继承了`OpRegistryInterface`类。

```c++
// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering ops via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register(
//     [](OpRegistrationData* op_reg_data)->Status {
//       // Populate *op_reg_data here.
//       return Status::OK();
//   });
class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;

  OpRegistry();
  ~OpRegistry() override;

  void Register(const OpRegistrationDataFactory& op_data_factory);

  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Fills *ops with all registered OpDefs (except those with names
  // starting with '_' if include_internal == false) sorted in
  // ascending alphabetical order.
  void Export(bool include_internal, OpList* ops) const;

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  string DebugString(bool include_internal) const;

  // A singleton available at startup.
  static OpRegistry* Global();

  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);

  // Get all `OpRegistrationData`s.
  void GetOpRegistrationData(std::vector<OpRegistrationData>* op_data);

  // Watcher, a function object.
  // The watcher, if set by SetWatcher(), is called every time an op is
  // registered via the Register function. The watcher is passed the Status
  // obtained from building and adding the OpDef to the registry, and the OpDef
  // itself if it was successfully built. A watcher returns a Status which is in
  // turn returned as the final registration status.
  typedef std::function<Status(const Status&, const OpDef&)> Watcher;

  // An OpRegistry object has only one watcher. This interface is not thread
  // safe, as different clients are free to set the watcher any time.
  // Clients are expected to atomically perform the following sequence of
  // operations :
  // SetWatcher(a_watcher);
  // Register some ops;
  // op_registry->ProcessRegistrations();
  // SetWatcher(nullptr);
  // Returns a non-OK status if a non-null watcher is over-written by another
  // non-null watcher.
  Status SetWatcher(const Watcher& watcher);

  // Process the current list of deferred registrations. Note that calls to
  // Export, LookUp and DebugString would also implicitly process the deferred
  // registrations. Returns the status of the first failed op registration or
  // Status::OK() otherwise.
  Status ProcessRegistrations() const;

  // Defer the registrations until a later call to a function that processes
  // deferred registrations are made. Normally, registrations that happen after
  // calls to Export, LookUp, ProcessRegistrations and DebugString are processed
  // immediately. Call this to defer future registrations.
  void DeferRegistrations();

  // Clear the registrations that have been deferred.
  void ClearDeferredRegistrations();

 private:
  // Ensures that all the functions in deferred_ get called, their OpDef's
  // registered, and returns with deferred_ empty.  Returns true the first
  // time it is called. Prints a fatal log if any op registration fails.
  bool MustCallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Calls the functions in deferred_ and registers their OpDef's
  // It returns the Status of the first failed op registration or Status::OK()
  // otherwise.
  Status CallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add 'def' to the registry with additional data 'data'. On failure, or if
  // there is already an OpDef with that name registered, returns a non-okay
  // status.
  Status RegisterAlreadyLocked(const OpRegistrationDataFactory& op_data_factory)
      const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Status LookUpSlow(const string& op_type_name,
                    const OpRegistrationData** op_reg_data) const;

  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ GUARDED_BY(mu_);
  // Values are owned.
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      GUARDED_BY(mu_);
  mutable bool initialized_ GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ GUARDED_BY(mu_);
};
```

`OpRegistry`类是单例模式，通过`Global`获取单例对象，并且是线程安全的。

注册函数`Register`的定义为：

```c++
void OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory) {
  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
}
```

其中，`OpRegistrationDataFactory`是一个function类型：

```c++
typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;
```

也就是说，`Register`注册时传入的是一个函数，最终在`Register`中完成对函数的调用。

从代码看，只有`RegisterAlreadyLocked(op_data_factory)`中可能产生对`op_data_factory`的调用，所以可以从这儿入手看注册过程。姑且不论`initialized_`字段的值。

```c++
// Add 'def' to the registry with additional data 'data'. On failure, or if
// there is already an OpDef with that name registered, returns a non-okay
// status.
Status OpRegistry::RegisterAlreadyLocked(
    const OpRegistrationDataFactory& op_data_factory) const {
  std::unique_ptr<OpRegistrationData> op_reg_data(new OpRegistrationData);
  Status s = op_data_factory(op_reg_data.get());
  if (s.ok()) {
    s = ValidateOpDef(op_reg_data->op_def);
    if (s.ok() &&
        !gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
                                 op_reg_data.get())) {
      s = errors::AlreadyExists("Op with name ", op_reg_data->op_def.name());
    }
  }
  Status watcher_status = s;
  if (watcher_) {
    watcher_status = watcher_(s, op_reg_data->op_def);
  }
  if (s.ok()) {
    op_reg_data.release();
  } else {
    op_reg_data.reset();
  }
  return watcher_status;
}
```

函数的注释写的很清楚了，新增一个def到register中。失败或者算子name已经被注册，返回非okey结果。

这个函数中构造了一个`OpRegistrationData`对象，并最终对`op_data_factory`进行了调用。

`OpRegistrationData`的定义如下，其中包含了一个`OpDef`的变量。

```c++
struct OpRegistrationData {
 public:
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def) : op_def(def) {}
  OpRegistrationData(const OpDef& def, const OpShapeInferenceFn& fn,
                     bool is_function = false)
      : op_def(def), shape_inference_fn(fn), is_function_op(is_function) {}

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
  bool is_function_op = false;
};
```

对`op_data_factory`的调用构造了一个`OpRegistrationData`空对象，最终进入`wrapper.builder().Finalize(op_reg_data)`中进行处理。

`wrapper.builder()`返回的是`OpDefBuilder`对象。函数`Finalize`的实现为：

```c++
Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
  std::vector<string> errors = errors_;
  *op_reg_data = op_reg_data_;

  OpDef* op_def = &op_reg_data->op_def;
  for (StringPiece attr : attrs_) {
    FinalizeAttr(attr, op_def, &errors);
  }
  for (StringPiece input : inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (StringPiece output : outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
  for (StringPiece control_output : control_outputs_) {
    FinalizeControlOutput(control_output, op_def, &errors);
  }
  FinalizeDoc(doc_, op_def, &errors);

  if (errors.empty()) return Status::OK();
  return errors::InvalidArgument(str_util::Join(errors, "\n"));
}
```

这里把最开始`wrapper`中保存的`inputs_`、`outputs_`、`attrs_`等信息依次取出，用于构建`OpDef`对象。

得到的`OpDef`对象首先经过`ValidateOpDef(op_reg_data->op_def);`进行校验，然后插入到`Register`的`registry_`中。

```c++
gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
                                 op_reg_data.get()))
```

到这里就完成了一个算子的注册过程。

下面这个代码值得注意：

```c++
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
```

只有在`initialized_`是true时，才进行注册，否则把`op_data_factory`放到`deferred_`这个vector中。

注意到`Register`类有如下两个方法：

```c++
// Ensures that all the functions in deferred_ get called, their OpDef's
// registered, and returns with deferred_ empty.  Returns true the first
// time it is called. Prints a fatal log if any op registration fails.
bool OpRegistry::MustCallDeferred() const {
  if (initialized_) return false;
  initialized_ = true;
  for (size_t i = 0; i < deferred_.size(); ++i) {
    TF_QCHECK_OK(RegisterAlreadyLocked(deferred_[i]));
  }
  deferred_.clear();
  return true;
}

// Calls the functions in deferred_ and registers their OpDef's
// It returns the Status of the first failed op registration or Status::OK()
// otherwise.
Status OpRegistry::CallDeferred() const {
  if (initialized_) return Status::OK();
  initialized_ = true;
  for (size_t i = 0; i < deferred_.size(); ++i) {
    Status s = RegisterAlreadyLocked(deferred_[i]);
    if (!s.ok()) {
      return s;
    }
  }
  deferred_.clear();
  return Status::OK();
}
```

可以看出，在特定的调用中，把`deferred_`中保存的算子注册函数全部取出，执行`RegisterAlreadyLocked`真正的执行算子注册过程。

这里有几点值得关注：

- 注册函数`Register`的输入是一个函数引用，这个函数接收一个`OpRegistrationData`指针作为输入；
- `Watcher`是一个监视器，当每次注册一个算子的时候，在注册步骤的最后都要调用一下这个监视器，它可方便对注册的操作进行监控，所有的算子注册动作都逃不过它的眼，可以根据需求定制特殊Watcher；
- registry_`是已注册的算子真正存放的位置，它的结构很简单，是一个key为算子名、value为算子数据的map；
- `initialized_`和`deferred_`是与注册模式相关的两个数据成员，注册器在注册操作时，分为两种模式：
  - **即时注册模式**和**懒惰注册模式**
  - 注册模式通过`initialized_`字段区分，true为即时注册模式，false为懒惰注册模式；
  - 在懒惰注册模式中，被注册的算子先 被保存在`deferred_`向量中，在特定的函数调用时再将`deferred_`中的算子注册到`registryy_`，而即时注册模式下，待注册的算子不用经过`deferred_`，直接注册到`registry_`。
  -懒惰注册模式的使用场景是，部分算子组合的注册是原子的，即要么全部注册，要么全部不注册，因为这些算子之间可能会有相互依赖关系。
  - 构造函数将`initialized_`设置为false,进入懒惰注册模式，随后一旦调用了`MustCallDeferred`或者`CallDeferred`中的任意一个，都会将`initialized_`设置为true,进入即时注册模式。想要重新返回懒惰注册模式也很简单，只需要调用`DeferRegistrations`即可。

### 参考

[https://www.cnblogs.com/jicanghai/p/9539513.html](https://www.cnblogs.com/jicanghai/p/9539513.html)

> 注：文中代码基于`tensorflow1.12.0`版本。