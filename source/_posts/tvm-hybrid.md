---
title: tvm_hybrid
date: 2019-08-17 17:58:23
tags: TVM hybrid
---

本文介绍TVM的hybrid编程方式。

## Hybrid

### Hybrid简介

### Hybrid示例

### Loops

不同range的区别:

In HalideIR, loops have in total 4 types: serial, unrolled, parallel, and vectorized.

Here we use range aka serial, unroll, parallel, and vectorize, these 4 keywords to annotate the corresponding types of for loops. The the usage is roughly the same as Python standard range.

Besides all the loop types supported in Halide, const_range is supported for some specific conditions.

#### serial

```python
@tvm.hybrid.script
def foo(a, b): # b is a tvm.container.Array
    c = output_tensor(a.shape, a.dtype)
    for i in serial(len(a)):
        c[i] = a[i] + b[i]
    return c
```

报错：

```
ValueError: Function call id not in intrinsics' list
```

#### unrool

```python
@tvm.hybrid.script
def foo(a, b): # b is a tvm.container.Array
    c = output_tensor(a.shape, a.dtype)
    for i in unroll(len(a)):
        c[i] = a[i] + b[i]
    return c
```

输出IR为：

‘’‘
produce foo {
  // attr [0] extern_scope = 0
  foo[0] = (a[0] + b[0])
  foo[1] = (a[1] + b[1])
  foo[2] = (a[2] + b[2])
  foo[3] = (a[3] + b[3])
  foo[4] = (a[4] + b[4])
}
’‘’

#### parallel

```python
@tvm.hybrid.script
def foo(a, b): # b is a tvm.container.Array
    c = output_tensor(a.shape, a.dtype)
    for i in parallel(len(a)):
        c[i] = a[i] + b[i]
    return c
```

输出IR为：

‘’‘
produce foo {
  // attr [0] extern_scope = 0
  parallel (i, 0, 5) {
    foo[i] = (a[i] + b[i])
  }
}
’‘’

#### vectorize

```python
@tvm.hybrid.script
def foo(a, b): # b is a tvm.container.Array
    c = output_tensor(a.shape, a.dtype)
    for i in vectorize(len(a)):
        c[i] = a[i] + b[i]
    return c
```

输出IR为：

‘’‘
produce foo {
  // attr [0] extern_scope = 0
  foo[ramp(0, 1, 5)] = (a[ramp(0, 1, 5)] + b[ramp(0, 1, 5)])
}
’‘’

#### const_range

```python
@tvm.hybrid.script
def foo(a, b): # b is a tvm.container.Array
    c = output_tensor(a.shape, a.dtype)
    for i in const_range(len(a)):
        c[i] = a[i] + b[i]
    return c
```

输出IR为：

‘‘’
produce foo {
  // attr [0] extern_scope = 0
  foo[0] = (a[0] + b[0])
  foo[1] = (a[1] + b[1])
  foo[2] = (a[2] + b[2])
  foo[3] = (a[3] + b[3])
  foo[4] = (a[4] + b[4])
}
‘’‘

可以看出，不同的range关键字，生成的IR不一样。

### 变量

hybrid中，所有的变量，在IR中，都是一个大小为1的数组。

‘’‘python
import tvm

@tvm.hybrid.script
def foo(a): # b is a tvm.container.Array
    c = output_tensor((a.shape[0],), a.dtype)
    for i in range(a.shape[0]):
        s = 0.0
        # declaration, this s will be a 1-array in lowered IR
        for j in range(a.shape[1]):
            s += a[i, j]
        c[i] = s
    
    return c

a = tvm.placeholder((5, 5), name='a', dtype = "float32")
c = foo(a) # return the output tensor(s) of the operator

sch = tvm.create_schedule(c.op)
print(tvm.lower(sch, [a, c], simple_mode=True))
```

打印的IR如下，可以看到变量`s`是一个大小为1的数组：

‘’‘
// attr [s] storage_scope = "global"
allocate s[float32 * 1]
produce foo {
  // attr [0] extern_scope = 0
  for (i, 0, 5) {
    s[0] = 0f
    for (j, 0, 5) {
      s[0] = (s[0] + a[((i*5) + j)])
    }
    foo[i] = s[0]
  }
}
’‘’

另外，在hybrid中，变量只能在其声明的作用域内使用，不像正常的python中，超过其声明的作用域范围还能使用。

比如，下面的写法是非法的：

‘’‘python
for i in range(5):
    s = 0 # declaration, this s will be a 1-array in lowered IR
    for j in range(5):
       s += a[i, j] # do something with sum
    b[i] = sum # you can still use sum in this level
a[0] = s # you CANNOT use s here, even though it is allowed in conventional Python
’‘’

当前，在hybrid中，只能使用python基本的数据类型变量。

### 属性

当前，只有tensor的`shape`和`dtype`属性是支持访问的，其他的都不支持。`shape`属性都是一个元祖，按照一个数组来访问它。而且，只能使用常量index来访问shape。

‘’‘python
x = a.shape[2] # OK!
for i in range(3):
   for j in a.shape[i]: # BAD! i is not a constant!
       # do something
’‘’

### 条件声明和表达式

‘’‘python
if condition1 and condition2 and condition3:
    # do something
else:
    # do something else
# Select
a = b if condition else c
’‘’

注意，无法使用`True`或`False`.

### Math Intrinsics

可以使用的数学函数： `log, exp, sigmoid, tanh, power, and popcount`.
不需要导入，可直接使用。

### Array Allocation

正在支持中，目前还不支持。

### Keywords

For keywords: `serial, range, unroll, parallel, vectorize, bind, const_expr`

Math keywords: `log, exp, sigmoid, tanh, power, popcount`
