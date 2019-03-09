---
title: tensorflow_常见错误
date: 2018-08-28 07:37:55
tags:
---

### 'NoneType' object is not iterable

产生原因：

当sess.run的第一个参数列表和返回值列表不匹配时，会出现这个错误。比如：

```python
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        eval, cost = sess.run(evaluation_step, feed_dict={xxx})
```

修改之后就可以了：

```python
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        eval, cost = sess.run([evaluation_step, cost], feed_dict={xxx})
```