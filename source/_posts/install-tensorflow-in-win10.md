---
title: install_tensorflow_in_win10
date: 2018-08-26 17:22:41
tags: [tools,install]
---

确定如何安装 TensorFlow
您必须选择安装 TensorFlow 的方式。目前可支持如下几种方式：

- “原生”pip
- Anaconda

pip方式安装Tensorflow需要额外安装其依赖的其他python包（而且需要python 3.6及以下的版本，我安装的是python 3.7，不想重新安装一次折腾了），所以使用Anacoda来安装。

### 使用 Anaconda 进行安装

按照以下步骤在 Anaconda 环境中安装 TensorFlow：

- 1.按照 Anaconda 下载网站上的说明下载并安装 Anaconda。
- 2.通过调用以下命令创建名为 tensorflow 的 conda 环境：
    ```
    C:> conda create -n tensorflow pip python=3.5 
    ```
- 3.通过发出以下命令激活 conda 环境：
    ```
    C:> activate tensorflow
    (tensorflow)C:>  # Your prompt should change 
    ```
- 4.发出相应命令以在 conda 环境中安装 TensorFlow。要安装仅支持 CPU 的 TensorFlow 版本，请输入以下命令：
    ```
    (tensorflow)C:> pip install --ignore-installed --upgrade tensorflow 
    ```
- 5.要安装 GPU 版本的 TensorFlow，请输入以下命令（在同一行）：
    ```
    (tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu 
    ```

#### 验证安装

- 1.启动终端，激活 Anaconda 环境。
- 2.从 shell 中调用 Python，如下所示：
```
$ python
```
在 Python 交互式 shell 中输入以下几行简短的程序代码：
```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```
如果系统输出以下内容，说明您可以开始编写 TensorFlow 程序了：
```
Hello, TensorFlow!
```
如果系统输出一条错误消息而不是问候语，请参阅常见的安装问题。

### 通过jupyter使用tensorflow

启动终端，并激活 Anaconda 环境后，再启动jupyter notebook。在jupyter中就可以使用tensorflow了。

```
(base) e:\jupyter>activate tensorflow

(tensorflow) e:\jupyter>
(tensorflow) e:\jupyter>jupyter notebook --config=./jupyter_notebook_config.json
[I 17:17:42.556 NotebookApp] Serving notebooks from local directory: e:\jupyter
[I 17:17:42.556 NotebookApp] The Jupyter Notebook is running at:
[I 17:17:42.557 NotebookApp] http://localhost:8888/
```

在页面上新建python3文件，通过如代码查看tensorflow是否可用：

```python
import tensorflow as tf

print(tf.__version__)
```

运行cell，正常情况下会打印出tf的版本号。