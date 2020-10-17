---
title: jupyter_in_windows10
date: 2018-08-19 10:16:30
tags: [tools,install]
---

本文介绍在windows上安装jupyter，使用notebook。

## windows上使用jupyter

### 安装jupyter

- 1.安装python
	- 下载安装包安装
- 2.安装pip
	-升级pip3:`pip3 install --upgrade pip`

- 3.安装jupyter
	- `pip3 install jupyter`
	- 可修改文件`C:\Users\anan\pip\pip.ini`修改pip源：
		```
		[global]
 		index-url = https://pypi.tuna.tsinghua.edu.cn/simple
        ```
    - 如果超时，可加大超时时间：
    	`pip3 --default-timeout=100 install jupyter`
- 4. 查看是否安装成功
        ```
        PS C:\Users\anan> jupyter notebook  --version
        5.6.0
        PS C:\Users\anan>
        ```

### 启动jupyter

- 1.生成配置文件
    - `jupyter notebook --generate-config`
    - 生成的文件在`C:\Users\anan\.jupyter\jupyter_notebook_config.json`
- 2.设置密码
	- `jupyter notebook password`
	- 输入密码
- 3.拷贝配置文件到当前目录
    - cp C:\Users\anan\.jupyter\jupyter_notebook_config.json .
- 4.启动jupyter
	- `jupyter notebook --config=./jupyter_notebook_config.json `

> 注：这里是指定配置文件（设置里默认密码）的启动方式。也可以不指定配置文件启动，即`jupyter notebook`可以启动，启动时会自动生成一个token，作为登陆密码，这样有个问题是每次启动生成的token会变。所以这种指定密码的方式会方便很多。

### jupyter的使用