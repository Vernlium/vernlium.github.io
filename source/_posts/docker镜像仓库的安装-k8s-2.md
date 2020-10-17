---
title: docker镜像仓库的安装-k8s_2
date: 2017-08-14 21:13:24
tags: docker
categories: k8s
---

镜像仓库是存储镜像和分发镜像的系统。docker的使用离不开镜像仓库。本文介绍如何安装docker镜像仓库。

## 简介

镜像仓库是存储镜像和分发镜像的系统。docker的使用离不开镜像仓库。当前最大的镜像仓库是`dockerhub`，这里面包含了很多镜像，比如各大常用软件的镜像，如nginx/ubuntu/mysql/redis等，还有很多个人开发者制作的镜像，或者是完全新的镜像或者是在现有镜像基础上开发的特殊需求的镜像等。

有时候开发者可能需要进行测试或者涉及安全问题不想把镜像流传到外网，这时就需要自己搭建私有镜像仓库，只能在自己到内网中访问。

下面就来讲一下私有镜像仓库如何搭建。

## 安装

本安装指导使用到操作系统是Ubuntu 14.04。

下面我们安装一个域名为`vernlium.com`的镜像仓库。

### 环境检查

我们到镜像仓库是以docker容器到方式启动到，所以需要安装docker。

```shell
if [ `which docker | wc -l` -ne 0 ]; then
  apt-get install docker-engine
fi
```

另外，安装过程中需要用到except软件，所以安装前需要检查此软件是否已安装，没安装到进行安装。

```
#检查是否安装except
if [ `which expect | wc -l` -ne 0 ]; then
    #安装except
    apt-get install except
fi
```

### 证书生成

```
mkdir certs
openssl req -newkey rsa:2048 -nodes -sha256 -keyout certs/domain.key -x509 -days 365 -out certs/domain.crt
```

在shel脚本中可以使用expect命令，代码如下：

```
expect -c "
set timeout 300
spawn openssl req -newkey rsa:2048 -nodes -sha256 -keyout certs/domain.key -x509 -days 365 -out certs/domain.crt
expect {
    \"Country Name *\" {send \"CN\r\";exp_coutinue}
    \"State or Province Name*\" {send \"JS\r\";exp_coutinue}
    \"Locality Name *\" {send \"NJ\r\";exp_coutinue}
    \"Organization Name *\" {send \"HW\r\";exp_coutinue}
    \"Organiztional Unit Name *\" {send \"DW\r\";exp_coutinue}
    \"Common Name *\" {send \"vernlium.com:5005\r\";exp_coutinue}
    \"Email Address *\" {send \"zhanganan0425@163.com\r\"}
}
expect eof;
"
```

执行完上述命令后，会在执行脚本的目录下certs下生成两个文件：domain.crt和domain.key，这个是证书文件和私钥文件。

这个证书在K8S安装时需要用到。

### 生成用户名和密码

```
docker run --entrypoint htpasswd registry -Bbn \
anan qwe123 | tee auth/htpasswd
```

其中：

- `registry`:是registry镜像
- anan : 用户名，可随意填写
- qwe123 : 密码，可随意填写

### 镜像仓库容器启动

使用如下命令启动

```
docker run -d -ti \
  -p 5005:5000 \
  --restart=always \
  --name registry \
  --log-opt max-size=50m \
  -v /etc/localtime:/etc/localtime:ro \
  -v `pwd`/auth:/auth \
  -v /var/lib/docker/registry:/var/lib/registry \
  -e "REGISTRY_AUTH=htpasswd" \
  -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
  -e REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd \
  -v `pwd`/certs:/certs \
  -e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
  -e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
  registry
```

命令中挂载来三个文件夹，其作用为：

- auth: 认证信息
- certs: 证书信息
- /var/lib/docker/registry: 存放镜像文件到目录，如果镜像比较多且单个镜像比较大到话，这个文件夹到磁盘空间需要比较大。

### 连接到镜像仓库

镜像仓库搭建完成后，如何让一个机器连接到此镜像仓库从此镜像仓库拉取镜像和向此镜像仓库推送镜像呢？

- 1.证书下载
    - 新建目录：`/etc/docker/certs.d/vernlium.com\:5005/`
    - 将镜像仓库安装过程中生成的docker.crt拷贝到此文件夹下；
- 2.域名解析
    - /etc/hosts文件中添加镜像仓库域名对应的ip: 100.120.45.26 vernlium.com  
- 3.生成密码文件
    - 新建文件`/root/.docker/config.json`，并文件中添加如下内容：
    ```
    {
        “auth”:{
                "vernlium.com:5005":{
                       "auth":"dGVzdDphbmFu"
                ｝
        }
    }
    ```
    - 其中：那一串字符是`userName:passwd`的base64加密后密文，可使用浏览器进行加密。
    - Chrome在任意页面，按F12，切到Console页签，使用如下方式加密：
    ![image](https://user-images.githubusercontent.com/11350907/29341217-85fdd234-8255-11e7-9a2b-1c19b39867f9.png)

上述操作完成后，执行'docker login vernlium.com:5000'，按照提示输入镜像仓库的用户名（一般会给出默认值，就是config.json文件中配的）和密码，如果配置都正确的话，则显示：`Login Successed`

如果提示509错误，则可能是证书配的不对。

## 镜像仓库的使用

### 获取镜像列表

#### 命令方式

命令方式只能在镜像仓库到机器上执行才行，其原理就是遍历存放镜像文件到目录，获取所有到镜像。

```
find /var/lib/docker/registry/ -print | grep 'v2/repositories' | grep 'current' | grep -v 'link' | sed -e 's/\/_manifests\/tags\//:/' | sed -e 's/\/current//' | sed -e 's/^.*repositories/vernlium.con:5005/' | sort
```

#### api方式

镜像仓库提供来一些api接口可以来查询镜像列表。可以通过浏览器或者其他api工具如postman等进行访问。

##### 获取镜像列表

```
GET https://registry_ip:5005/v2/_catalog
```

返回结果为：

```
{
  "repositories": [
    "anan/ss-server",
    "anan/dev",
    "anan/jenkins",
    "anan/live-api-runtime",
    "anan/pingstart-runtime",
    "anan/warehouse-runtime",
    "ubuntu"
  ]
}
```

##### 获取单个镜像到tag列表

```
GET https://registry_ip:5005/v2/anan/live-api-runtime/tags/list
```

返回结果为：

```
{
  "name": "anan/live-api-runtime",
  "tags": [
    "v1.0",
    "v1.1",
    "v2.0"
  ]
}
```

##### 删除镜像

```
DELETE https://registry_ip:5005/v2/<name>/manifests/<reference>
```

### pull和push镜像

私有镜像仓库建立好之后是没有镜像的，我们要往里面push镜像才行。我们可以自己生成镜像，也可以从其他镜像仓库pull镜像后通过`docker tag`命令把镜像重新打tag，然后push到私有镜像仓库。

下面以ubuntu镜像为例，我们从dockerhub中下载镜像到本地，然后push到私有仓库。

```
$ docker pull ubuntu
$ docker tag ubuntu vernlium.com:5005/ubuntu
$ docker push vernlium.com:5005/ubuntu
```

其他节点就可以从镜像仓库下载此镜像了

```
docker pull vernlium.com:5005/ubuntu
```

### 镜像仓库的清理

我们都知道docker镜像是以层的形式组织的，而镜像仓库提供的删除镜像的能力只是删除镜像的tag，而非真正的删除镜像的层文件。那么镜像仓库使用一段时间后，就会产生很多废弃的层文件，如何清理这些文件呢？镜像仓库提供了GC能力，就是清理无用的镜像层文件。

执行如下命令进行GC操作：

```
docker exec -it registry /bin/registry garbage-collect [--dry-run] /etc/docker/registry/config.yml
```

其中：
`--dry-run` 参数可选，加上此参数表示：运行GC列出可GC的文件列表，并不真正的进行GC，而不加此参数则表示直接进行GC操作。

### 其他

#### 节点连到多个私有镜像仓库

有时候有的节点需要连接多个私有镜像仓库，应该如何做呢？

- 1.证书下载
    - 新建目录：`/etc/docker/certs.d/xxx.com\:5005/`
    - 将镜像仓库安装过程中生成的docker.crt拷贝到此文件夹下；
- 2.域名解析
    - /etc/hosts文件中添加镜像仓库域名对应的ip: 100.100.22.22 xxxx.com  
- 3.密码文件中添加信息
    - 在文件`/root/.docker/config.json`中添加如下内容：
    ```
    {
        "auths":{
                 "vernlium.com:5005":{
                       "auth":"dGVzdDphbmFu"
                },
                "xxx.com:5005":{
                       "auth":"xxx"
                }
        }
    }
    ```

可以按照这种方式添加多个，且无需重启docker，直接可以使用。


### 参考

https://docs.docker.com/registry/spec/api/#overview
http://www.csdn.net/article/2015-11-24/2826315

