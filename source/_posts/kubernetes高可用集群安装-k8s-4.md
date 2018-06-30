---
title: kubernetes高可用集群安装-k8s-4
date: 2017-08-30 08:17:21
tags: [docker,k8s]
categories: k8s
---

这篇博客来讲解下k8s高可用集群的安装。

上篇博客中，已经介绍了k8s集群的基础安装，对k8s的组件和使用用了大致的了解。

一般在生产环境中，考虑到可靠性和可用性，避免单节点故障时，环境不可用，需要需要通过安装k8s master集群来提供系统可用性。本篇博客来讨论如何安装k8s master集群。

### 环境准备

准备7台虚拟机，其用途分别如下：

| IP | 用途 | 配置 | 
| -- | ---- | ---- |
| 192.168.0.1 | nginx | 4C16G |
| 192.168.0.2 | master | 4C16G |
| 192.168.0.3 | master | 4C16G |
| 192.168.0.4 | master | 4C16G |
| 192.168.0.5 | node | 4C16G |
| 192.168.0.6 | node | 4C16G |
| 192.168.0.7 | node | 4C16G |

### 组网图

![kubernetes](https://user-images.githubusercontent.com/11350907/29849869-41e46f06-8d5c-11e7-9c05-9c8e47f43c21.jpg)

一般集群为了保障可用性，避免单点故障，至少需要3个节点。

### master集群安装

从上面的组网图中可以看出， 需要使用nginx对三个master进行负载均衡。所以需要启动一个nginx。为了方便安装和使用，我们使用容器的方式启动nginx。

#### nginx

首先需要下载nginx的镜像，从可以连接外网的机器上`docker pull nginx`下载。

然后修改nginx的配置文件，创建文件`pwd/conf.d/mynginx.conf`，修改其内容为：

```
upstream k8smaster {
    server 192.168.0.2:8080;
    server 192.168.0.3:8080;
    server 192.168.0.4:8080;    
}

server {
    listen  8888;
    server_name localhost;

    location / {
        proxy_pass http://k8smaster;
    }
}
```

其中，三个server的地址对应的是三个master的地址。

然后使用docker run的方式启动nginx，需要将这个配置文件挂载上去，启动命令为：

```
docker run -d --name k8snginx -p 8543:8888 \
-v `pwd`/conf.d:/etc/nginx/conf.d nginx
```

#### 3个master的安装

上的组网图中，etcd是一个独立的集群，master也是一个集群，kube-apiserver和etcd集群连接，kube-apiserver需要配置etcd的地址（集群中的3个节点的地址都要配）。每个master上启动四个组件，etcd组成一个集群，k8s也组成一个集群。

这4个组件的启动依然使用upstart脚本，这些在基本安装中已经讲过，不再赘述。

主要讲一下各组件的启动参数。

#### etcd

etcd0:

```
ETCD_OPTS="--name etcd0 --data-dir=/etcd0.etcd \
--initial-advertise-peer-urls http://192.168.0.2:2380 \
--listen-peer-urls http://0.0.0.0:2380 \
--listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
--advertise-client-urls http://192.168.0.2:2379,http://192.168.0.2:4001 \
--initial-cluster-token k8setcd \
--initial-cluster etcd0=http://192.168.0.2:2380,etcd1=http://192.168.0.3:2380,etcd2=http://192.168.0.4:2380 \
--initial-cluster-state new"
```

etcd1:

```
ETCD_OPTS="--name etcd1 --data-dir=/etcd1.etcd \
--initial-advertise-peer-urls http://192.168.0.3:2380 \
--listen-peer-urls http://0.0.0.0:2380 \
--listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
--advertise-client-urls http://192.168.0.3:2379,http://192.168.0.3:4001 \
--initial-cluster-token k8setcd \
--initial-cluster etcd0=http://192.168.0.2:2380,etcd1=http://192.168.0.3:2380,etcd2=http://192.168.0.4:2380 \
--initial-cluster-state new"
```

etcd2:

```
ETCD_OPTS="--name etcd2 --data-dir=/etcd2.etcd \
--initial-advertise-peer-urls http://192.168.0.4:2380 \
--listen-peer-urls http://0.0.0.0:2380 \
--listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
--advertise-client-urls http://192.168.0.4:2379,http://192.168.0.4:4001 \
--initial-cluster-token k8setcd \
--initial-cluster etcd0=http://192.168.0.2:2380,etcd1=http://192.168.0.3:2380,etcd2=http://192.168.0.4:2380 \
--initial-cluster-state new"
```

etcd集群需要leader选举，所以初始时，其中任意一个节点都有知道其他节点的地址，其中initial-cluster参数配置的就是集群中的所有etcd的地址。

其中，几个参数的作用为：

- name : etcd的名称
- data-dir : 存放数据的路径
- initial-advertise-peer-urls : 广播给集群内部其他成员访问的url
- listen-peer-urls : 集群内部通信使用的url
- listen-client-urls : 外部客户端使用的url
- advertise-client-urls : 广播给外部客户端使用的url
- initial-cluster-token : etcd集群名称，用于区分不同集群
- initial-cluster : 初始集群成员列表，集群中所有的节点都有列出
- initial-cluster-state : 初始集群状态，new为新建集群

集群中三个etcd节点都启动成功后，可以检测集群状态是否ok。

执行 `etcdctl cluster-health`可以看到集群整体的健康度：


执行`etcdctl member list`可以查看集群的leader以及所有成员：


### k8s master集群

etcd集群启动后，再启动k8s的3个组件。3个master上的k8s的3个组件的启动参数是相同的。

master集群的启动参数和单master的略有不同，不同之处在于：

- apiserver的etcd_server配置为etcd集群的3个节点的地址，用逗号隔开；
- controller-manager和schedule的master参数配置为nginx的地址；
- controller-manager和schedule的启动参数多了一个`--leader-elect=true`，因为是一个集群，需要leader选举；

kube-apiserver的启动参数：

```
KUBE_APISERVER_OPTS="--insecure-bind-address=0.0.0.0 \
--insecure-port=8080 \
--cors-allowed-origins=.* \
--etcd_servers=http://192.168.0.2:2379,http://192.168.0.3:2379,http://192.168.0.4:2379 \
--service-cluster-ip=172.17.0.0/16 \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/apiserver"
```

kube-controller-manager的启动参数
```
KUBE_CONTROLLER_MANAGER_OPTS="--master=192.168.0.1:8543 \
--enable-hostpath-provisioner=false \
--pod-eviction-timeout=60m0s
--v=1 --logtostderr=false --log_dir=/var/k8slogs/controller-manager \
--leader-elect=true"
```

kube-scheduler的启动参数

```
KUBE_SCHEDULER_OPTS="--master=192.168.0.1:8543  \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/scheduler \
--leader-elect=true"
```

启动命令依然是：

```
start etcd
start kube-apiserver
start kube-controller-manager
start kube-scheduler
```

3个master都启动后，通过nginx地址就可以访问k8s集群了。


### node的安装

#### kubelet

kubelet的启动参数

```
KUBELET_OPTS="--address=0.0.0.0 \
--port=10250 \
--api_servers=http://192.168.0.1:8543 \
--pod-infra-container-image=docker.io/kubernetes/pause \
--node-labels=node=anannode \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/kubelet"
```

#### kube-proxy

kube-proxy的启动参数

```
KUBE_PROXY_OPTS="--master=http://192.168.0.1:8543 \
--proxy-mode=iptables \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/kube-proxy"
```

和单master的node安装基本一致，修改master参数为nginx的地址即可。

### 安装脚本化

上述的流程已经比较详细了，但是每个节点上都要手工操作一次会略显复杂，尤其是node节点，很多。所以我们需要将这些流程用脚本来执行，可以提升我们的效率。

首先，需要将用到的k8s组件放到一个压缩文件中；
- 将kube-apiserver、kube-controller-manager、kube-scheduler、kubectl、etcd放到k8s-master-pkg.tar.gz中
- 将kubelet、kube-proxy放到k8s-node-pkg.tar.gz中
- 同时，将镜像仓库的证书、配置文件等亚务放到k8s-node-pkg.tar.gz中

将master的安装过程和node的安装过程分别编写一个sh脚本，命名为：k8s-master-install.sh和k8s-node-install.sh

将安装master的虚机的ip放到文件master_ip中，安装node的虚机的ip列表放到node_ip中。

然后在任意一个和这些虚机都能互通的机器上执行如下的安装流程：

- 1.读取master_ip文件，得到master机器的ip
- 2.安装nginx
    - 将3个master的apiserver地址写入到ngin×的配置文件，然后docker run方式启动nginx
- 3.参数组装，需要将安装过程中需要用到的参数组装好
    - etcd的参数initial_cluster，按如下格式组装
    - etcd0:http://192.168.0.2:2380, etcdl:http://192.168.0.2:2380, etcd2:http://192.168.0.2:2380
    - apiserver的参数etcd_servers，按如下格式组装·
    - http://192.168.0.2:2380,http://192.168.0.2:2380,http://192.168.0.2:2380
- 4.依次执行scp命令将k8s_master_install.sh脚本和k8s_master_pkg.tar.gz发送的3个master机器上
- 5.依次在3个master机器上远程执行master安装脚本，需要传入ngin×地址和上面组装的两个etcd的参数，做如下操作:
    - a，检查是否安装了docker，如没有则安装docker（ubuntu中使用apt-getinstalldocker-engine安装）；安装成功则继续，安装失败则退出
    - b.将k8s-master-pkg.tar.gz解压，然后将其中的几个k8s组件拷贝到/usr/al/bin目录下
    - c.生成master上4个组件的upstart脚本文件（/etc/init目录下）
    - d.生成master上4个组件的启动参数配置文件（/etc/default目录下）
    - e.检耷是否有4个组件在运行，如果有则停掉
    - f.通过`start xxxx`启动四个组件
- 6.读取nod_ip文件，得到node机器的ip列表，对每个node依次做如下操作（可以使用shell的并行执行能力，加快执行速度）
- 7.scp命令将node的安装脚本和k8s_node_pkg.tar.gz发送的node机器上
- 8.远程执行node安裝脚本，做如下操作:
    - a.检查是否安装了docker，如没有则安装docker；安装成则继续，安装失败则退出（单个node的失败不影响其他node的安装）
    - b.将k8s_node_pkg.tar.gz解压，然后将其中的2个k8s node组件拷贝到/usr/local/bin目录下
    - c.镜像仓库的配置
        - I.  把`registry_ip registry域名`加入的/etc/hosts文件中
        - II. 在/etc/docker/certs.d/目录下建立registry域名:5005的文件夹，并将证书拷贝进去
        - III.将config.json文件放到/root/.docker/目录下
    - d.生成node上2个组件的upstart脚本文件（/etc/init目录下）
    - e.生成node上2个组件的启动参数配置文件（/etc/default目录下）
    - f·检查是否有2个组件在运行，如果有，则停掉
    - g.通过`start xxx`，启动2个组件

这里给出的大致的流程，具体的脚本就不给出了，写的过程中还有很多细节需要处理，比如使用expect执行scp和执行命令等，比如记录日志，方便出错时定位等。