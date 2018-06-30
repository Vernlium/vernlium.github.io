---
title: kubernetes基本安装-k8s-3
date: 2017-08-16 22:20:26
tags: [docker,k8s]
categories: k8s
---

这篇博客来讲解下k8s的安装。

为什么标题中叫“基本安装”呢？因为一般生产环境中会考虑可靠性等问题，需要k8s master安装成集群，集群的安装稍微复杂，我们可先从基本的安装开始，学习如何安装单master的k8s集群，然后在扩展成master 集群，这样有个循序渐进的过程，会了解的更清晰。

k8s的安装有多种方式，本文中以手动的方式安装，最后形成shell脚本，可以一键安装k8s集群，并且可以扩容node节点。

### 获取安装包

#### etcd

地址：https://github.com/coreos/etcd/releases/tag/v3.2.5

下载后解压，里面有etcd（服务进程）和etcdctl（命令行工具）两个文件。

![image](https://user-images.githubusercontent.com/11350907/30039971-073d18c2-920b-11e7-885d-70327b9483fe.png)

#### k8s

k8s 1.5.0之前的版本，k8s github主页上的release是以二进制包发布的，发布包中包含编译后的二进制文件，下载后可以直接解压使用。而之后的版本，发布包中只有源码，下载编译后的包需要充其他地方。我们选用k8s 1.7.0版本，链接如下：

https://github.com/kubernetes/kubernetes/releases/tag/v1.7.0

![image](https://user-images.githubusercontent.com/11350907/30007656-927ef920-9146-11e7-8b31-e703848ed058.png)

点击上图中的CHANGELOG链接，可以找到对应的二进制包。

地址为： https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#downloads-for-v170

点击图中红色框标注的tar包，下载。

![image](https://user-images.githubusercontent.com/11350907/30007732-239117c6-9148-11e7-93b6-f4230064e7c1.png)

下载后可以看到包中的二进制文件。

![image](https://user-images.githubusercontent.com/11350907/30040034-f54d340c-920b-11e7-869a-120c42a55d5b.png)

server包中包含了很多组件，master上需要用到kube-apiserver、kube-controller-manager、kube-scheduler三个组件，node上需要用到kubelet和kube-proxy两个组件。

![image](https://user-images.githubusercontent.com/11350907/30040034-f54d340c-920b-11e7-869a-120c42a55d5b.png)

### 组网图

![kubernetes-1master](https://user-images.githubusercontent.com/11350907/29775709-c46db038-8c38-11e7-989d-164e49669cef.jpg)

### 环境准备

我们准备两个虚拟机，一个安装master，一个安装node。


K8s和etcd版本：

- k8s : 1.7.0
- etcd : 3.2.5

### k8s master安装

#### 软件包

将上面下载的etcd、kube-apiserver、kube-controller-manager、kube-scheduler四个组件拷贝到master机器上的/usr/local/bin目录下。这4个组件是k8s集群运行必需的。为了后续方便使用，把kubectl和etcdctl也拷贝到/usr/local/bin目录下，这两个是命令行工具，可以方便对集群进行操作。

#### etcd

ETCD的启动参数配置，放到`/etc/default/etcd`文件中。

```
ETCD_OPTS="--name etcd0 --listen-client-urls http://127.0.0.1:2379 \
--advertise-client-urls http://127.0.0.1:2379"
```

其中，几个参数的作用为：

- name : etcd的名称
- listen-client-urls : 外部客户端使用的url
- advertise-client-urls : 广播给外部客户端使用的url

etcd的upstart脚本，放到`/etc/init/etcd.conf`文件中。

```
description "Etcd service"

respawn

start on (net-device-up
        and local-filesystem
        and runlevel[12345])
stop on runlevel[!12345]

pre-start script
    ETCD=/usr/local/bin/$UPSTART_JOB
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    if [ -f $ETCD ]; then
        exit 0
    fi
    echo "$ETCD binary not found,exiting"
    exit 22
end script

script 
    ETCD=/usr/local/bin/$UPSTART_JOB
    ETCD_OPTS=""
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    export HOME=/root
    exec "$ETCD" $ETCD_OPTS
end script 
```

#### kube-apiserver

kube-apiserver的启动参数配置，放到`/etc/default/kube-apiserver`文件中。

```
KUBE_APISERVER_OPTS="--insecure-bind-address=0.0.0.0 \
--insecure-port=8080 \
--cors-allowed-origins=.* \
--etcd_servers=http://127.0.0.1:2379 \
--service-cluster-ip=172.17.0.0/16 \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/apiserver"
```

kube-apiserver的upstart脚本，放到`/etc/init/kube-apiserver.conf`文件中。

```
description "Kube-Apiserver service"

respawn

#start in conjunction with etcd
start on started etcd
stop on stopping etcd

pre-start script
    KUBE_APISERVER=/usr/local/bin/$UPSTART_JOB
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    if [ -f $KUBE_APISERVER ]; then
        exit 0
    fi
    echo "$KUBE_APISERVER binary not found,exiting"
    exit 22
end script

script 
    KUBE_APISERVER=/usr/local/bin/$UPSTART_JON
    KUBE_APISERVER_OPTS=""
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    exec "$KUBE_APISERVER" $KUBE_APISERVER
end script 
```

#### kube-controller-manager

kube-controller-manager的启动参数配置，放到`/etc/default/kube-controller-manager`文件中。
```
KUBE_CONTROLLER_MANAGER_OPTS="--master=127.0.0.1:8080 \
--enable-hostpath-provisioner=false \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/controller-manager"
```

controller-manager需要配置apiserver的地址。

kube-controler-manager的upstart脚本，放到`/etc/init/kube-controller-manager.conf`文件中。

```
description "Kube-Controller-Manager service"

respawn

#start in conjunction with etcd
start on started etcd
stop on stopping etcd

pre-start script
    KUBE_CONTROLLER_MANAGER=/usr/local/bin/$UPSTART_JOB
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    if [ -f $KUBE_CONTROLLER_MANAGER ]; then
        exit 0
    fi
    echo "$KUBE_CONTROLLER_MANAGER binary not found,exiting"
    exit 22
end script

script
    KUBE_CONTROLLER_MANAGER=/usr/bin/$UPSTART_JON
    KUBE_CONTROLLER_MANAGER_OPTS=""
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    exec "$KUBE_CONTROLLER_MANAGER" $KUBE_CONTROLLER_MANAGER_OPTS
end script 
```

#### kube-scheduler

kube-scheduler的启动参数配置，放到`/etc/default/kube-scheduler`文件中。
```
KUBE_SCHEDULER_OPTS="--master=127.0.0.1:8080  \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/scheduler"
```

scheduler需要配置apiserver的地址。

kube-scheduler的upstart脚本，放到`/etc/init/kube-scheduler.conf`文件中。

```
description "Kube-Scheduler service"

respawn

#start in conjunction with etcd
start on started etcd
stop on stopping etcd

pre-start script
    KUBE_SCHEDULER=/usr/local/bin/$UPSTART_JOB
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    if [ -f $KUBE_SCHEDULER ]; then
        exit 0
    fi
    echo "$KUBE_SCHEDULER binary not found,exiting"
    exit 22
end script

script 
    KUBE_SCHEDULER=/usr/bin/$UPSTART_JON
    KUBE_SCHEDULER_OPTS=""
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    exec "$KUBE_SCHEDULER" $KUBE_SCHEDULER_OPTS
end script 
```

#### 启动

使用如下命令启动master上的4个组件

```
start etcd
start kube-apiserver
start kube-controller-manager
start kube-scheduler
```

其实，启动etcd后，后面三个组件会自动启动，这是upstart的机制（注意upstart脚本中的`start on`）。

启动后，可以查看到如下进程：



### k8s node安装

#### 软件包

需要将上面下载的二进制可执行文件拷贝到相关目录下。

将上的kubelet、kube-proxy两个组件拷贝到node机器上的/usr/local/bin/目录下。

#### kubelet

kubelet的启动参数配置，放到`/etc/default/kubelet`文件中。

```
KUBELET_OPTS="--address=0.0.0.0 \
--port=10250 \
--api_servers=http://master_ip:8080 \
--pod-infra-container-image=docker.io/kubernetes/pause \
--node-labels=node=anannode \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/kubelet"
```

> master_ip是master机器的ip，记得修改为真实ip。

其中：

- port : kubelet的监听端口
- api_servers : apiserver的地址，
- pod-infra-container-image : 简介中介绍的pause镜像的名称
- node-labels : 给此node加的label


kubelet的upstart脚本，放到`/etc/init/kubelet.conf`文件中。

```
description "kubelet service"

respawn

start on runlevel[12345]
stop on runlevel[!12345]

pre-start script
    KUBELET=/usr/local/bin/$UPSTART_JOB
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    if [ -f $KUBELET ]; then
        exit 0
    fi
    echo "$KUBELET binary not found,exiting"
    exit 22
end script

script 
    KUBELET=/usr/local/bin/$UPSTART_JOB
    KUBELET_OPTS=""
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    exec "$KUBELET" $KUBELET_OPTS
end script 
```


#### kube-proxy

kube-proxy的启动参数配置，放到`/etc/default/kube-proxy`文件中。

```
KUBE_PROXY_OPTS="--master=http://master_ip:8080 \
--proxy-mode=iptables \
--v=1 --logtostderr=false --log_dir=/var/k8slogs/kube-proxy"
```

> master_ip是master机器的ip，记得修改为真实ip。

其中：
- master : apiserver的地址
- proxy-mode : proxy的模式，默认是iptables，还有一种是userspace（现在很少用了）

kube-proxy的upstart脚本，放到`/etc/init/kube-proxy.conf`文件中。

```
description "kube-proxy service"

respawn

start on runlevel[12345]
stop on runlevel[!12345]

pre-start script
    KUBE_PROXY=/usr/local/bin/$UPSTART_JOB
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    if [ -f $KUBE_PROXY ]; then
        exit 0
    fi
    echo "$KUBKUBE_PROXYELET binary not found,exiting"
    exit 22
end script

script 
    KUBE_PROXY=/usr/local/bin/$UPSTART_JOB
    KUBE_PROXY=""
    if [ -f /etc/default/$UPSTART_JOB ]; then
        . /etc/default/$UPSTART_JOB
    fi
    exec "$KUBE_PROXY" $KUBE_PROXY_OPTS
end script 
```

#### 启动

使用如下命令启动node上的两个组件

```
start kubelet
start kube-proxy
```

### k8s集群的使用

#### kubectl（命令行方式）

kubectl的命令很多，这里先列出几个常用的，后续再详细讲解。
可以先参考： https://kubernetes.io/docs/user-guide/kubectl-overview/

kubectl常用命令如下：

- 获取节点信息
    - kubectl get nodes --show-lables
    - 这个命令获取到的节点只有节点的hostname，没有ip，如果需要ip可以使用如下命令：
    - kubectl get nodes -o=custom-columns=NAME:.metadata.name,IP=.metadata.Addresses[0],LABELS=.metadata.Labels
- 获取service列表
    - kubectl get services --namespace=xxxx
    - namespace需要指定，否则在default namespace下查找
    - 不知道namespace可以使用--all-namespaces
- 获取pod列表
    - kubectl get pod --namespace=xxxx
    - namespace需要指定，否则在default namespace下查找
- 创建资源（service、pod、deployment、endpoint等）
    - kubectl create -f xxx.yaml
    - xxx.yaml文件中定义了service、pod等资源
- 获取资源详细信息（node、service、pod、deployment、endpoint等）
    - kubectl describe pod xxx --namespace=xxx
    - 这里以pod为例，其他类似，node不需要指定namespace
- 删除资源label
    - kubectl label node xxx-
- 修改资源label
    - kubectl label node --voer-wirte xxxx=aaaa

kubectl的使用可以参考kubectl帮助文档，使用kubectl help查看，每个子命令都可以使用kubectl xxx --help查看帮助，有很详细的解释和示例。

#### k8s api

k8s集群搭建完成后，在浏览器或者其他restapi工具如postman，输入http://master_ip:8080/api/v1，可以看到如下信息：

这里给出了v1版本有哪些接口可以使用，加到url后面就可以使用了，读者自己可以进行尝试。

详细的k8s api介绍，可以参考：https://kubernetes.io/docs/api-reference/v1.7/

### dashboard

dashboard是k8s的一个图形化管理工具，可以在k8s集群中启动。

首先建立如下两个文件：

dashboard-service.yaml，内容如下：

```
apiVersion: v1
kind: Service
metadata:
  labels:
    name: anandashboard
  name: anandashboard
  namespace: kube-system
spec:
  type: NodePort
  ports:
    -port: 9090
     targetPort: 9090
  selector:
    name: anandashboard
```

dashboard-pod.yaml，内容如下：

```
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: anandashboard
  name: anandashboard
  namespace: kube-system
spec:
  containers:
    - resources:
        limits:
          cpu: 1
        images: mrid/kubernetes-dashboard-amd64:latest
        imagePullPolicy: IfNotPresent
        name: anandashboard
        ports:
          - containerPort: 9090
            hostPort: 9090
            name: anandashboard
        args:
          - --apiserver-host=http://master_ip:8080
```

使用如下命令创建dashboard:

```
kubectl create -f dashboard-service.yaml
kubectl create -f dashboard-pod.yaml
```

创建一段时候后，可以通过如下命令看到

```
kubectl get svc --namesapce=kube-system
kubectl get pod --namespace=kube-system
```


在浏览器输入地址：httpL//ip:9090 ，即可看到dashboard的界面了

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
- 2.scp命令将k8s-master-install.sh脚本和k8s-master-pkg.tar.gz发送的master机器上
- 3．远程执行master安装脚本，做如下操作:
    - a．检查是否安装了docker，如没有则安装docker（ubuntu中使用apt-getinstalldocker-engine安装）;安装成功则继续，安装失败则退出
    - b.将k8s_master-pkg.tar.gz解压，然后将其中的几个k8s组件拷贝到/usr/bin目录下
    - c.生成master上4个组件的upstart脚本文件（/etc/init目录下）
    - d.生成master上4个组件的启动参数配置文件（/etc/default目录下）
    - e.检耷是否有4个组件在运行，如果有则停掉
    - f.通过`start xxxx`启动四个组件
- 4.读取node_ip文件，得到node机器的ip列表，对每个node依次做如下操作（可以使用shell的并行执行能力，加快执行速度）
- 5.scp命令将node的安装脚本和k8s_node_pkg.tar.gz发送的node机器上
- 6.远程执行node安裝脚本，做如下操作:
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

#### 远程发送文件和执行命令

这个操作在脚本安装的过程中很多次用到，所以可以封装一下。

```
#远程发送文件
function run_scp() {
    local ip=$1 
    local port=$2 
    local SCP_FILE=$3 
    
#scp file 
expect -c " 
set timeout $TIMEOUT 
spawn scp -P ${port} ${SCP_FILE} root@${ip}:/root/ 
expect { 
    \"*yes/no*\" { send \"yes\r\";exp_continue}
    \"*assword*\" { send \"${PASSWD}\r\";}
}
expect { 
    eof { puts \"expect ok\"; exit 0} 
    timeout { puts \"expect timeout\"; exit 1}
    default { puts \"expect error\"; exit 2}
}
" >> ${LOG_FILE} 2>&1
} 

#远程执行命令
function run_cmd() {
    local ip=$1 
    local port=$2 
    local CMD=$3 
    
#exec cmd
expect -c " 
set timeout $TIMEOUT 
spawn ssh -p ${port} root@${ip} \"$CMD\"
expect { 
    \"*yes/no*\" { send \"yes\r\";exp_continue}
    \"*assword*\" { send \"${PASSWD}\r\";}
}
expect { 
    eof { puts \"expect ok\"; exit 0} 
    timeout { puts \"expect timeout\"; exit 1}
    default { puts \"expect error\"; exit 2}
}
" >> ${LOG_FILE} 2>&1
}
```

#### dashboard安装脚本化

dashboard也可以脚本执行，我们需将两yaml文件和dashboard的镜像（通过dockersave保存地文件）都放到pkg压缩包中，使用的时候dockerload把镜像恢复，然再使用kubectl命令创建即可。

有兴的读者可自行尝试，这里就不细讲了。