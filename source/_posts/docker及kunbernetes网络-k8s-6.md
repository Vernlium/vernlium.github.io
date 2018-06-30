---
title: docker及kunbernetes网络-k8s-6
date: 2017-09-15 07:46:47
tags: [docker,k8s,iptables]
categories: k8s
---

前面讲了k8s的安装，我们下面来了解一些k8s的网络。

说到k8s的网络，不得不说docker，k8s说白了还是一个调度工具，其用到的容器技术还是docker。docker的网络对于k8s来说也很重要，我们先来了解一些docker的网络，再来看k8s是如何使用docker网络的，以及k8s针对网络所做的工作。

### docker网络原理

docker的网络结构如下图所示

![image](https://user-images.githubusercontent.com/11350907/30645929-e668263c-9e49-11e7-98bc-6969f66c604b.png)

其中：

- 最下面的一层：各种网络设备，比如主机的网卡、各种网桥、overlay的网络设备等
- libnetwork:是docker的网络库，其中使用了CNM(Container Network Model)。CNM定义了构建容器虚拟化网络的模型，同时还提供了可以用于开发各种网络驱动的标准化接口和组件。
- Dockerdaemon：就是docker进程

CNM中有3个核心组件：

- 沙盒（sandbox）：一个沙盒包含了一个容器网络栈的信息。沙盒可以对容器的接口、路由和DNS设备等进行管理。沙盒的实现可以是Linux network namespace、FreeBSD Jail或者类似的机制。
- 端点(endpoint)：一个端点可以加人一个沙盒和一个网络。端点的实现可以是vethpair、OpenvSwitch
内部端口或者相似的设备。
- 网络：一个网络是一组可以直接互相联通的端点。网络的实现可以是Linux bridge、VLAN
等。

他们之间的关系是：
- 一个沙盒可以有多个端点和多个网络。
- 一个端点只可以属于一个网络并且只属于一个沙盒。
- 一个网络可以包含多个端。

libnetwork中内置5种驱动。

- **bridge驱动**，默认模式，即docker0网桥模式
	- bridge驱动此驱动为docker的默认设置，使用这个驱动的时候，libnetwork将创建出的Docker容器连接到Docker网桥上（Docker网桥稍后会做介绍）。作为最常规的模式，bridge模式已经可以满足Docker容器最基本的使用需求了。然而其与外界通信使用NAT，增加了通信的复杂性，在复杂场景下使用会有诸多限制。
- host驱动和宿主机公用网络，适用于对容器集群规模不大的场景
	- host驱动使用这种驱动的时候，libnetwork将不为Docker容器创建网络协议栈，即不创建独立的network namespace。Docker容器中的进程处于宿主机的网络环境中，相当于Docker容器和宿主机共用同一个network namespace，使用宿主机的网卡、和端口等信息。但是，容器其他方面，如文件系统、进程列表等还是和宿主机隔离的。host模式很好地解决了容器与外界通信的地址转换问题，可以直接使用宿主机的进行通信，不存在虚拟的网络带来的额外性能负担。但是host驱动也降低了容器与容器之间、容器与宿主机之间网络层面的隔离性，引起网络资源的竞争与冲突。因此可以认为host驱动适用于对于容器集群规模不大的场景。
- overlay驱动，flannel使用的既是这种模式
	- overlay驱动。此驱动采用IETF标准的VXLAN方式，并且是VXLAN中被普遍认为最大规模的云计算虚拟化环境的SDN controller模式。在使用的过程中，需要一个额外的内置存储服务，例如consul、etcd或zookeeper。还需要在启动Docker daemon的的时候额外添加参数来指定所使用的配置存储服务地址。
- remote驱动
	- rmote驱动这个驱动实际上并未做真正的网络服务实现，而是调用了用户自行实现，网络驱动插件，使libnetwork实现了驱动的可插件化，更好地满足了用户的多种需求。用户只要根据libnetwork提供的协议标准，实现其所要求的各个接口并向Docker daemon进行注册。
- null驱动
	- null驱动使用这种驱动的时候，Docker容器拥有自己的network namespace，但是并不需要Docker容器进行任何网络配置。也就是说，这个Docker容器除了network namespace自带的loopback网卡外，没有其他任何网卡、IP、路由等信息，需要用户为Docker容器添加网卡、配置IP等。这种模式如果不进行特定的配置是无法正常使用的，但是优点也非常明显，给了用户最大的自由度来自定义容器的网络环境。

我们在实践中，主要用到的是bridge驱动模式，下面着重讲解一下这个模式的实现原理。

> 注：**网桥的概念**：网桥是一个二层网络设备，可以解析收发的报文，读取目标MAC地址的信息，和自己记录的MAC表结合，来决策报文的转发端口。为了实现这些功能，网桥会学习源MAC。在转发报文的时候，网桥只需要向特定的网络接口进行转发，从而避免不必要的网络交互。

### bridge驱动实现机制

#### docker0网桥

docker网络的bridge模式示意图如下：

![image](https://user-images.githubusercontent.com/11350907/30779279-0fbf7094-a11f-11e7-8870-ed61304785d2.png)


这里网桥的概念等同于交换机，为连在其上的设备转发数据帧。网桥上的veth网卡设备相当于交换机上的端口，可以将多个容器或虚机连接在其上，这些端口工作在二层，没有IP。

docker0网桥是在Docker daemon启动时自动创建的，其IP默认为172.17.0.1/16，之后创建的Docker容器都会在docker0子网的范围内选取一个未占用的IP使用，并连接到docker0网桥上。

容器里面的eth0网卡，在宿主机上都有一个vethxxxx的虚拟网卡和其对应出现，容器里的eth0网卡即通过这个虚拟网卡和docker0网桥进行通信的。

docker0网桥和宿主机的eth0网卡连接，所有数据最终还是从真实网卡转发出去。

Docker提供了如下参数可以帮助用户自定义docker0的设置。

- --bip-CIDR：设置docker0的IP地址和子网范围，使用CIDR格式，如192.168.100.1/24。注意这个参数仅仅是配置docker0的，对其他自定义的网桥无效。并且在指定这个参数的时候，宿主机是不存在docker0的或者docker0已存在且docker0的IP和参数指定的IP一致才行。（下一篇博客中安装flannel要用到这个参数）
- --fixed-cidr=CIDR：限制Docker容器获取IP的范围。Docker容器默认获取的IP范围为Docker网桥（docker0网桥或者--bridge指定的网桥）的整个子网范围，此参数可将其缩小到某个子网范围内，所以这个参数必须在Docker网桥的子网范围内。如docker0的IP为172.17.0.1/16，可将--fixedid了设为172.17.1.1/24，那么Docker容器的IP范围将为172.17.1.1~172.17.1.254。
- --mtu=BYTES：指定docker0的最大传输单元（MTU）。

#### iptables规则

Docker安装完成后，将默认在宿主机系统上增加一些iptables规则，以用于Docker容器和容器之间以及和外界的通信，可以使用iptables-save命令查看其中nat表上的POSTROUTING链有这么一条规则：

```
-A POSTROUTING -s 172.17.0.0/16 ! -o docker0 -j MASQUERADE
```

这条规则关系着Docker容器和外界的通信，含义是将源地址为172.17.0.0/16的数据包（即Docker容器发出的数据），当不是从docker0网卡发出时做SNAT源地址转换，将IP包的源地址替换为相应网卡的地址）。这样一来，从Docker容器访问外网的流量，在外部看来就是从宿主机上发出的，外部感觉不到Docker容器的存在。

> 注：MASQUERADE会动态的将源地址转换为可用的IP地址，其实与SNAT实现的功能完全一致都是修改源地址，只不过SNAT需要指明将报文的源地址改为哪个IP，而MASQUERADE则不用指定明确的IP，会动态的将报文的源地址修改为指定网卡上可用的IP地址。

我们来通过命令查询一下和docker0网桥相关的iptabels规则

和docker0相关的iptables规则：

filter表
```
# iptables -w -S -t filter | grep docker0
-A FORWARD -o docker0 -j DOCKER
-A FORWARD -o docker0 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
-A FORWARD -i docker0 ! -o docker0 -j ACCEPT
-A FORWARD -i docker0 -o docker0 -j ACCEPT
-A DOCKER -o docker0 -j ACCEPT
```

nat表

```
# iptables -w -S -t nat | grep docker0
-A POSTROUTING -s 172.100.5.144/28! -o docker0 -j MASQUERADE
-A DOCKER -o docker0 -j RETURN
```

> 注：关于iptables规则，后面会专门写一篇博客介绍，可以参考这篇博客[iptables概念](http://www.zsythink.net/archives/1199)。

### 实例分折

我们通过`docker run -p 8543:8888 image` 方式启动的容器，来观察iptables规则的变化。

`docker run -p 8543:8888 image` 方式启动的容器后，可以发现，会加入如下iptables规则：

filter表：

```
-A DOCKER -d 172.17.0.2/32 ! -i docker0 -o docker0 -p tcp -m tcp --dport 8888 -j ACCEPT
```

其作用是接受满足如下条件的数据：
- 目的地址为172.17.0.2/32
- 目的port为8888
- 非docker0网桥进入
- docker0网桥流出
- tcp协议的数据

nat表：
```
-A DOCKER ! -i docker0 -p tcp -m tcp --dport 8543 -j DNAT --to-destination 172.17.0.2:8888
```

其作用是将访问宿主机8543端口请求的流量转发到容器172.17.0.2的8888端口上。所以，外界访问Docker容器是通过iptables做DNAT实现的。

```
root@LFG1ee0826664:~# iptables -w -S -t filter | grep docker0 
-A FORWARD -o docker0 -j DOCKER
-A FORWARD -o docker0 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
-A FORWARD -i docker0 ! -o docker0 -j ACCEPT
-A FORWARD -i docker0 -o docker0 -j ACCEPT
-A DOCKER -o docker0 -j ACCEPT
root@LFG1ee0826664:~# 
root@LFG1ee0826664:~# iptables -w -S -t nat | grep docker0 
-A POSTROUTING -s 172.100.5.144/28 ! -o docker0 -j MASQUERADE
-A DOCKER -i docker0 -j RETURN
```

### k8s网络