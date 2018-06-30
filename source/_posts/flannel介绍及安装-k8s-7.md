---
title: flannel介绍及安装-k8s_7
date: 2017-09-19 23:04:12
tags: [docker,k8s,flannel] 
categories: k8s
---

前面几篇博客讲述了k8s的安装和基本使用，在使用过程中我们可以发现，在每一个节点上，每个容器都有一个节点中唯一的ip，但是不同的节点上，容器的虚拟ip可能是相同的。

上一篇博客中讲到了docker网络，其中提到：
> libnetwork中内置5种驱动，有一种是overlay驱动，flannel就是使用的这种驱动方式。

那么，本节我们就了讲一下flannel的安装、使用和原理。

### flannel简介

flannel是CoreOS团队针对Kubernetes设计的一个网络规划服务，简单来说，它的功能是让集群中的不同节点主机创建的Docker容器都具有全集群唯一的虚拟IP地址。

在默认的Docker配置中，每个节点上的Docker服务会分别负责所在节点容器的IP分配。这样导致的一个问题是，不同节点上容器可能获得相同的内网IP地址。这样就会带来一些问题。比如，使用类似dubbo等使用zookeeper进行服务注册和发现的场景下，分布在不同节点上的服务提供者，如果使用相同的ip到zookeeper上注册服务，那么服务消费者就无法分别这些服务，而且访问也是一个障碍。

flannel的设计目的就是为集群中的所有节点重新规划IP地址的使用规则，从而使得不同节点上的容器能够获得“同属一个内网”且”不重复的”IP地址，并让属于不同节点上的容器能够直接通过内网IP通信。

### flannel安装流程

#### 1.下载flannel安装包
https://github.com/coreos/flannel/releases/tag/v0.7.1
将下载的安装包解压，将其中的flannel拷贝到/usr/local/bin/目录下。

##### 安装etcdctl

将etcd中的etcdctl可执行二进制文件拷贝到/usr/local/bin目录下。

#### 2.配置文件

新建文件/etc/default/flanneld，其内容为:

```
FLANNELD_OPTS="--etcd-endpoints=http://192.168.0.1:2379,http://192.168.0.2:2379,http://192.168.0.3:2379"
```

编写flannel的upstart脚本，新建文件/etc/init/flanneld.conf，其内容为:

```
description "flanneld service"

respawn 

start on (net-device-up 
    and local-filesystems 
    and runlevel [2345])

stop on runlevel [ !2345])

pre-start script 
    FLANNELD=/usr/local/bin/$UPSTART_JOB
    if [ -f $FLANNELD ]; then 
        exit 0
    fi 
    exit 22 
end script 

script 
    FLANNELD=/usr/local/bin/$UPSTART_JOB
    FLANNELD_OPTS="" 
    if [ -f /etc/default/$UPSTART_JOB ]; then 
        · /etc/default/$UPSTART_JOB
    fi 
    export HOME:/root 
    exec "$FLANNELD" $FLANNELD_OPTS 
end script 
```

#### 3.flannel配置信息写入etcd

flannel的配置信息全部在etcd里面记录，往etcd里面写入下面这个最简单的配置，只指定flannel能用来分配给每个Docker节点的拟IP地址段:

```
etcdctl --endpoints http://192.168.0.2:2379,http://192.168.0.2:2379,http://192.168.0.2:2379 \ 
set /coreos.com/network/config \
'{"Network": "172.100.0.0/16,"SubnetLen": 28,"Backend": {"Type": "vxlan"}}' 
```

其中`{"Network": "172.100.0.0/16,"SubnetLen": 28,"Backend": {"Type": "vxlan"}}`配置中各字段的含义如下：

- Network (string): IPv4 network in CIDR format to use for the entire flannel network. (This is the only mandatory key.)**必选字段**
- SubnetLen (integer): The size of the subnet allocated to each host. Defaults to 24 (i.e. /24) unless Network was configured to be smaller than a /24 in which case it is one less than the network.**可选**
- Backend (dictionary): Type of backend to use and specific configurations for that backend. The list of available backends and the keys that can be put into the this dictionary are listed below. Defaults to udp backend.**可选**
    - Backeng的Type字段有如下几种选择：
        - vxlan
        - host-gw
        - udp

我们配置的`"Network":"172.100.0.0/16","SubnetLen":28`含义为：

flannel集群（包含k8s master和node）的所有节点以及节点上的容器的ip都在`172.100.0.0/16`子网内。

子网长度28表示每个node节点上的子网长度，即每个node上也有一个小的子网，这个子网最多可支持2^（32-28）- 1=2^4 - 1=15个容器（-1是因为docker0网桥占了一个ip）
这个flannel集群最多支持2^（28-16）=2^12=4096个节点。

这个子网长度根据自己的场景来选择，例如项目的使用场景，每个node上容器个数少，node节点多，所以长度长一些。如果节点多，节点上的容器也多，可以将上面设置的`172.100.0.0/16`其中的16变小，子网的容量更大一些。

还有两个字段这里没有使用:

- SubnetMin (string): The beginning of IP range which the subnet allocation should start with. Defaults to the first subnet of Network.**可选**
- SubnetMax (string): The end of the IP range at which the subnet allocation should end with. Defaults to the last subnet of Network.**可选**

#### 4.flannel的启动

`start flanneld`命令启动flanneld。

启动成功后，会新增一个文件`/run/flannel/subnet.env`，查看其中的内容：

```
FLANNEL_NETWORK=172.100.0.0/16 
FLANNEL_SUBNET=172.100.4.49/28 
FLANNEL_MTU=1450
FLANNEL_IPMASQ=false
```

这个是flanneld进程从etcd上拉取到的配置写入到本地文件中。

其中:
- FLANNEL_NETWORK: flannel的网络地址
- FLANNEL_SUBNET: 本节点上的子网网段 
- FLANNEL_MTU: 最大传输单元大小
- FLANNEL_IPMASQ: 是否开启IPMASQ防火墙

#### 5.重 启 docker 
 
将 /run/flannel/subnet.env 文 件 中 的 内 容 加 入 到 docker 启 动 的 配 置 项 中 
`--bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU}。然后重启docker0。

```
root@LFG1000826665:~# stop docker 
root@LFG1000826665:~# cat /run/flannel/subnet.env 
FLANNEL_NETWORK=172.100.0.0/16 
FLANNEL_SUBNET=172.100.32.1/28 
FLANNEL_MTU=1450 
FLANNEL_IPMASQ=false
root@LFG1000826665:~# cat /etc/default/docker 
DOCKER_OPTS="$DOCKER_OPTS -H unix:///var/run/docker.sock -H tcp://0.0.0.0:2375 -bip=172.100.4.49/28 --mtu=1450" 
DOCKER_CERT_PATH="/etc/docker" 
root@LFG1000826665:~#
root@LFG1000826665:~# start docker 
root@LFG1000826665:~# ps ex 丨 grep dockerd 
1039 ? Ssl 1:13 /usr/bin/dockerd -H unix:///var/run/docker.sock -H  tcp://0.0.0.0:2375 -bip=172.100.4.49/28 --mtu=1450
```

注:我们用的系统是ubuntu14.04，使用upstart来启动服务。将配置项写入到`/etc/default/docker`文件中，启动服务时会upstart脚本会读取此文件。

#### 6.安装结果

上述步骤执行完成后，在任意一个node上可以看一下网络配置 

```
root@LFG1000826665:~# ifconfig 
dockerO Link encap:Ethernet HWaddr 02:42:cb:44:b0:50 
	inet addr:172.100.4.49 Bcast:0.0.0.0 Mask:255.255.255.240 
	inet6 addr:fe80::42:cbf:fe44:b050/64 Scope:Link 
	UP BROADCAST RUNNING MULTICAST MTU:1450 Metric:1
	RX packets:248 errors:0 dropped:0 overruns:0 frame:0 
	TX packets:291 errors:0 dropped:0 overruns:0 carrier:0 
	collisions:0 txqueuelen:0 
	RX bytes:36349(36.3 KB) TX bytes:38066 (38.0 KB) 
	 
eth0 Link encap:Ethernet HWaddr 28:6e:d4:89:35:ad 
	inet addr:100.120.45.93 Bcast:100.120.45.255 Mask:255.255.254.0
	inet6 add fe80:2a6e:d4ff:fe89:35ad/64 Scope Link 
	UP BROADCAST RUNNING M ULTICAST MTU:1500 Metric:1 
	RX packets:53074 errors:0 dropped:0 overruns:0 frame:0 
	TX packets:9571 errors:0 dropped:0 overruns:0 carrier:0 
	collisions:0 txqueuelen:1000 
	RX bytes:7036582(7.0 MB) TX b es:3556172(3.5 MB) 

flannel.l Link encap:Ethernet HWaddr da:05B6:6a:01:ad 
	inet addr:172.100.4.48 Bcast:0.0.0.0 Mask:255.255.255.255 
	inet6 addr:fe80:d805:36ff:fe6a:1ad/64 Scope:Link 
	UP BROADCAST RUNNING MULTICAST MTU:1450 Metric:1 
	RX packets:33 errors:0 dropped:0 overruns:0 frame:0 
	TX packets:33 errors:0 dropped:9 overruns:0 carrier:0 
	collisions:0 txqueuelen:0
	RX bytes:2604(2.6 KB) TX bytes:3594(3.5 KB) 
```

可以看到有一个flannel.l的网桥，doker0网桥的网段在我们配置的网络内。

安装多个节点后，可以看一下etcd中`/sandbox/network`中的信息:

```
root@LFG1000826665:~# etcdctl --endpoints=http://100.120.76.244:2379,http://100.120.77.15:2379,http://100.120.109.1:2379 get /sandbox/network/config 
{ "Network":"172.100.0.0/16","SubnetLen":28,"Backend": {"Type":"vxlan"}} 
root@LFG1000826665:~#
root@LFG1000826665:~# etcdctl --endpoints=http://100.120.76.244:2379,http://100.120.77.15:2379,http://100.120.109.1:2379 ls /sandbox/network/subnets 
/sandbox/network/subnets/172.100.0.208-28 
/sandbox/network/subnets/172.100.5.0-28 
/sandbox/network/subnets/172.100.0.240-28 
/sandbox/network/subnets/172.100.0.96-28 
/sandbox/network/subnets/172.100.2.208-28 
/sandbox/network/subnets/172.100.3.48-28 
/sandbox/network/subnets/172.100.4.48-28 
/sandbox/network/subnets/172.100.5.144-28 
/sandbox/network/subnets/172.100.5.48-28 

root@LFG1000826665:~# etcdctl --endpoints=http://100.120.76.244:2379,http://100.120.77.15:2379,http://100.120.109.1:2379 get /sandbox/network/subnets/172.100.4.48-28
{"PublicIP":100.120.45.93","BackendTypen":"vxlan","BackendData":{"VtepMAC":"da:05:36:6a:01:ad"}}
```

可以看到`/sandbox/network`路径下多了一个subnets路径，有很多子节点，这些节点正好对应flannel集群中的9个节点，
flannel会给每个节点分配一个网段，并记录在etcd中。

可以看到，`172.100.4.48-28`网段分配给的节点是100.120.45.93，也就是我们上面ifconfig看到的那个节点。

### 容器互通

在k8s集群中部署几个容器（任意分布的不同或相同的节点），然后进入容器内查看ip，并在不同沙箱之间互相ping，可以发现，任意两个沙箱都能互相ping通.


| 客器名       | 所在节点 		| 客器ip 		| 所在子网 			|
| ------------ | -------------- | ------------- | ----------------- |
| bizflannel01 | 100.120.93.1 	| 172.100.5.2 	| 172.100.5.0/28 	|
| bizflannel02 | 100.120.79.159	| 172.100.2.210	| 172.100.2.208/28 	|
| bizflannel03 | 100.120.45.93 	| 172.100.4.50 	| 172.100.4.48/28 	|
| bizflannel04 | 100.120.93.1 	| 172.100.5.3 	| 172.100.5.0/28 	|	 

我们在bizflanne104中ping其他容器，发现都可以Ping通：

```
bizflannel4:/ # ping 172.100.5.2 
PING 172.100.5.2 (172.100.5.2) 56(84) bytes of data. 
64 bytes from 172.100.5.2: icmp_seq=1 ttl=64 time=0.118 ms
64 bytes from 172.100.5.2: icmp_seq=2 ttl=64 time=0.068 ms
^C
--- 172.100.5.2 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 999ms
rtt min/avg/max/mdev = 0.068/0.093/0.118/0.025 ms 
bizflannel4:/ #
bizflannel4:/ # ping 172.100.4.50
PING 172.100.5.2 (172.100.4.50) 56(84) bytes of data. 
64 bytes from 172.100.4.50: icmp_seq=1 ttl=64 time=2.19 ms
64 bytes from 172.100.4.50: icmp_seq=2 ttl=64 time=0.463 ms
^C
--- 172.100.4.50 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 999ms
rtt min/avg/max/mdev = 0.463/1.330/2.198/0.868 ms 
bizflannel4:/ #
bizflannel4:/ # ping 172.100.2.210 
PING 172.100.5.2 (172.100.5.2) 56(84) bytes of data. 
64 bytes from 172.100.2.210: icmp_seq=1 ttl=64 time=1.54 ms
64 bytes from 172.100.2.210: icmp_seq=2 ttl=64 time=0.520 ms
64 bytes from 172.100.2.210: icmp_seq=3 ttl=64 time=0.396 ms
^C
--- 172.100.2.210 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 999ms
rtt min/avg/max/mdev = 0.396/0.820/1.544/0.514 ms 
bizflannel4:/ #
```

通过tcpdump命令抓包，可以看到包的内部含有小网ip，正如flannel网络示意图中展示的那样。

从bizflanne104中`ping 172.100.4.50`的同时，在`100.120.45.93`上抓取从`100.120.93.144`来的包。

抓包命令为`tcpdump -i eth0 src host 100.120.93.144`，结果如下

![image](https://user-images.githubusercontent.com/11350907/30549280-f3a39ee6-9cc6-11e7-96bf-3872bba098c9.png)


### flannel原理

flannel是一种“覆盖网络(overlay network)”，也就是将TCP数据包装后进行路由转发和通信，目前已经支持UDP、VxLAN、AWS VPC和GCE路由等数据转发方式。

默认的节点间数据通信方式是UDP转发，flannel的GitHub主页有如下的一张原理图：

![image](https://user-images.githubusercontent.com/11350907/30548777-a060a9a0-9cc5-11e7-8e28-f0b62d60aa96.png)

从图中我们可以看到，数据从源容器中发出后，经过主机的docker0网桥转发到flannel0虚拟网卡，这是个P2P的虚拟网卡，flanneld服务监听在网卡的另外一端。

从上面的实例中，我们可以看到，etcd中记录了所有的分配给每个节点的子网段，以及此子网段对应的节点的信息，这个就类似一个路由表。

源主机的flanneld服务将原本的数据内容UDP封装后根据etcd中的路由表投递给目的节点的flanneld服务，数据到达以后被解包，然后直接进入目的节点的flannel0虚拟网卡，然后被转发到目的主机的docker0虚拟网卡，最后就像本机容器通信一下的有docker0路由到达目标容器。

#### 一条网络报文是怎么从一个容器发送到另外一个容器的

报文从一个容器发送到另一个容器（不同节点上的容器），也正如图中描述的那样，其流程如下：

- 容器直接使用目标容器的ip访问，默认通过容器内部的eth0发送出去
- 报文通过vethpair被发送到vethXXX
- vethXXX是直接连接到docker0网桥，报文通过docker0网桥发送出去
- 查找路由表，外部容器ip的报文都会转发到flannel0虚拟网卡，这是一个P2P的虚拟网卡（关于这一点的工
理，我也不是很清楚），然后报文就被转发到监听在另一端的flanneld
- flanneld通过etcd维护了各个节点之间的路由表，把原来的报文UDP封装一层，通过节点的网卡发送出去
- 报文通过主机之间的网络找到目标主机
- 报文继续往上，到传输层，交给监听在8285端口的flanneld程序处理
- 数据被解包，然后发送给flannel0虚拟网卡
- 查找路由表，发现对应容器的报文要交给docker0
- docker0找到连到自己的容器，把报文发送过去

#### 一条网络报文是怎么从一个容器发送到外部网络

容器不仅和其他容器互通，也和其他外部节点互通，和节点的通信，主要还是docker的能力，和flannel没有关系。

其流程如下：

- 容器直接使用目标容器的ip访问，默认通过容器内部的eth0发送出去
- 报文通过veth pair被发送到vethXXX
- vethXXX是直接连接到docker0网桥的，报文通过bridge docker0发送出去
- 查找路由表，外部网络ip的报文都会转发到eth0网卡
- eth0发送报文时会匹配如下iptables规则：
`-A POSTROUTING -s 172.100.9.16/28 ! -o docker0 -j MASQUERADE`
这条规则的作用是将pod network地址转换为node ip发出去。包回来时再将地址转换为容器的network地址
- 最终，这个请求会变成此node ip和外部网络的交互。