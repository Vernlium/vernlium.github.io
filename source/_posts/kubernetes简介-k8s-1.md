---
title: kubernetes简介-k8s_1
date: 2017-08-09 20:55:48
tags: [docker,k8s]
categories: k8s
---

### 什么是Kubernetes？

Kubernetes（k8s）是自动化容器操作的开源平台，这些操作包括部署，调度和节点集群间扩展。如果你曾经用过Docker容器技术部署容器，那么可以将Docker看成Kubernetes内部使用的低级别组件。Kubernetes不仅仅支持Docker，还支持其他的容器技术，比如Rocket。
使用Kubernetes可以：
- 自动化容器的部署和复制
- 随时扩展或收缩容器规模
- 将容器组织成组，并且提供容器间的负载均衡
- 很容易地升级应用程序容器到新版本
- 提供容器弹性伸缩，强大的故障发现和自我修复能力

由于kubernetes较长，因此很多人都习惯简写成k8s，用8替换中间的8个字母，类似于i18n（internationalization，国际化）。后面我们的介绍都是用k8s替代kubernetes。

“Kubernetes”是古希腊词汇，其原意是“万能的神”，因为希腊是航海大国，不可预测的海上风浪经常会引起翻船事故，为保佑船只和人的安全，古希腊人引用“万能的神”这个词作为舵手。

而Docker的意思是集装箱，正好让舵手掌管装载集装箱的船的航向，很贴切。

### k8s相关概念

k8s中的概念很多，如Master、Node、Pod、Container、Service、Namespace、Replication Controller、Deployment、Endpoint、Volume、Label等等。要使用好k8s，首先要了解这些概念，下面来介绍一下这些概念。

在了解这些概念之前，我们先来看一下k8s的一个典型组网。

![kubernetes](https://user-images.githubusercontent.com/11350907/29643391-6c2e1e84-88a1-11e7-8fbd-5dc7f989d66e.jpg)

这个图中涉及了很多相关的概念。

#### kubernetes集群

kubernets集群由k8s master和node组成。这个组网中就1个master和2个node，在生产环境中master一般有多个（2n+1）组成一个集群，来实现高可用性。node可以有很多个，当node不够用时，可以不断扩容。node的规模需要master来支撑，因为单个master所能管理的node节点的个数是有限的。

#### master

master是集群的控制节点，负责整个集群的管理和控制。master节点可以运行在物理机或虚拟机中，因为它是整个集群的“大脑”，非常重要，所以要保证它的可用性与可靠性。可以把它独占一个物理机，或者放到虚拟机中，用master集群来保证其可靠性。

k8s master由三个组件(进程)组成：

- **kube-apiserver**: 提供http rest接口的服务进程，对k8s里面的所有资源进行增、删、改、查等操作。也是集群控制的入口；
- **kube-controler-manager**: k8s里所有资源对象的自动控制中心，比如各个node节点的状态、pod的状态等；
- **kube-scheduler**: 负责资源调度；

另外，集群中还有一个etcd进程，k8s里面的所有资源的数据全部保存在etcd中。这里也是一个etcd节点，生产环境中一般用集群。etcd 是一个应用在分布式环境下的 key/value 存储服务。利用 etcd 的特性，应用程序可以在集群中共享信息、配置、leader选举或作服务发现，etcd 会在集群的各个节点中复制这些数据并保证这些数据始终正确。

#### node

除了master，k8s集群中的其他节点是node节点。node节点可以是一台物理机或者虚拟机。node节点是k8s集群中的工作负载节点，master会把一些任务调度到node节点上进行。当某个node出现故障时，master会把这个节点上的任务转移到其他节点上。

每个node节点上会运行3个组件（进程）：

- **kubelet**：与master的apiserver进程保持通信，上报node节点信息和状态，同时负责pod对应的容器的创建、启停等；
- **kube-proxy**：实现k8s service的通信和负载均衡等；
- **Docker Engine**：docker引擎，负责本机的容器创建和管理工作。

Node节点可以在运行期间动态添加到k8s集群，在默认情况下kubelet会向Master注册自己。一旦node被纳入集群管理范围，kubelet进程就会定时向Master节点汇报自身的情况，例如操作系统、Docker版本、cpu和内存、运行的pod等，这样master可以获得每个node的资源情况，可以实现高效的资源调度。

而某个Node超过指定时间不上报信息时，Master会认为此node失效了，会把此node的状态标记为不可用（Not Ready）,随后会根据一些策略将此Node上的Pod进行转移。

#### Pod

Pod是Kubernetes最基本的操作单元，包含一个或多个紧密相关的容器；一个Pod中的多个容器应用通常是紧密耦合的，Pod在Node上被创建、启动或者销毁；每个Pod里运行着一个特殊的被称之为“根容器”的Pause容器，还包含一个或多个业务容器，这些业务容器共享Pause容器的网络栈和Volume挂载卷，因此他们之间通信和数据交换更为高效。同一个Pod里的容器之间仅需通过localhost就能互相通信。

k8s会为每个pod都分配一个唯一的IP地址（至少在node节点上是唯一的），称之为PodIP，一个pod的里的多个容器共享PodIP。

Pod被创建后，k8s master会将其调度到某个具体的Node上，随后该Node上的kubelet进程会将Pod中的一组容器启动。在默认情况下，当pod里面的某个容器停止时，k8s会自动检测到这个又问题的Pod，并重新启动这个Pod（重启Pod里的所有容器）。如果Pod所在的Node出现故障，不可用了，则K8s会将这个Node上的Pod重新调度到其他节点上（这个调度需要Replication Controller或者Deployment的支持）。

#### Label

上图中Node或pod上，都有一个小的标识，那个就是Label。一个Label是一个key=value的键值对，其中key和value均可自定义。Label可以附加到各种资源对象上，例如Pod、Node/Service等，每个资源对象都可以定义任意多个Label，Label可以动态的增加或者删除。

Label的作用是：通过给指定的资源对象绑定一个或多个Label，来实现多维度的资源分组管理，以便灵活的进行资源分配、调度、配置和部署等。给某个资源加上Label后，后面就可以通过Label Selector查询和筛选有某些Label的资源对象。

当前Label Selector有两种表达式：基于等式的（Equality-based）和基于集合的（Set-based）。

- 基于等式的Label Selector通过= 或者!= 来匹配标签，就类似与sql语句中 `where name=anan`，例如：`name=dbnode`:匹配具有标签是`name=dbnode`的资源对象。
- 基于集合的Label Selector，通过in 或not in来匹配标签，例如：`name in (dbnode，biznode,plnode)`,匹配所有具有Label：`name=dbnode`或`name=biznode`或`name=plnode`的资源对象。

#### EndPoint

每个Pod都会有一个IP，而Pod中的容器都会开放一些端口对外提供服务，这个端口被称为容器端口（ContainerPort），PodIP+ContainerPort就组成了一个新概念——EndPoint。它代表了此Pod中的一个服务进程的对外同曦地址。当然一个Pod可以存在多个EndPoint，这取决于pod中的业务容器。

#### Service

从上Pod的介绍中，可以知道：Pods是短暂的，可能会重启或者转移，这样IP地址可能会改变，如果这个PodIP经常变化，我们就无法正常的使用它了。

Service是定义一组Pod以及访问这些Pod的策略的一层抽象。Service通过Label找到Pod组。因为Service是抽象的，所以在图表里通常看不到它们的存在，这也就让这一概念更难以理解。

K8s的Service定义了一个服务的访问入口地址，一组应用可以通过这个入口地址访问其背后的一组Pod组成的集群实例，Service可以通过Label Selector将一组pod纳入到此Service中。

每个Pod都有一个IP，而每个Pod都有一个独立的EndPoint被客户端访问，那么多个Pod副本组成的一个集群来提供服务的话，客户端就需要负载均衡来访问它们。K8s的负载均衡是通过Node上的kube-proxy来实现的（后面会专门写一篇博客来讲它，这一块关于网络的东西很有意思，而且我在工作中接触的也比较多，所以比较熟悉，研究了很多实例，并且看了kube-proxy的源码）。可以通过下图来描述Service的使用。

![service-proxu](https://user-images.githubusercontent.com/11350907/29721253-07d51c40-89ef-11e7-844f-688a5b73b4e8.gif)

每个Service会分配一个全局唯一的虚拟IP地址，这个IP被称为ClusterIP。这样，每个服务就只有一个唯一入口，客户端调用就无需关心这个服务有多少个Pod提供。ClusterIP在Service的整个生命周期内是不会变化的，所以Pod重启或者因Node失效进行了转移，也不会影响到Service。

上面的讲述中，我们已经提到了3种IP：

- NodeIP: Node节点的IP，是一个真是存在的物理网络（哪怕是虚拟机），外部可以直接访问这个网络。
- PodIP: Pod的IP，这个IP是每个Node上的docker0网桥根据自己的IP地址段进行分配的，是一个虚拟的二层网络，K8s集群中的node之间，无论是nodeIP还是PodIP，都可以和这个网络互通，集群外的机器则无法通信，但是在Pod中可以和外部通信。
- CllusterIP: Service的IP，这个一个虚拟的IP，它仅作用于Service这个对象，由K8s管理和分配，它无法别ping，因为没有一个实体网络对象来响应。它属于k8s集群内部地址，无法直接在集群外部使用。

有一些情况下，服务是要提供给集群外的应用来使用的，当然k8s也提供了方法，后面再详细讲解。

#### Replication Controller 和 Replica Set

Replication Controller(后文简称RC)是kube-controller-manager组件中的一个Controller，它定义了一个期望的场景，即使得某个Pod的副本数量在任意时刻都满足某种期望值。RC的定义包含如下几个部分：

- Pod期望的副本数（replicas）
- 用于筛选目标Pod的Label Selector
- 当Pod的副本数量小于期望数量的时候，用于创建新Pod的Pod模版（template）

当定义了一个RC并提交到k8s集群后，master上的controller manager就会得到通知，定期检查集群中当前存活的目标Pod，并确保目标Pod的实例的数量等于RC的期望值。如果有过多的Pod在运行，就会停掉多的Pod，如果数量不够，则再根据Pod模版创建一些Pod。

下面的动图展示了一个例子：

![rc](https://user-images.githubusercontent.com/11350907/29736597-5deab3c4-8a35-11e7-8c70-c3994fe06d36.gif)
Replication Controller是k8s中最开始版本的Pod自动管理和调度工具，它只支持基于等式的Label Selector，有一定的局限性，k8s 1.2版本中引入了Replica Set，是“下一代的RC”，它与RC的唯一区别是：Replica Set支持基于集合的Label Selector。这样Replica Set的功能更强大。

#### Deployment

Deployment也是k8s 1.2引入的新概念，它的目的是更好的解决Pod的编排问题。Deployment无论是作用与目的、yaml定义文件，还是使用方式，都和RC很相似，可以看作是RC的一次升级。Deployment内部使用Replica Set来实现Pod的部署与调度。

Delpoyment有如下几个典型的使用场景：
- 创建一个Deployment对象来生成对应Replica Set并完成Pod副本的创建
- 检查Deployment的状态来看部署动作是否完成（Pod副本的数量是否达到预期）
- 更新Deployment来创建新的Pod（比如升级镜像等）
- 如果当前Deployment不稳定，则回滚到先前的Deployment版本
- 暂停或恢复一个Deployment

#### Volume

上文中讲到，Pod可能被重启或转移，那么Pod中的容器中保存的文件就会丢失，为了解决这个问题，k8s提出了Volume的概念。当然，Volume还有一个作用是解决Pod中的容器之间的文件共享。

Volume是Pod中能够被多个容器访问的共享目录。k8s的Volume概念、用途和目的与Docker的Volume很类似，但是又不完全相同。

- k8s中的Volume定义在Pod上，然后被一个Pod中的多个容器挂载到具体的文件目录下。
- k8s的Volume与Pod的生命周期相同，但是与容器的生命周期不相关，当容器终止或重启是，Volume中的数据也不会丢失。
- k8s支持多种类型的Volume，比如GlusterFS、Ceph等分布式文件系统。

#### registry

严格来说，registry不是k8s中的组件或者概念，registry是docker中的概念，registry是镜像仓库，是存储镜像和分发镜像的系统。docker的使用离不开镜像仓库。当前最大的镜像仓库是dockerhub，这里面包含了很多镜像，比如各大常用软件的镜像，如nginx/ubuntu/mysql/redis等，还有很多个人开发者制作的镜像，或者是完全新的镜像或者是在现有镜像基础上开发的特殊需求的镜像等。

有时候开发者可能需要进行测试或者涉及安全问题不想把镜像流传到外网，这时就需要自己搭建私有镜像仓库，只能在自己到内网中访问。

镜像仓库的搭建可以参考[docker镜像仓库的安装-k8s_2](https://vernlium.github.io/2017/08/14/docker%E9%95%9C%E5%83%8F%E4%BB%93%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85-k8s-2/)。

### 小结

上面介绍了k8s的相关概念，对k8s有了一个大概的认识。

多说无益，实践最重要，我们就来手动安装一下k8s集群来进行实践。


#### 参考

- [k8s官方文档](https://kubernetes.io/docs/concepts/)
- [十分钟带你理解Kubernetes核心概念](http://www.dockone.io/article/932),文中的动图来自此文，建议多看几遍此文
- 《Kubernetes权威指南 从Docker到Kubernetes实践全接触》
