---
title: kube-proxy源码解析-k8s_9
date: 2017-09-21 07:21:25
tags: k8s
---

### kube-proxy原理介绍

kube-proxy有两种模式：
- userspace模式
- iptables模式

userspace模式在kube-proxy的最早版本中存在，由于这种模式需要数据两次在用户空间和内核中间进行流转，带来很刍的性能损耗。从k8s V1.1版本开始添加了iptables模式，并且在V1.2版本正式取代userspace成为默认的模式。

iptables模式，当kube-proxy检测到有新增的service时，它将会建立一系列的iptables规则，用于将vip重定向到per-Service规则，per-Service规则会连接一个per-Endpoint规则，这个规则将请求重定向到后端。当客户端有连接到vip时，由iptables规则接入，由iptables规则选择一个后端，开始将客户端的请求发送到后端。这样请求包就不会再回到用户空间，可以减少部分性能损耗。

![image](https://user-images.githubusercontent.com/11350907/30700581-890c2ff2-9f1a-11e7-9d3e-90fec135fd2f.png)


下面给出一个iptables模式的实例，方便理现在有一个pod，里面启动了一个zookeeper,其端囗号是12181。通过iptables-w-S-tnat可以查看虚机上的iptables规则，对podbesspace/blzproxytest01(besspace是namespace,bizproxytest01是pod名）有如下规则：


这些规则中，每条规则都有匹配条件和相关的动作，当网络请求进入后，会依次去匹配这些规则，如果匹配上，则执行后面的动作。
比如规则：-AKUBE-NODEPORTStcp—mcomment——comment'Ibesspace/bizproxytest01:zk—portIItcp——dport41446—jKUBE—SVC—GCFVKTPLV5XAFEPM的意思是·—m对于协议是tcp且目的端囗是41446的报文，执行．j(jump)动作，将报文给KUBE—SVC—GCFVKTPLV5XAFEPM处报文在这些规则中的流转如下图所示，主要就是通过几次NAT(networkaddresstrans|a№n，网络地址转换）最终将数据转到pod对应的容器里面。
注：为了画图方便，图中的规则把规则中的--comment信息省略了，这个信息类似注释信息，说明这个规则的用途。这个例子只是描述了pod副本是一个的情况，pod副本有个的情况下会牵扯到负载均衡等，可以参考这个博客
2.kube-proxy代码分析
详细的代码比较复杂，可以参考这篇文
图中的serviceLW和update是kube-proxy的核心，serviceLW中有两个操作，分别是List和Watch。
List就是从apiserver中获取全量的service和endpoint(s息。Watch是从apiserver监听service和endpoint的变化，并进行更新。
List是获取全量的service和endpoint信息，每隔30s执行一次（可以通过启动参数一一iptables—sync—period配置，默讠人是30s），定期执行的目的是进行补偿，防止service或endpoint的变化没有获取到。
Watch的作用是，调用apiserver的watch相关的api,如果service或endpoint有变化，watch的协程中会收到晌应的事件，并进行处理，比如有endpoint新增，则新增相应的
iptables规则。

Watch函数会收到从kube-apisever的一件．add、modify、deleteo（这里的add、modify、delete是对service和endpoint而言的，即有一个service或endpoint新增加或者修改或者被册刂除，对这种情况，我们需要修改对应的ipta，es规则来应对变化）代码中的逻辑是收到一个事件，都会将此事件扔给servicestore进行处理，保存到内存中，然后更新iptables。