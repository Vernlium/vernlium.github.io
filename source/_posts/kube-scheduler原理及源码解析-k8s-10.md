---
title: kube-scheduler原理及源码解析-k8s_10
date: 2017-09-21 07:21:54
tags: [k8s,scheduler,源码分析]
categories: k8s
---

kubernetesScheduler运行在master节点，它的核心功能是监听apiserver来获取PodSpec.N0deName为空的pod,然后为每个这样的pod创建一个binding指示pod应该调度到哪个节点上。

从apiserver读取还没有调度的pod。可以通过spec.nodeName指定pod要部署在特定的节点上。调度器也是一样，它会向apiserver请求spec.nodeName字段为空的pod，然后调度得到结果之后，把结果写入apiserver。

调度分为两个过程predicate和priority。

![image](https://user-images.githubusercontent.com/11350907/30699597-ba4381f4-9f17-11e7-9131-b31adf6139f8.png)


- 首先是过滤掉不满足条件的节占这个过程称为predicate；
- 然后对通过的节点按照优先级排序，这个是priorlty；
- 最后从中选择优先级最高的节点。

### 代码解析

scheduler的代码在plugin/目录下：plugin/cmd/kube-scheduler/是代码的main函数入囗，plugin/pkg/scheduler/是具体调度算法。从这个目录结构也可以看出来，kube-scheduler是作为插件接入到集群中的，它的最终形态一定是用户可以很容易地去定制化和二次开发的。

Config的定义在文件plugins/pkg/scheduler/scheduler.go中。它把调度器的逻辑分成几个组件，提供了这些功能：

- NextPod()方法能返回下一个需要调度的pod
- Algorithm.Schedule()方法能计算出某个pod在节点中的结果
- Error（）方法能够在出错的时候重新把pod放到调度队列中进行重试
- schedulerCache能够暂时保存调度中的保证pod信息，占用着pod需要的资源，资源不会冲突
- Binder.Bind在调度成功之后把调度结果发送到apiserver中保存起来后面可以看到Scheduler对象就是组合这些逻辑组件来完成最终的调度任务的。

Config中的逻辑组件中，负责调度pod的是Algorithm.Schedule()方法。其对应的值是GenericScheduler,GenericScheduler是Scheduler的一种实现，也是kube-scheduler默认使用的调度器，它只负责单个pod的调度并返回结果。
总结起来，configFactory、config和scheduler三者的关系如下图所示：

![image](https://user-images.githubusercontent.com/11350907/30699633-d21a9e52-9f17-11e7-8e1a-67c50ed9c245.png)

- configFactory对应工厂模式的工厂模型根据不同的配置和参数生成config，当然事先会准备好config需要的各种数据
- config是调度器中最重要的组件，里面实现了调度的各个组件逻辑
- scheduler使用config提供的功能来完成调度

参考
https://feisky.gitbooks.io/kubernetes/plugins/scheduler.html
http://dockone.i0/articIe/895