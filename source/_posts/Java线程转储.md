title: "Java线程转储"
date: 2015-08-26 22:14:23
tags: 性能调优
categories: java
---
今天看《Java并发编程实战》中讲到通过**线程转储**信息来分析死锁，之前并没有听说过“线程转储”这个概念，特地搜了一下，总结如下：

## java线程转储

### 定义

java的线程转储可以被定义为JVM中在某一个给定的时刻运行的所有线程的快照。一个线程转储可能包含一个单独的线程或者多个线程。在多线程环境中，比如J2EE应用服务器，将会有许多线程和线程组。每一个线程都有它自己的调用堆栈，在一个给定时刻，表现为一个独立功能。线程转储将会提供JVM中所有线程的堆栈信息，对于特定的线程也会给出更多信息。

### 生成java线程转储

线程转储可以通过向JVM进程发送一个SIGQUIT信号来生成。有两种不同方式来向进程发送这个信号：

- Windows:

1.转向服务器的标准输出窗口并按下Control + Break组合键, 之后需要将线程堆栈复制到文件中；

- UNIX/ Linux：
首先查找到服务器的进程号(process id), 然后获取线程堆栈.
    1. ps –ef  | grep java
    2. kill -3 <pid>

JVM 自带的工具获取线程堆栈:

JDK自带命令行工具获取PID，再获取ThreadDump:

1. jps 或 ps –ef|grepjava (获取PID)
2. jstack [-l ]<pid> | tee -a jstack.log  (获取ThreadDump)

## 线程的状态

**线程的状态分析** 

要看懂线程转储信息，线程的状态是一个重要的指标，它会显示在线程 Stacktrace的头一行结尾的地方。那么线程常见的有哪些状态呢？线程在什么样的情况下会进入这种状态呢？我们能从中发现什么线索？ 

**Runnable**

该状态表示线程具备所有运行条件，在运行队列中准备操作系统的调度，或者正在运行。 

**Wait on condition**

该状态出现在线程等待某个条件的发生。具体是什么原因，可以结合 stacktrace来分析。最常见的情况是线程在等待网络的读写，比如当网络数据没有准备好读时，线程处于这种等待状态，而一旦有数据准备好读之后，线程会重新激活，读取并处理数据。在 Java引入 NewIO之前，对于每个网络连接，都有一个对应的线程来处理网络的读写操作，即使没有可读写的数据，线程仍然阻塞在读写操作上，这样有可能造成资源浪费，而且给操作系统的线程调度也带来压力。在 NewIO里采用了新的机制，编写的服务器程序的性能和可扩展性都得到提高。 

如果发现有大量的线程都在处在 Wait on condition，从线程 stack看， 正等待网络读写，这可能是一个网络瓶颈的征兆。因为网络阻塞导致线程无法执行。一种情况是网络非常忙，几 乎消耗了所有的带宽，仍然有大量数据等待网络读 写；另一种情况也可能是网络空闲，但由于路由等问题，导致包无法正常的到达。所以要结合系统的一些性能观察工具来综合分析，比如 netstat统计单位时间的发送包的数目，如果很明显超过了所在网络带宽的限制 ; 观察 cpu的利用率，如果系统态的 CPU时间，相对于用户态的 CPU时间比例较高；如果程序运行在 Solaris 10平台上，可以用 dtrace工具看系统调用的情况，如果观察到 read/write的系统调用的次数或者运行时间遥遥领先；这些都指向由于网络带宽所限导致的网络瓶颈。 

另外一种出现 Wait on condition的常见情况是该线程在 sleep，等待 sleep的时间到了时候，将被唤醒。 

** Waiting for monitor entry 和 in Object.wait()**

在多线程的 JAVA程序中，实现线程之间的同步，就要说说 Monitor。 Monitor是 Java中用以实现线程之间的互斥与协作的主要手段，它可以看成是对象或者 Class的锁。每一个对象都有，也仅有一个 monitor。下 面这个图，描述了线程和 Monitor之间关系，以 及线程的状态转换图： 
 
![java线程状态转换](http://7xngpc.com1.z0.glb.clouddn.com/vernjava线程状态转换.jpg)

从图中可以看出，**每个 Monitor在某个时刻，只能被一个线程拥有，该线程就是 “Active Thread”，而其它线程都是 “Waiting Thread”，分别在两个队列 “ Entry Set”和 “Wait Set”里面等候。在 “Entry Set”中等待的线程状态是 “Waiting for monitor entry”，而在 “Wait Set”中等待的线程状态是 “in Object.wait()”。** 

先看 “Entry Set”里面的线程。我们称被 synchronized保护起来的代码段为临界区。当一个线程申请进入临界区时，它就进入了 “Entry Set”队列。对应的 code就像： 

```java
synchronized(obj) { 
    ......... 
} 
```

这时有两种可能性： 
1）该 monitor不被其它线程拥有， Entry Set里面也没有其它等待线程。本线程即成为相应类或者对象的 Monitor的 Owner，执行临界区的代码 
2）该 monitor被其它线程拥有，本线程在 Entry Set队列中等待。 

在第一种情况下，线程将处于 “Runnable”的状态，而第二种情况下，线程 DUMP会显示处于 “waiting for monitor entry”。如下所示： 

```
"Thread-0" prio=10 tid=0x08222eb0 nid=0x9 waiting for monitor entry [0xf927b000 ..0xf927bdb8 
at testthread.WaitThread.run(WaitThread.java:39) 
- waiting to lock <0xef63bf08> (a java.lang.Object) 
- locked <0xef63beb8> (a java.util.ArrayList) 
at java.lang.Thread.run(Thread.java:595) 
```

临界区的设置，是为了保证其内部的代码执行的原子性和完整性。但是因为临界区在任何时间只允许线程串行通过，这 和我们多线程的程序的初衷是相反的。 如果在多线程的程序中，大量使用 synchronized，或者不适当的使用了它，会造成大量线程在临界区的入口等待，造成系统的性能大幅下降。如果在线程 DUMP中发现了这个情况，应该审查源码，改进程序。 

现在我们再来看现在线程为什么会进入 “Wait Set”。当线程获得了 Monitor，进入了临界区之后，如果发现线程继续运行的条件没有满足，它则调用对象（一般就是被 synchronized 的对象）的 wait() 方法，放弃了 Monitor，进入 “Wait Set”队列。只有当别的线程在该对象上调用了 notify() 或者 notifyAll() ， “ Wait Set”队列中线程才得到机会去竞争，但是只有一个线程获得对象的 Monitor，恢复到运行态。在 “Wait Set”中的线程， DUMP中表现为： in Object.wait()，类似于： 

```
"Thread-1" prio=10 tid=0x08223250 nid=0xa in Object.wait() [0xef47a000..0xef47aa38] 
at java.lang.Object.wait(Native Method) 
- waiting on <0xef63beb8> (a java.util.ArrayList) 
at java.lang.Object.wait(Object.java:474) 
at testthread.MyWaitThread.run(MyWaitThread.java:40) 
- locked <0xef63beb8> (a java.util.ArrayList) 
at java.lang.Thread.run(Thread.java:595) 
```
  
仔细观察上面的 DUMP信息，你会发现它有以下两行： 

- locked <0xef63beb8> (a java.util.ArrayList) 
- waiting on <0xef63beb8> (a java.util.ArrayList) 

这里需要解释一下，为什么先 lock了这个对象，然后又 waiting on同一个对象呢？让我们看看这个线程对应的代码： 
```java
synchronized(obj) { 
       ......... 
       obj.wait(); 
       ......... 
} 
```

线程的执行中，先用 synchronized 获得了这个对象的 Monitor（对应于 locked <0xef63beb8> ）。当执行到 obj.wait(), 线程即放弃了 Monitor的所有权，进入 “wait set”队列（对应于 waiting on <0xef63beb8> ）。 

## 分析一个Java线程

为了可以理解/分析线程转储，首先要理解线程转储的各个部分。让我们先拿一个简单的线程堆栈为例，并且去了解他的每个部分。

```
"ExecuteThread: '1' " daemon prio=5 tid=0x628330 nid=0xf runnable [0xe4881000..0xe48819e0]
at com.vantive.vanjavi.VanJavi.VanCreateForm(Native Method)
at com.vantive.vanjavi.VanMain.open(VanMain.java:53)
at jsp_servlet._so.__newServiceOrder.printSOSection( __newServiceOrder.java:3547)
at jsp_servlet._so.__newServiceOrder._jspService (__newServiceOrder.java:5652)
at weblogic.servlet.jsp.JspBase.service(JspBase.java:27)
at weblogic.servlet.internal.ServletStubImpl.invokeServlet (ServletStubImpl.java:265)
at weblogic.servlet.internal.ServletStubImpl.invokeServlet (ServletStubImpl.java:200)
at weblogic.servlet.internal.WebAppServletContext.invokeServlet(WebAppServletContext.java:2495)
at weblogic.servlet.internal.ServletRequestImpl.execute (ServletRequestImpl.java:2204)
at weblogic.kernel.ExecuteThread.execute (ExecuteThread.java:139)
at weblogic.kernel.ExecuteThread.run(ExecuteThread.java:120)

In the above Thread Dump, the interesting part to is the first line. The rest of the stuffis nothing more than a general stack trace. Lets analyze the first line here
```

各个项的含义为：

- Execute Thread : 1 说明了线程的名字
- daemon 表明这个线程是一个守护线程
- prio=5 线程的优先级 (默认是5)
- tid：Java的线程Id (这个线程在当前虚拟机中的唯一标识).
- nid 线程本地标识. 也就是Solaris中的LWP，线程在操作系统中的标识
- runnable 线程的状态 (参考上面的)
- [x..y] 当前运行的线程在堆中的地址范围

这个线程转储的剩余部分是调用堆栈。在这个例子中，这个线程（Execute Thread 1）是操作系统守护线程，当前正在执行一个本地方法vanCreateForm()。


## 参考文献

1.[http://www.cnblogs.com/huangfox/p/3442746.html](http://www.cnblogs.com/huangfox/p/3442746.html)

2.Javadoc关于ThreadDump的介绍[https://docs.oracle.com/cd/E13150_01/jrockit_jvm/jrockit/geninfo/diagnos/using_threaddumps.html](https://docs.oracle.com/cd/E13150_01/jrockit_jvm/jrockit/geninfo/diagnos/using_threaddumps.html)

3.[https://sites.google.com/site/threaddumps/java-thread-dumps-2](https://sites.google.com/site/threaddumps/java-thread-dumps-2)

4.[http://blog.csdn.net/rachel_luo/article/details/8920596](http://blog.csdn.net/rachel_luo/article/details/8920596)

5.[http://blog.csdn.net/wanyanxgf/article/details/6944987](http://blog.csdn.net/wanyanxgf/article/details/6944987)

**4、5的文章中有更详细的介绍和例子分析**