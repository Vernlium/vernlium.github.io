title: "《深入理解Java虚拟机》读书笔记"
date: 2015-10-11 20:41:32
tags: 
- JVM
- 读书笔记
categories: java
description: 
---

## Java内存区域

### 运行时数据区

这一块书上讲的内容很繁杂，而且很多，看到晕头转向的，结合[这篇博客](http://blog.csdn.net/luanlouis/article/details/40043991)的内容，总结了下图，我觉得还是比较清晰的。

<img src="http://7xngpc.com1.z0.glb.clouddn.com/vernJVM运行时数据区.jpg"/>


### HotSpot虚拟机对象

#### 1.对象的创建

new关键字创建一个对象时，虚拟机中对象的创建过程：

（1）虚拟机检查这个指令的参数是否能在常量池中定位到一个类的符号引用，并检查这个符号引用代表的类是否已经加载、解析和初始化过。如果没有，则必须先执行相应的类加载过程。

（2）虚拟机为新生对象分配内存，对象所需内存的大小在类加载完成后已经确定，为对象分配空间的任务等同于把一块确定大小的内存从Java堆中划分出来。

内存分配方式有两种：指针碰撞和空闲列表。内存分配方式由Java堆是否规整有关，而Java堆是否规整又由所采用的垃圾收集器是否带有压缩整理功能决定。

还有一个要考虑的问题是，多个线程同时新对象，如何解决并发问题。解决这个问题有两种方案：意识对分配内存空间的动作进行同步处理；另一种是把内存分配的动作按照线程划分在不同的空间之中急性，即每个线程在Java堆中预先分配一小块内存，称为本地线程分配缓冲（Thread Local Allocation Buffer,TLAB）。


（3）内存分配完成后，虚拟机需要将分配到的内存空间都初始化为零值。

（4）接下来，虚拟机对对象惊喜必要的设置，如这个对象是哪个类的实例、如何才能找到类的元数据信息、对象的哈希吗、对象的GC分代年龄等信息。

（5）上面的工作都完成之后，从虚拟机的视角看，一个新的对象已经产生了，但是从Java程序的视角来看，对象的创建才刚刚开始——<init>方法还没执行，所有的字段都还是零。所以，一般来说，执行new指令后会接着执行<init>方法，把对象按照程序员的意愿进行初始化，这样一个真正可用的对象才算完成产生出来。

#### 2.对象的内存布局

对象在内存中存储的布局可以分为三块区域：对象头（Header）、实例数据（Instance Data）和对齐填充（Padding）。

对象头包括两部分信息：第一部分用于存储对象自身的运行时数据，如哈希码、GC分代年龄、锁状态标志、线程持有的锁、偏向线程ID、偏向时间戳等，这部分数据的长度在32位和64位的虚拟机中分别为32bit和64bit，官方称它为“Mark Word”。

对象头的另一部分是类型支持，即对象指向它的类元数据的指针，虚拟机通过这个指针来确定这个对象是哪个类的实例。

数据实例部分是对象真正存储的有效信息，也是在程序代码中所定义的各种类型的字段内容。无论是从父类继承下来的，还是在子类中定义的，都需要记录起来。

第三部分对齐填充并不是必然存在的，也没有特别的含义，它仅仅起着占位符的作用。

#### 3.对象的访问定位

Java程序通过栈上的reference数据来操作对上的具体对象。由于reference类型在Java虚拟机规范中只规定了一个指向对象的引用，并没有定义这个引用应该通过何种方式去定位、访问堆中的对象的具体位置，所以对象访问方式也是取决于虚拟机实现而定的。目前主流的访问方式有使用句柄和直接指针两种。

- 如果直接使用句柄访问，java堆中将会划分出一块内存来作为句柄池，reference中存储的是对象的句柄地址，而句柄中包含了对象数据与类型数据各自的具体地址信息，如下图所示。

![通过句柄访问对象](http://7xngpc.com1.z0.glb.clouddn.com/vern通过句柄访问对象.jpg)

- 如果使用直接指针访问，那么java堆对象的布局中就必须考虑如何放置访问类型数据的相关信息，而reference中存储的直接就是对象地址，如下图所示。

![通过直接指针访问对象](http://7xngpc.com1.z0.glb.clouddn.com/vern通过直接指针访问对象.jpg)

这两种对象访问方式各有优势，使用句柄来访问的最大好处是reference中存储的是稳定的句柄地址，在对象被移动时只会改变句柄中的实例数据指针，而reference本身不需要修改。

使用直接指针访问方式的最大好处就是速度更快，它节省了一次指针定位的时间开销。HotSpot虚拟机使用的是直接指针访问的方式。句柄来访问的情况也十分常见。

## 垃圾收集器

垃圾收集（Garbage Collection,GC）需要完成3件事：

- **哪些内存需要回收？**
- **什么时候回收？**
- **如何回收？**

Java内存的程序计数器、虚拟机栈、本地方法栈3个区域随线程而生，随线程而灭；栈中的栈帧随着方法的进入和退出而有条不紊地执行着入栈和出栈操作。每一个栈帧中分配多少内存基本上是在类结构确定下来时就已知的，因此这几个区域的内存分配和回收都具备确定性，在这几个区域内就不需要过多考虑回收的问题，因为方法结束或者线程结束，内存自然就跟随着回收了。

而Java堆和方法区则不一样，一个接口中的多个实现类需要的内存可能不一样，一个方法中的多个实现类需要的内存可能不一样，一个方法中的多个分支需要的内存也可能不一样，只有在程序处于运行期间时才能知道会创建哪些对象，这部分内存的分配和回收是动态的，垃圾收集器所关注的是这部分的内存。

### 第一个问题：哪些内存需要回收

就是判断哪些对象实例还“活着”，哪些已经“死去”。

#### 可达性分析算法

Java中使用可达性分析（Reachability Analysis）来判定对象是否存活的。

通过一系列的称为“GC Roots”的对象作为起始点，从这些节点开始向下搜索，搜索所走过的路径称为引用链（Reference Chain）,当一个对象到GC Roots没有任何引用链相连时，则证明此对象是不可用的。

![可达性分析](http://7xngpc.com1.z0.glb.clouddn.com/vern可达性分析.jpg)

在Java语言中，可作为GC Roots的对象包括下面几种：

- 虚拟机栈（栈帧中的本地变量表）中引用的对象。
- 方法区中类静态属性引用的对象。
- 方法区中常量引用的对象。
- 本地方法栈中JNI（即一般说的Native方法）引用的对象。

Java对引用的概念进行了扩充，将引用分为强引用（Strong Reference）、软引用（Soft Reference）、弱引用（Weak Reference）、虚引用（Phantom Reference）4种，这4种引用强度依次逐渐减弱。

- 强引用就是指在程序代码之中普遍存在的，类似“Object obj = new Object()”这类的引用，只要强引用还存在，垃圾收集器永远不会回收掉被引用的对象。

- 软引用是用来描述一些还有用但并非必需的对象。对于软引用关联着的对象，在系统将要发生内存溢出异常之前，将会把这些对象列进回收范围之中进行第二次回收。如果这次回收还没有足够的内存，才会抛出内存溢出异常。在JDK 1.2之后，提供了SoftReference类来实现软引用。

- 弱引用也是用来描述非必需对象的，但是它的强度比软引用更弱一些，被弱引用关联的对象只能生存到下一次垃圾收集发生之前。当垃圾收集器工作时，无论当前内存是否足够，都会回收掉只被弱引用关联的对象。在JDK 1.2之后，提供了WeakReference类来实现弱引用。

- 虚引用也称为幽灵引用或者幻影引用，它是最弱的一种引用关系。一个对象是否有虚引用的存在，完全不会对其生存时间构成影响，也无法通过虚引用来取得一个对象实例。为一个对象设置虚引用关联的唯一目的就是能在这个对象被收集器回收时收到一个系统通知。在JDK 1.2之后，提供了PhantomReference类来实现虚引用。

### 第二个问题：什么时候回收？

#### 次收集（Minor GC）和全收集（Full GC）

当这三个分代的堆空间比较紧张或者没有足够的空间来为新到的请求分配的时候，垃圾回收机制就会起作用。有两种类型的垃圾回收方式：次收集和全收集。当新生代堆空间满了的时候，会触发次收集将还存活的对象移到年老代堆空间。当年老代堆空间满了的时候，会触发一个覆盖全范围的对象堆的全收集。

#### 次收集

- 当新生代堆空间紧张时会被触发
- 相对于全收集而言，收集间隔较短

#### 全收集

- 当老年代或者持久代堆空间满了，会触发全收集操作
- 可以使用System.gc()方法来显式的启动全收集
- 全收集一般根据堆大小的不同，需要的时间不尽相同，但一般会比较长。不过，如果全收集时间超过3到5秒钟，那就太长了

### 第三个问题：如何回收？

垃圾回收使用到一些垃圾收集算法。

#### 垃圾收集算法

##### 1.标记-清除算法

算法分为标记和清除两个阶段：首先标记出所有需要回收的对象，在标记完成后统一回收所有被标记的对象，它的标记过程就是使用可达性算法进行标记的。

主要缺点有两个：

- 效率问题，标记和清除两个过程的效率都不高
- 空间问题，标记清除之后会产生大量不连续的内存碎片

![标记清除算法](http://7xngpc.com1.z0.glb.clouddn.com/vern标记清除算法.jpg)

![标记清理算法](http://7xngpc.com1.z0.glb.clouddn.com/vernmark-clean.jpg)

##### 2.复制算法

复制算法：将可用内存按照容量分为大小相等的两块，每次只使用其中的一块。当这一块的内存用完了，就将还存活着的对象复制到另一块上面，然后把已使用过的内存空间一次清理掉。

内存分配时不用考虑内存碎片问题，只要一动堆顶指针，按顺序分配内存即可，实现简单，运行高效。代价是将内存缩小为原来的一半。

![复制算法](http://7xngpc.com1.z0.glb.clouddn.com/vern复制算法.jpg)

![复制算法](http://7xngpc.com1.z0.glb.clouddn.com/verncopy.jpg)

实际应用中将内存分为一块较大的Eden空间和两块较小的Survivor空间，每次使用Eden和其中的一块Survivor。当回收时，将Eden和Survivor中还存活着的对象一次性复制到另一块Survivor空间上，最后清理掉Eden和刚才用过的Survivor空间。

Hotspot虚拟机中默认的Eden和Survivor的大小比例是8:1.

##### 3.标记-整理算法

标记整理算法（Mark-Compact），标记过程仍然和“标记-清除”一样，但后续不走不是直接对可回收对象进行清理，而是让所有存活对象向一端移动，然后直接清理掉端边界以外的内存。

![标记整理算法](http://7xngpc.com1.z0.glb.clouddn.com/vern标记整理算法.jpg)

![标记整理算法](http://7xngpc.com1.z0.glb.clouddn.com/vernmark-trim.jpg)

##### 4.分代收集算法

根据对象存活周期的不同将内存分为几块。一般把Java堆分为新生代和老年代，根据各个年代的特点采用最合适的收集算法。在新生代中，每次垃圾收集时有大批对象死去，只有少量存活，可以选用复制算法。而老年代对象存活率高，使用标记清理或者标记整理算法。

### HotSpot虚拟机内存

下图是Sun HotSpot虚拟机的Heap区的分区，分为三个区：分别是Young Gereration新生代、Old Gerenation老年代、Permanent Generation持久区。

![jvm-memory-generation](http://7xngpc.com1.z0.glb.clouddn.com/vernjvm-memory-generation.png)

![新生代内存区域](http://7xngpc.com1.z0.glb.clouddn.com/vernYoung_generation_memory_areas.png)

### Collector的职责

- 分配内存。
- 保证有引用的内存不被释放。
- 回收没有指针引用的内存。

对象被引用称为活对象，对象没有被引用称为垃圾对象/垃圾/垃圾内存，找到垃圾对象并回收是Collector的一个主要工作，该过程称为GC。

### 好的Collector的特性

- 保证有引用的对象不被GC。
- 快速的回收内存垃圾。
- 在程序运行期间GC要高效，尽量少的影响程序运行。和大部分的计算机问题一样，这是一个关于空间，时间，效率平衡的问题。
- 避免内存碎片，内存碎片导致占用大量内存的大对象内存申请难以满足。
- 良好的扩展性，内存分配和GC在多核机器上不应该成为性能瓶颈。

### GC性能指标

- Throughput: 程序时间(不包含GC时间)/总时间。
- GC overhead: GC时间/总时间。
- Pause time: GC运行时程序挂起时间。
- Frequency of GC: GC频率。
- Footprint: Size度量，如堆大小。
- Promptness:对象变为垃圾到该垃圾被回收后内存可用的时间。

### HotSpot虚拟机垃圾收集器

下面是Sun HotSpot虚拟机1.6版本Update22包含的所有收集器。

![HotSpot虚拟机的垃圾收集器](http://7xngpc.com1.z0.glb.clouddn.com/vernhotspot_jvm_collector.jpg)

#### Serial Collecor

Serial收集器是单线程收集器，是分代收集器。

新生代：单线程复制收集算法

![Serial收集器新生代收集过程](http://7xngpc.com1.z0.glb.clouddn.com/vernserial_young_generation_collection_2.png)

老年代：单线程标记整理算法

![Serial收集器老年代收集](http://7xngpc.com1.z0.glb.clouddn.com/vernCompaction_of_the_old_generation.png)

Serial一般在单核的机器上使用，是Java 5非服务端JVM的默认收集器，参数**-XX:UseSerialGC**设置使用。

#### Parallel Collector

现在大部分的应用都是运行在多核的机器上，显然Serial收集器无法充分利用物理机的CPU资源，因此出现了Parallel收集器。Parallel收集器和Serial收集器的主要区别是新生代的收集，一个是单线程一个是多线程。可以从下图看到区别。

![Serial和Parallel收集器新生代收集过程对比](http://7xngpc.com1.z0.glb.clouddn.com/vernComparison_between_serial_and_parallel_young_generation_collection.png)

老年代的收集和Serial收集器是一样的。

Parallel收集器多在CPU的服务器上，是Java5 服务器端JVM的默认收集器。参数**-XX:+UseParallelGC**进行设置使用。

#### Parallel Compacting Collector

Parallel Compaction收集器出现在J2SE 5.0 update 6。和Parallel收集的主要区别在于老年代的收集，主要是为了解决老年代收集程序暂停时间过长的问题。

Parallel Compacting收集器分为三个阶段（每个区域在逻辑上是固定的）：

**①标记阶段（marking phase）**:并行标记所有代码能够直达的存活的对象。

**②总结阶段（summary phase）**:这一阶段是在区域进行而不是在对象上。一般情况下，区域靠左侧，存活对象的密度会高一些，在这一侧进行垃圾回收的花费会很高，代价大，并不值得。因此，总结阶段首先检查区域的对象密度，然后从左到右找到一个点：这个点的右侧区域垃圾收集的代价不大。这一点右侧使用标记整理算法进行回收。在收集过程中会计算并存储每一个收集区域的存活对象的新位置。这一阶段是单线程的。

**③整理阶段（compacting phase）**:使用上一阶段中的数据，使用copying算法进行整理，最终一侧是高密度的存活对象，另一侧为空。

此收集器多使用在多CPU的服务器上，并且程序对暂停时间要求较高。参数**-XX:+UseParallelOldGC**来使用它。还可以通过参数-XX:ParallelGCThreads=n来指定用于GC的线程数。

#### Concurrent Mark-Sweep(CMS) Collector

也称“low-latency collector”，为了解决老年代暂停时间过长的问题，并且真正实现并行收集（程序和GC并行执行）。

新生代：收集和Parallel Collector新生代收集方式一致。

老年代：GC和程序同时进行。

分为四个阶段：

**①初始标记(initial mark)**:暂停一会，找出所有活着对象的初始集合。

**②并行标记(concurrent marking)**：根据初始集合，标记出所有的存活对象，由于程序在运行，一部分存活对象无法标出。
此过程标记操作和程序同时执行。

**③重新标记(remark)**:程序暂停一会，多线程进行重新标记所有在②中没有被标记的存活对象。

**④并行清理concurrent sweep**：回收所有被标记的垃圾区域。和程序同时进行。

过程如下图所示。

![Serial和CMS收集器新生代收集过程对比](http://7xngpc.com1.z0.glb.clouddn.com/vernComparison_between_serial_and_CMS_old_generation_collection.png)

由于此收集器在remark阶段重新访问对象，因此开销有所增加。

此收集器的不足是，老年代收集采用标记清除算法，因此会产生很多不连续的内存碎片。

![CMS收集器老年代收集过程](http://7xngpc.com1.z0.glb.clouddn.com/vernCMS_sweeping_of_old_generation.png)

此收集器一般多用于对程序暂停时间要求更短的程序上，多由于web应用（实时性要求高）。参数-XX:+UseConcMarkSweepGC设置使用它。

#### G1收集器

下面将一步步的介绍G1收集器的收集过程。

##### 1、G1收集器的堆结构（G1 Heap Structure）

heap区被划分成很多固定大小的区域。区域的大小由JVM启动时选择。一般情况下，JVM会产生2000个左右的区域，每个区域的大小在1到32MB不等。

![G1 Heap Structure](http://7xngpc.com1.z0.glb.clouddn.com/verng1_heap_structure.png)

##### 2、G1收集器的堆分配（G1 Heap Allocation）

这些区域在逻辑上被影射成Eden、Survivor和老年代区。

![G1 Heap Allocation](http://7xngpc.com1.z0.glb.clouddn.com/verng1_heap_allocation.png)

活着的对象可以从一个区域拷贝或者移动到另一个区域。这样设计区分的划分可以在不停在其他线程的情况下分配内存空间。

##### 3、G1收集器的新生代（Young Generation in G1）

heap区被划分成大概2000个区域，最小的是1Mb，最大的是32Mb。

![Young Genertation in G1](http://7xngpc.com1.z0.glb.clouddn.com/vernyoung_generartion_in_g1.png)

##### 4、G1收集器的新生代GC(A Young GC in G1）

新生代的GC，活着的对象被复制或移动到survivor区，如果对象的年龄达到设置的阈值（比如设置的阈值是10，如果对象经过10GC后仍然存活，那么这个对象就达到了阈值）这些对象将被提升到老年代区中。

这个阶段stop the world（所有应用的线程都停止）。然后会计算eden区和survivor区的大小，下一次新生代GC的时候会用到这些信息。

![A Young GC in G1](http://7xngpc.com1.z0.glb.clouddn.com/verng1_young_gc.png)

##### 5、新生代收集结束（End of Yonug GC with G1）

存活的对象都被移到了survivor区或老年代区。这样新生代的收集就结束了。

![End of Yonug GC with G1](http://7xngpc.com1.z0.glb.clouddn.com/verng1_end_young_gc.png)

G1收集器的新生代GC可以总结如下：

- 堆区是一个单独的内存区，被分成了很多小区域。
- 新生代区由很多不连续的小区域组成，当需要的时候，重新分配新生代区的大小很容易。
- 新生代GC是stop the world的。
- 新生代GC是多现场并发进行的。
- 最后，活着的对象在survivor区或者老年代区。

##### 6、Initial Marking Phase

![Initial Marking Phase](http://7xngpc.com1.z0.glb.clouddn.com/verng1_initial_marking_phase.png)

##### 7、Concurrent Marking Phase

![Concurrent Marking Phase](http://7xngpc.com1.z0.glb.clouddn.com/verng1_concurrent_marking_phase.png)

##### 8、Remark Phase

![Remark Phase](http://7xngpc.com1.z0.glb.clouddn.com/verng1_remark_phase.png)

##### 9、Copying/Cleanup Phase

![Copying/Cleanup Phase](http://7xngpc.com1.z0.glb.clouddn.com/verng1_copying_phase.png)

##### 10、After Copying/Clean Phase

![After Copying/Clean Phase](http://7xngpc.com1.z0.glb.clouddn.com/verng1_after_copying_phase.png)


### 实例

    package com.idouba.jvm.demo;
    /**
     * Use shortest code demo jvm allocation, gc, and someting in gc.
     *
     * In details
     * 1) sizing of young generation (eden space，survivor space),old generation.
     * 2) allocation in eden space, gc in young generation,
     * 3) working with survivor space and with old generation.
     *
     */
    public class SimpleJVMArg {
        /**
         * @param args
         */
        public static void main(String[] args)
        {
            demo();
        }
        /**
         * VM arg：-verbose:gc -Xms200M -Xmx200M -Xmn100M -XX:+PrintGCDetails -XX:SurvivorRatio=8 -XX:MaxTenuringThreshold=1 -XX:+PrintTenuringDistribution
         *
         */
        @SuppressWarnings("unused")
        public static void demo() {
            final int tenMB = 10* 1024 * 1024;

            byte[] alloc1, alloc2, alloc3;
            
            alloc1 = new byte[tenMB / 5];
            alloc2 = new byte[5 * tenMB];
            alloc3 = new byte[4 * tenMB];
            alloc3 = null;
            alloc3 = new byte[6 * tenMB];
        }
    }

本实例来源于[最简单例子图解JVM内存分配和回收](http://ifeve.com/a-simple-example-demo-jvm-allocation-and-gc/)

设置虚拟机执行参数如下：

    -verbose:gc -Xms200M -Xmx200M -Xmn100M -XX:+PrintGCDetails -XX:SurvivorRatio=8 -XX:+PrintTenuringDistribution

其中，**-Xms200M -Xmx200M**设置Java堆大小为200M，不可扩展，
    **-Xmn100M**设置其中100M分配给新生代，则剩下的100M分配给老年代。
    **-XX:SurvivorRatio=8**设置了新生代中eden与survivor的空间比例是1：8。


<table>
   <tr>
      <td>GC 命令行选项 </td>
      <td>描述</td>
   </tr>
   <tr>
      <td>-Xms</td>
      <td>设置Java堆大小的初始值/最小值。例如：-Xms512m (请注意这里没有”=”).</td>
   </tr>
   <tr>
      <td>-Xmx</td>
      <td>设置Java堆大小的最大值</td>
   </tr>
   <tr>
      <td>-Xmn</td>
      <td>设置新生代对空间的初始值，最小值和最大值。请注意，年老代堆空间大小是依赖于新生代堆空间大小的</td>
   </tr>
   <tr>
      <td>-XX:PermSize=<n>[g|m|k]</td>
      <td>设置持久代堆空间的初始值和最小值</td>
   </tr>
</table>

执行结果如下：

    [GC (Allocation Failure) [DefNew
    Desired survivor size 5242880 bytes, new threshold 15 (max 15)
    - age   1:    2573248 bytes,    2573248 total
    : 56525K->2512K(92160K), 0.0524011 secs] 56525K->53712K(194560K), 0.0524798 secs] [Times: user=0.01 sys=0.03, real=0.05 secs] 
    [GC (Allocation Failure) [DefNew
    Desired survivor size 5242880 bytes, new threshold 15 (max 15)
    - age   2:    2572736 bytes,    2572736 total
    : 43472K->2512K(92160K), 0.0034130 secs] 94672K->53712K(194560K), 0.0034658 secs] [Times: user=0.00 sys=0.00, real=0.00 secs] 
    Heap
     def new generation   total 92160K, used 64772K [0x03c00000, 0x0a000000, 0x0a000000)
      eden space 81920K,  76% used [0x03c00000, 0x078cce58, 0x08c00000)
      from space 10240K,  24% used [0x08c00000, 0x08e741c0, 0x09600000)
      to   space 10240K,   0% used [0x09600000, 0x09600000, 0x0a000000)
     tenured generation   total 102400K, used 51200K [0x0a000000, 0x10400000, 0x10400000)
       the space 102400K,  50% used [0x0a000000, 0x0d200010, 0x0d200200, 0x10400000)
     Metaspace       used 97K, capacity 2242K, committed 2368K, reserved 4480K

### 分析


### 参考资料

- 《深入理解Java虚拟机：JVM高级特性与最佳实践》
- [http://www.oracle.com/technetwork/java/javase/memorymanagement-whitepaper-150215.pdf](http://www.oracle.com/technetwork/java/javase/memorymanagement-whitepaper-150215.pdf) java5官方文档，介绍了HotShot虚拟机的详细内容
- [http://www.oracle.com/technetwork/tutorials/tutorials-1876574.html#t1](http://www.oracle.com/technetwork/tutorials/tutorials-1876574.html#t1) java7关于G1收集器的详细介绍 
- [http://jbutton.iteye.com/blog/1569746](http://jbutton.iteye.com/blog/1569746)
- [http://www.importnew.com/1551.html](http://www.importnew.com/1551.html)
- [JVM实用参数（五）新生代垃圾回收](http://ifeve.com/useful-jvm-flags-part-5-young-generation-garbage-collection/)
- [JAVA的内存模型及结构](http://ifeve.com/under-the-hood-runtime-data-areas-javas-memory-model/)