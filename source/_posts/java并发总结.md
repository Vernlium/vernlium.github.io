title: "java并发总结"
date: 2015-07-08 15:54:26
tags: java并发
categories: java
---
## 并发

### 1 基本的线程机制

#### 1.1 定义任务

- 实现Runnable接口并编写run()方法。

Thread.yield()方法的作用是：对线程调度器的一种建议,此时可以切换给其他任务执行。这完全是选择性的。

- 将Runnable对象交给一个Thread构造器。调用Thread对象的start()为该线程执行必需的初始化操作，然后调用Runnable的run()方法。start()调用后迅速返回。

#### 1.2 使用Executor

java.util.concurrent包中的执行器(Executor)可以管理Thread对象。Executor可以管理异步任务的执行，而无须显式的管理线程的生命周期。Executor是Java SE5/6中启动任务的优选方法。

```java
public class CachedThreadPool{
    public static void main(String[] args){
        ExecutorService exec = Executors.newCachedThreadPool();
        for(int i=0;i<5;i++)
            exec.execute(new LifeOff());
        exec.shutdown();
    }
}
```

调用shutdown()方法可以防止新任务被提交给这个Executor,当前线程将继续运行在shutdown()被调用之前提交的所有任务。

#### Executors的可以返回ExecutorService的几个方法比较：

**newFixedThreadPool**

public static ExecutorService newFixedThreadPool(int nThreads)

创建一个可重用固定线程数的线程池，以共享的无界队列方式来运行这些线程。在任意点，在大多数 nThreads 线程会处于处理任务的活动状态。如果在所有线程处于活动状态时提交附加任务，则在有可用线程之前，附加任务将在队列中等待。如果在关闭前的执行期间由于失败而导致任何线程终止，那么一个新线程将代替它执行后续的任务（如果需要）。在某个线程被显式地关闭之前，池中的线程将一直存在。 

参数：

nThreads - 池中的线程数 

返回：

新创建的线程池 

抛出： 

IllegalArgumentException - 如果 nThreads <= 0

---

**newFixedThreadPool**

public static ExecutorService newFixedThreadPool(int nThreads,ThreadFactory threadFactory)

创建一个可重用固定线程数的线程池，以共享的无界队列方式来运行这些线程，在需要时使用提供的 ThreadFactory 创建新线程。在任意点，在大多数 nThreads 线程会处于处理任务的活动状态。如果在所有线程处于活动状态时提交附加任务，则在有可用线程之前，附加任务将在队列中等待。如果在关闭前的执行期间由于失败而导致任何线程终止，那么一个新线程将代替它执行后续的任务（如果需要）。在某个线程被显式地关闭之前，池中的线程将一直存在。 

参数：

nThreads - 池中的线程数

threadFactory - 创建新线程时使用的工厂 

返回：

新创建的线程池 

抛出： 

NullPointerException - 如果 threadFactory 为 null 

IllegalArgumentException - 如果 nThreads <= 0

---

**newSingleThreadExecutor**

public static ExecutorService newSingleThreadExecutor()

创建一个使用单个 worker 线程的 Executor，以无界队列方式来运行该线程。（注意，如果因为在关闭前的执行期间出现失败而终止了此单个线程，那么如果需要，一个新线程将代替它执行后续的任务）。可保证顺序地执行各个任务，并且在任意给定的时间不会有多个线程是活动的。与其他等效的 newFixedThreadPool(1) 不同，可保证无需重新配置此方法所返回的执行程序即可使用其他的线程。 

返回：

新创建的单线程 Executor

---

**newSingleThreadExecutor**

public static ExecutorService newSingleThreadExecutor(ThreadFactory threadFactory)

创建一个使用单个 worker 线程的 Executor，以无界队列方式来运行该线程，并在需要时使用提供的 ThreadFactory 创建新线程。与其他等效的 newFixedThreadPool(1, threadFactory) 不同，可保证无需重新配置此方法所返回的执行程序即可使用其他的线程。 

参数：

threadFactory - 创建新线程时使用的工厂 

返回：

新创建的单线程 Executor 

抛出： 

NullPointerException - 如果 threadFactory 为 null

-------

**newCachedThreadPool**

public static ExecutorService newCachedThreadPool()

创建一个可根据需要创建新线程的线程池，但是在以前构造的线程可用时将重用它们。对于执行很多短期异步任务的程序而言，这些线程池通常可提高程序性能。调用 execute 将重用以前构造的线程（如果线程可用）。如果现有线程没有可用的，则创建一个新线程并添加到池中。终止并从缓存中移除那些已有 60 秒钟未被使用的线程。因此，长时间保持空闲的线程池不会使用任何资源。注意，可以使用 ThreadPoolExecutor 构造方法创建具有类似属性但细节不同（例如超时参数）的线程池。 

返回：

新创建的线程池

------
**newCachedThreadPool**

public static ExecutorService newCachedThreadPool(ThreadFactory threadFactory)

创建一个可根据需要创建新线程的线程池，但是在以前构造的线程可用时将重用它们，并在需要时使用提供的 ThreadFactory 创建新线程。 

参数：
threadFactory - 创建新线程时使用的工厂 

返回：

新创建的线程池 

抛出： 

NullPointerException - 如果 threadFactory 为 null

-----

CachedThreadPool在程序执行过程中通常会创建与所需数量相同的线程，然后在它回收旧线程时创建新线程，因此它是合理的Exector的首选。

#### 1.3 从任务中产生返回值

Runnable是执行工作的独立任务，但是它不返回任何值。如果你希望任务在完成时能够返回一个值，那么可以实现Callable接口而不是Runnable接口。

Callable是一种具有类型参数的泛型，它的类型参数表示的是从方法call()中的返回值，并且必须使用ExecutorService.submit()方法调用它。

submit()方法会产生Future对象，它用Callable返回结果的特定类型进行了参数化。可以使用isDone()方法来查询Future是否已经完成。当任务完成时，它具有一个结果，可以调用get()方法来获取该结果。也可以不用isDone()进行检查就直接调用get()，在这种情况下，get()将阻塞直到结果准备就绪。

#### 1.4 休眠

调用sleep()，这将使任务中止执行给定的时间。

#### 1.5 优先级

线程的**优先级**将该线程的重要性传递给了调度器，调度器将倾向于让优先级最高的线程先执行，但是是不确定的，这并不意味着优先级较低的线程将得不到执行。优先级较低的线程仅仅是执行的频率较低。

#### 1.6 让步

通过调用yield()方法暗示线程：你的工作做得的差不多了，可以让别的线程使用CPU了。当yield()调用时，也是在建议具有相同优先级的其他线程可以运行。

#### 1.7 后台（daemon）线程

后台（daemon）线程，是指在程序运行的时候在后台提供一种通用服务的线程，并且这种线程并不属于程序中不可或缺的部分。

在线程启动之前调用sertDaemon()方法，可以把线程设置为后台线程。

后台进程在不执行finally子句的情况下就会终止其run()方法。

#### 1.8 加入一个线程

一个线程可以在其他线程之上调用join()方法，其效果是等待一段时间直到第二个线程结束才继续执行。如果某个线程在另一个线程t上调用t.join()，此线程将被挂起，直到目标线程t结束才恢复（即t.isAlive()返回false）。

也可以在调用join()方法时带上一个超时参数，如果目标线程在这段时间到期时还没有结束的话，join()方法总能返回。

对join()方法的调用可以被中断，做法是在调用线程上调用interrupt()方法，这时需要用到try-catch子句。

#### 1.9 捕获异常

由于线程的本质特性，无法捕获从线程中逃逸的异常。一旦异常逃出任务的run()方法，它就会向外传播到控制台。

```
public class ExceptionThread implements Runnable{
    public void run(){
        throw new RuntimeException();
    }
    public static void main(String[] args){
        ExecutorService exec = Executors.newCachedThreadPool();
        exec.execute(new ExceptionThread());
    }
}
```

输出如下：

```
Exception in thread "pool-1-thread-1" java.lang.RuntimeException
··at com.anan.p21.ExceptionThread.run(ExceptionThread.java:8)
··at java.util.concurrent.ThreadPoolExecutor.runWorker(Unknown Source)
··at java.util.concurrent.ThreadPoolExecutor$Worker.run(Unknown Source)
··at java.lang.Thread.run(Unknown Source)
```

将main的主体放到try-catch语句块中没有作用：

```java
public calss NaiveExceptionHandling{
        public static void main(String[] args){
            try{
                ExecutorService exec = Executors.newCachedThreadPool();
            exec.execute(new ExceptionThread());
        }catch(RuntimeException e){
                System.out.println("Execption has been handled");
        }
    }
}
```

这将产生和前面一样的效果：未捕获异常。

为了解决这个问题，要修改Executor产生线程的方式。使用Thread.UncaughtExceptionHandler接口，它允许在每个Thread对象上附着一个异常处理器。Thread.UncaughtExceptionHandler.uncaughtException()会在线程因未捕获的异常而临近死亡时被调用。

### 2.共享受限资源

#### 2.1synchronized关键字

当任务要执行被synchronized关键字保护的代码片段时，它将检查锁是否可用，然后获取锁，执行代码，释放锁。

注意：synchronized关键字不属于方法特征签名的组成部分，所以可以在覆盖方法的时候加上去。

#### 2.2 使用显式的Lock对象

Locks对象必须被显式的创建、锁定和释放。

其通用用法为：

```java
private Lock lock = new ReentrantLock();
public int fun(){
    lock.lock();
    try{
        doing something...
        //the return must be in there
    }finally{
        lock.unlock();
    }
}
```

使用synchronized关键字时，需要的代码少，并且用户错误出现的可能性也会降低，因此通常只有在解决特殊问题时，才使用显式的Lock对象。比如，用synchronized关键字不能尝试着获取锁且最终获取锁会失败，或者尝试着获取锁一段时间然后放弃它，要实现这些必须使用Lock对象。

#### 2.3 原子性与易变性

原子操作是不能被线程调度机制中断的操作。

volatile关键字确保了应用中的可视性。如果将一个域声明为volatile的，那么只要对这个域产生了写操作，那么所有的读操作就都可以看到这个修改。即使使用了本地缓存，情况也一样，volatile域会立即被写入到主存中，而读取操作就发生在主存中。

非volatile域上的原子操作不必刷新到主存中去，因此其他读取该域的任务也不必看到这个新值。如果多个任务在同时访问某个域，那么这个域就应该是volatile的，否则这个域就应该只能经由同步来访问。同步也会导致向主存中刷新，因此如果一个域完全由synchronized方法或语句块来防护，那就不必将其设置为volatile的。

使用volatile而不是synchronized的唯一安全的情况是类中只有一个可变的域。

#### 2.4 原子类

Java SE5中引入了AtomicInteger、AtomicLong、AtomicReference等特殊的原子性变量类，可以实现如下的方法进行原子性条件更新操作：

    boolean compareAndSet(expectedValue,updateValue);

Atomic类只用在特殊情况下才使用，通常依赖于锁要更安全一些。

#### 2.5 临界区

使用synchronized关键字来建立临界区（critical section）
    
    synchronized(syncObject){
        //This code can be accessed by only one task at a time
    }

这也被称为同步控制块，在进入此代码前，必须得到syncObject对象的锁。如果其他线程已经得到了这个锁，那么就要等锁被释放后才能进入临界区。


#### 2.6 线程本地存储

线程本地存储是一种自动化机制，可以为使用相同变量的每个不同的线程都创建不同的存储。

ThreadLocal<T>类提供了线程局部 (thread-local) 变量。这些变量不同于它们的普通对应物，因为访问某个变量（通过其 get 或 set 方法）的每个线程都有自己的局部变量，它独立于变量的初始化副本。ThreadLocal 实例通常是类中的 private static 字段，它们希望将状态与某一个线程（例如，用户 ID 或事务 ID）相关联。 get()方法将返回与其线程相关联的对象的副本，而set()会将参数插入到为其线程存储的对象中，并返回存储的原有对象。

### 3 终结任务

ExecutorService.awaitTermination()等待每个任务结束，如果所有的任务在超时时间到达之前结束，则返回true,否则返回false，表示不是所有的任务都已经结束了。

#### 在阻塞时终结

线程的五种状态：

- 1.**新建(new)**:当线程被创建时，它只会短暂的处于这种状态。此时它已经分配了必需的系统资源，并执行了初始化。此刻线程已经有资格获取CPU时间了，之后调度器将把这个线程转变为可运行状态或阻塞状态。 
- 2.**就绪(Runnable)**:在这种状态下，只要调度器把时间片分配给线程，线程就可以运行了。在任意时刻，线程可以运行也可以不运行，只要调度器能分配时间片给线程，它就可以运行；这不同于死亡和阻塞状态。
- 3.**运行(Running)**。线程调度程序将处于就绪状态的线程设置为当前线程，此时线程就进入了运行状态，开始运行run函数当中的代码。 
- 4.**阻塞(Blocked)**:线程能够运行，但有某个条件阻止它的运行。当线程处于阻塞状态时，调度器将忽略线程，不会分配给线程任何CPU时间。直到线程重新进入了就绪状态，它才有可能执行操作。
- 5.**死亡(Dead)**:处于死亡或终止状态的线程将不再是可调度的，并且再不会得到CPU时间，它的任务已结算，或不再是可运行的。任务死亡的通常方式是从run()方法返回，但是任务的线程还可以被中断。

其状态转换关系如下图：

![java线程状态转换](http://7xngpc.com1.z0.glb.clouddn.com/vernjava线程状态转换.jpg)


#### 进入阻塞状态

一个任务进入阻塞状态，可能有如下原因：

- 1.通过调用sleep(milliseconds)使任务进入休眠状态，在这种情况下，任务在指定的时间内不会运行。
- 2.通过调用wait()使线程挂起。直到线程得到了notify()或notifyAll()消息，线程才会进入就绪状态。
- 3.任务在等待某个输入/输出完成。
- 4.任务在试图在某个对象上调用其同步控制方法，但是对象锁不可用，因为另一个任务已经获取了这个锁。

早版本的java中使用suspend()和resume()来阻塞和唤醒线程，现代java中这些方法被废弃（因为可能导致死锁）。stop()方法也被废止了，因为它不释放线程获得的锁，并且如果线程处于不一致状态，其他任务可以在这种状态下浏览并修改它们。

#### 中断

Thread类包含interrupt()方法，可以终止被阻塞的任务，这个方法将设置线程的中断状态。如果一个线程已经被阻塞，或者试图执行一个阻塞操作，那么设置这个线程的中断状态将抛出InterruptedException。当抛出该异常或者该任务调用Thread.interrupted()时，中断状态将被复位。Thread.interrupted()提供了离开run()循环而不抛出异常的第二种方式。

新的concurrent类库在避免对Thread对象的直接操作，转而尽量通过Executor来执行所有操作。如果在Executor上调用shutdownNow(),那么它将发送一个interrupt()调用给它启动的所有线程。

有时希望只中断某个单一任务。如果使用Executor，通过调用submit()而不是executor()来启动任务，就可以持有该任务的上下文。submit()将返回一个泛型Future<?>,其中有一个未修饰的参数。持有这个Future的关键是在于可以在其上调用cancel()，并因此可以使用它来中断某个特定任务。如果将true传递给cancel(),那么它就会拥有在该线程上调用interrupt()以停止这个线程的权限。cancel()是一种中断由Executor启动的单个线程的方式。

Future<V>接口的方法：

boolean cancel(boolean mayInterruptIfRunning)

试图取消对此任务的执行。如果任务已完成、或已取消，或者由于某些其他原因而无法取消，则此尝试将失败。当调用 cancel 时，如果调用成功，而此任务尚未启动，则此任务将永不运行。如果任务已经启动，则 mayInterruptIfRunning参数确定是否应该以试图停止任务的方式来中断执行此任务的线程。

此方法返回后，对 isDone() 的后续调用将始终返回 true。如果此方法返回 true，则对 isCancelled() 的后续调用将始终返回 true。 

参数：

mayInterruptIfRunning - 如果应该中断执行此任务的线程，则为 true；否则允许正在运行的任务运行完成 

返回：

如果无法取消任务，则返回 false，这通常是由于它已经正常完成；否则返回 true

-------------------------------------------------------------------------

boolean isCancelled()如果在任务正常完成前将其取消，则返回 true。 

返回：

如果任务完成前将其取消，则返回 true

--------------------------------------------------------------------------

boolean isDone()

如果任务已完成，则返回 true。可能由于正常终止、异常或取消而完成，在所有这些情况中，此方法都将返回 true。 

返回：

如果任务已完成，则返回 true

---------------------------------------------------------------------------

V get() throws InterruptedException,ExecutionException

如有必要，等待计算完成，然后获取其结果。 

返回：

计算的结果 

抛出： 

CancellationException - 如果计算被取消 
ExecutionException - 如果计算抛出异常 
InterruptedException - 如果当前的线程在等待时被中断

sleep()是可中断的阻塞，而I/O和在synchronized块上等待是不可中断的阻塞。无论是I/O还是尝试调用synchronized方法，都不需要任何InterruptedException处理器。
也就是说，可以中断对sleep()的调用，而不能中断正在试图获取synchronized锁或者试图执行I/O操作的线程。关闭任务在其上发生阻塞的底层资源，可以解除I/O阻塞。

被阻塞的nio通道会自动的响应中断。

一个任务应该能够调用在同一个对象中的其他的synchronized方法，而这个任务已经持有锁了。

#### 检查中断

-------------------------------------------------------------------------

<div>
<font color=red><strong>
当我们调用t.interrput()的时候，线程t的中断状态(interrupted status)  会被置位,即设置为true。我们可以通过Thread.currentThread().interrupted()    来检查这个布尔型的中断状态。<br/>

在Core Java中有这样一句话："没有任何语言方面的需求要求一个被中断的程序应该终止。中断一个线程只是为了引起该线程的注意，被中断线程可以决定如何应对中断 "。</br>

这说明我们调用t.interrupt()之后，并不能中断线程，只有线程内部检查了中断状态并作出反应才行。

但是当t被阻塞的时候，比如被Object.wait, Thread.join和Thread.sleep三种方法之一阻塞时，调用它的interrput()方法，会产生一个InterruptedException异常。
</strong>
</font>
</div>

-------------------------------------------------------------------------

检查中断的常用写法：

```java
//Interrupted的经典使用代码  
public void run(){  
    try{  
        ....  
         while(!Thread.currentThread().interrupted()&& more work to do){  
            // do more work;  
        }  
    }catch(InterruptedException e){  
        // thread was interrupted during sleep or wait  
    }  finally{  
               // cleanup, if required  
    }  
}  
```

注意interrupted()方法和isInterruputed()方法的区别：

**interrupted()会改变线程的中断状态，isInterrupted()不会改变线程的中断状态。**

public static boolean interrupted()

测试当前线程是否已经中断。线程的中断状态 由该方法清除。换句话说，如果连续两次调用该方法，则第二次调用将返回 false（在第一次调用已清除了其中断状态之后，且第二次调用检验完中断状态前，当前线程再次中断的情况除外）。 
线程中断被忽略，因为在中断时不处于活动状态的线程将由此返回 false 的方法反映出来。 


返回：

如果当前线程已经中断，则返回 true；否则返回 false。

------------------------------------------------------------------------------

public boolean isInterrupted()

测试线程是否已经中断。线程的中断状态 不受该方法的影响。 
线程中断被忽略，因为在中断时不处于活动状态的线程将由此返回 false 的方法反映出来。 

返回：

如果该线程已经中断，则返回 true；否则返回 false。

### 4.线程之间的协作

第二节解决的是线程同步问题，即使用锁来同步两个任务的行为，从而使一个任务不会干涉另一个任务的资源。

下面要解决线程之间的协作，已使得多个任务可以一起工作去解决某个问题。

#### 4.1 wain()和notifyAll()

wait()会在等待外部世界产生变化的时候将任务挂起，并且只有在notify()或notifyAll()发生时，这个任务才会被唤醒并去检查所产生的变化。

调用sleep()的时候锁并没有被释放，调用yield()也是如此。当一个任务在方法里遇到了对wait()的调用时，线程的执行被挂起，对象的锁被释放。

还有一点要注意的是：wait()是Object对象的方法，而sleep()是Thread对象的方法。

有两种形式的wait().

第一个版本是以毫秒数为参数，含义与sleep()方法里的参数相同，都是“在此期间暂停”。但是与sleep()不同的是，对于wait()而言：

（1）在wait()期间对象锁是释放的

（2）可以通过notify()、notifyAll()，或者时间到期，从wait()中恢复执行。

第二种，也是更常用的形式的wait()不接受任何参数，这种wait()将无限等待下去，直到线程接收到notify()或notifyAll()消息。

wait()、notifyAll()、notify()这些方法都是基类Object的一部分。可以把wait()放进如何同步控制方法里，而不用考虑这个类是继承自Thread还是实现了Runnable接口。而且，只能在同步控制方法或同步控制块里调用wait、notify、notifyAll（sleep()不用操作锁，所以可以在非同步控制方法里使用。）

如果在非同步控制方法里调用这些方法，程序能编译通过，但是运行的时候，将得到IllegalmonitorStateException异常，并有一些消息，比如“当前线程不是拥有者”。意思是，调用wait、notify、notifyAll的任务在调用这些方法前必须拥有对象的锁。

#### 4.2 notify()与notifyAll()

使用notify()而不是notifyAll()是一种优化。使用notify()时，在众多等待同一个锁的任务中只有一个会被唤醒。

当notifyAll()因某个特定锁而被调用时，只有等待这个锁的任务才会被唤醒。

#### 4.3 生产者与消费者

使用互斥并允许任务挂起的基本类是Condition，可以通过在Condition上调用await()来挂起一个任务。当外部条件发生变化，意味着某个任务应该继续执行时，可以通过调用signal()来通知这个任务，从而唤醒一个任务，或者调用signalAll()来唤醒所有在这个Condition上被其自身挂起的任务（与使用notifyAll()相比，signalAll()是更安全的方式）。

使用方式为：

```java
class Car{
    private Lock lock = new ReentrantLock();
    private Condition condition = lock.newCondition();
    private boolean waxOn = false;
    public void waxed(){
        lock.lock();
        try{
            waxOn = true;
            condition.signalAll();
        }finally{
            lock.unlock();
        }
    }
    public void waitForWaxing() throws InterruptedException{
        lock.lock();
        try{
            while(waxOn == false)
                condition.await();
        }finally{
            lock.unlock();
        }
    }
}
```

Lock和Condition对象只有在更加困难的多线程问题中才是必需的。

#### 4.4 生产者-消费者队列

同步队列在任何时刻只能允许一个任务插入或移除元素。在java.util.concurrent.BlockingQueue接口中提供了这个队列，这个接口有大量的标准实现。LinkedBlockingQueue是无界队列，ArrayBlockingQueue具有固定尺寸，在它被阻塞之前，向其中放置有限数量的元素。

如果消费者试图从队列中获取对象，而该对象为空，那么这些队列还可以挂起消费者任务，并且当有更多的元素可用时恢复消费者任务。

以下是基于典型的生产者-使用者场景的一个用例。注意，BlockingQueue 可以安全地与多个生产者和多个使用者一起使用。 

```java
class Producer implements Runnable {
        private final BlockingQueue queue;
    Producer(BlockingQueue q) { queue = q; }
    public void run() {
            try {
                while(true) { queue.put(produce()); }
        } catch (InterruptedException ex) { ... handle ...}
    }
    Object produce() { ... }
}                                   

class Consumer implements Runnable {
        private final BlockingQueue queue;
    Consumer(BlockingQueue q) { queue = q; }
    public void run() {
            try {
                while(true) { consume(queue.take()); }
        } catch (InterruptedException ex) { ... handle ...}
    }
    void consume(Object x) { ... }
}

class Setup {
        void main() {
            BlockingQueue q = new SomeQueueImplementation();
        Producer p = new Producer(q);
        Consumer c1 = new Consumer(q);
        Consumer c2 = new Consumer(q);
        new Thread(p).start();
        new Thread(c1).start();
        new Thread(c2).start();
    }
}
```


#### 4.5 任务间使用管道进行输入/输出

通过输入/输出在线程间进行通信通常很有用。以“管道”的形式对线程间的输入/输出提供了支持。PipedWriter(允许任务向管道写)和PipedReader(允许不同任务从同一个管道中读取)。管道基本上是一个阻塞队列。

PipedReader与普通I/O之间最重要的差异——PipedReader是可中断的。

### 5.死锁

**死锁**：某个任务在等待另一个任务，而后者又在等待别的任务，这样一直下去，直到这个链条上的任务又在等待第一个任务释放锁。这些任务之间响度等待，没有哪一个线程能继续。

产生死锁的四个必要条件：

（1） 互斥条件：一个资源每次只能被一个进程使用。

（2） 请求保持条件：一个进程因请求资源而阻塞时，对已获得的资源保持不放。

（3） 不可抢占条件:进程已获得的资源，在末使用完之前，不能强行剥夺。

（4） 循环等待条件:若干进程之间形成一种头尾相接的循环等待资源关系。

### 6.新类库中的构件

#### 6.1 CountDownLatch

一个同步辅助类，在完成一组正在其他线程中执行的操作之前，它允许一个或多个线程一直等待。 

用给定的计数 初始化 CountDownLatch。由于调用了 countDown() 方法，所以在当前计数到达零之前，await 方法会一直受阻塞。之后，会释放所有等待的线程，await 的所有后续调用都将立即返回。这种现象只出现一次——计数无法被重置。如果需要重置计数，请考虑使用 CyclicBarrier。 

CountDownLatch 是一个通用同步工具，它有很多用途。

- 将计数 1 初始化的 CountDownLatch 用作一个简单的开/关锁存器，或入口：在通过调用 countDown() 的线程打开入口前，所有调用 await 的线程都一直在入口处等待。
- 用 N 初始化的 CountDownLatch 可以使一个线程在 N 个线程完成某项操作之前一直等待，或者使其在某项操作完成 N 次之前一直等待。 

CountDownLatch 的一个有用特性是，它不要求调用 countDown 方法的线程等到计数到达零时才继续，而在所有线程都能通过之前，它只是阻止任何线程继续通过一个 await。 

示例用法： 下面给出了两个类，其中一组 worker 线程使用了两个倒计数锁存器： 

第一个类是一个启动信号，在 driver 为继续执行 worker 做好准备之前，它会阻止所有的 worker 继续执行。 
第二个类是一个完成信号，它允许 driver 在完成所有 worker 之前一直等待。 
    
```java
class Driver { // ...
    void main() throws InterruptedException {
        CountDownLatch startSignal = new CountDownLatch(1);
        CountDownLatch doneSignal = new CountDownLatch(N);

        for (int i = 0; i < N; ++i) // create and start threads
            new Thread(new Worker(startSignal, doneSignal)).start();

        doSomethingElse();            // don't let run yet
        startSignal.countDown();      // let all threads proceed
        doSomethingElse();
        doneSignal.await();           // wait for all to finish
    }
}

class Worker implements Runnable {
    private final CountDownLatch startSignal;
    private final CountDownLatch doneSignal;
    Worker(CountDownLatch startSignal, CountDownLatch doneSignal) {
        this.startSignal = startSignal;
        this.doneSignal = doneSignal;
    }
    public void run() {
        try {
            startSignal.await();
            doWork();
            doneSignal.countDown();
        catch (InterruptedException ex) {} // return;
    }

    void doWork() { ... }
}
```

 另一种典型用法是，将一个问题分成 N 个部分，用执行每个部分并让锁存器倒计数的 Runnable 来描述每个部分，然后将所有 Runnable 加入到 Executor 队列。当所有的子部分完成后，协调线程就能够通过 await。（当线程必须用这种方法反复倒计数时，可改为使用 CyclicBarrier。） 

```java
class Driver2 { // ...
    void main() throws InterruptedException {
        CountDownLatch doneSignal = new CountDownLatch(N);
        Executor e = ...

        for (int i = 0; i < N; ++i) // create and start threads
            e.execute(new WorkerRunnable(doneSignal, i));

        doneSignal.await();           // wait for all to finish
    }
}

class WorkerRunnable implements Runnable {
    private final CountDownLatch doneSignal;
    private final int i;
    WorkerRunnable(CountDownLatch doneSignal, int i) {
        this.doneSignal = doneSignal;
        this.i = i;
    }
    public void run() {
        try {
            doWork(i);
            doneSignal.countDown();
        } catch (InterruptedException ex) {} // return;
    }

    void doWork() { ... }
}
```

内存一致性效果：线程中调用 countDown() 之前的操作 happen-before 紧跟在从另一个线程中对应 await() 成功返回的操作。 

#### 6.2 CyclicBarrier

一个同步辅助类，它允许一组线程互相等待，直到到达某个公共屏障点 (common barrier point)。在涉及一组固定大小的线程的程序中，这些线程必须不时地互相等待，此时 CyclicBarrier 很有用。因为该 barrier 在释放等待线程后可以重用，所以称它为循环 的 barrier。 

CyclicBarrier 支持一个可选的 Runnable 命令，在一组线程中的最后一个线程到达之后（但在释放所有线程之前），该命令只在每个屏障点运行一次。若在继续所有参与线程之前更新共享状态，此屏障操作 很有用。 

示例用法：下面是一个在并行分解设计中使用 barrier 的例子： 

```java
class Solver {
    final int N;
    final float[][] data;
    final CyclicBarrier barrier;
            
    class Worker implements Runnable {
        int myRow;
        Worker(int row) { myRow = row; }
        public void run() {
            while (!done()) {
                processRow(myRow);

                try {
                    barrier.await(); 
                } catch (InterruptedException ex) { 
                    return; 
                } catch (BrokenBarrierException ex) { 
                    return; 
                }
            }
        }
    }

    public Solver(float[][] matrix) {
        data = matrix;
        N = matrix.length;
        barrier = new CyclicBarrier(N, new Runnable() {
                                        public void run() { 
                                        mergeRows(...); 
                                        }
                                    });
        for (int i = 0; i < N; ++i) 
            new Thread(new Worker(i)).start();

        waitUntilDone();
    }
}
```

在这个例子中，每个 worker 线程处理矩阵的一行，在处理完所有的行之前，该线程将一直在屏障处等待。处理完所有的行之后，将执行所提供的 Runnable 屏障操作，并合并这些行。如果合并者确定已经找到了一个解决方案，那么 done() 将返回 true，所有的 worker 线程都将终止。 

如果屏障操作在执行时不依赖于正挂起的线程，则线程组中的任何线程在获得释放时都能执行该操作。为方便此操作，每次调用 await() 都将返回能到达屏障处的线程的索引。然后，您可以选择哪个线程应该执行屏障操作，例如： 

```java
if (barrier.await() == 0) {
    // log the completion of this iteration
}
```

对于失败的同步尝试，CyclicBarrier 使用了一种要么全部要么全不 (all-or-none) 的破坏模式：如果因为中断、失败或者超时等原因，导致线程过早地离开了屏障点，那么在该屏障点等待的其他所有线程也将通过 BrokenBarrierException（如果它们几乎同时被中断，则用 InterruptedException）以反常的方式离开。 


CountDownLatch和CyclicBarrier的区别

javadoc里面的描述是这样的。

> CountDownLatch: A synchronization aid that allows one or more threads to wait until a set of operations being performed in other threads completes.


> CyclicBarrier : A synchronization aid that allows a set of threads to all wait for each other to reach a common barrier point.

CountDownLatch : 一个线程(或者多个)， 等待另外N个线程完成某个事情之后才能执行。   
CyclicBarrier        : N个线程相互等待，任何一个线程完成之前，所有的线程都必须等待。

这样应该就清楚一点了，对于CountDownLatch来说，重点是那个“一个线程”, 是它在等待， 而另外那 N 的线程在把“某个事情”做完之后可以继续等待，可以终止。

而对于CyclicBarrier来说，重点是那N个线程，他们之间任何一个没有完成，所有的线程都必须等待。

有一篇博客介绍CountDownLatch和CyclicBarrier的实现原理的，有时间再研究一下

[CountDownLatch & CyclicBarrier源码实现解析](http://blog.csdn.net/pun_c/article/details/37658841?utm_source=tuicool)

#### 6.3 DelayQueue

DelayQueue<E extends Delayed>

这是一个无界阻塞队列（BlockingQueue），用于放置实现了Delayed接口的 对象，只有在延迟期满时才能从中提取元素。该队列的头部 是延迟期满后保存时间最长的对象。如果延迟都还没有期满，则队列没有头部，并且 poll 将返回 null。当一个元素的 getDelay(TimeUnit.NANOSECONDS) 方法返回一个小于等于 0 的值时，将发生到期。即使无法使用 take 或 poll 移除未到期的元素，也不会将这些元素作为正常元素对待。例如，size 方法同时返回到期和未到期元素的计数。此队列不允许使用 null 元素。 

#### 6.4 PriorityBlockingQueue

#### 6.5 ScheduledExecutor

public class ScheduledThreadPoolExecutor
extends ThreadPoolExecutor
implements ScheduledExecutorService

是一个ThreadPoolExecutor，它可另行安排在给定的延迟后运行命令，或者定期执行命令。需要多个辅助线程时，或者要求 ThreadPoolExecutor 具有额外的灵活性或功能时，此类要优于 Timer。 

一旦启用已延迟的任务就执行它，但是有关何时启用，启用后何时执行则没有任何实时保证。按照提交的先进先出 (FIFO) 顺序来启用那些被安排在同一执行时间的任务。 

虽然此类继承自 ThreadPoolExecutor，但是几个继承的调整方法对此类并无作用。特别是，因为它作为一个使用 corePoolSize 线程和一个无界队列的固定大小的池，所以调整 maximumPoolSize 没有什么效果。 

#### 6.6 Semaphore

public class Semaphoreextends Objectimplements Serializable一个计数信号量。从概念上讲，信号量维护了一个许可集。如有必要，在许可可用前会阻塞每一个 acquire()，然后再获取该许可。每个 release() 添加一个许可，从而可能释放一个正在阻塞的获取者。但是，不使用实际的许可对象，Semaphore 只对可用许可的号码进行计数，并采取相应的行动。 

Semaphore 通常用于限制可以访问某些资源（物理或逻辑的）的线程数目。例如，下面的类使用信号量控制对内容池的访问： 

```java
class Pool {
    private static final int MAX_AVAILABLE = 100;
    private final Semaphore available = new Semaphore(MAX_AVAILABLE, true); 
    public Object getItem() throws InterruptedException {
        available.acquire();
        return getNextAvailableItem();
    }   
    public void putItem(Object x) {
        if (markAsUnused(x))
            available.release();
    }

  // Not a particularly efficient data structure; just for demo

    protected Object[] items = ... whatever kinds of items being managed
    protected boolean[] used = new boolean[MAX_AVAILABLE];

    protected synchronized Object getNextAvailableItem() {
        for (int i = 0; i < MAX_AVAILABLE; ++i) {
            if (!used[i]) {
                used[i] = true;
                return items[i];
            }
        }
        return null; // not reached
    }

    protected synchronized boolean markAsUnused(Object item) {
        for (int i = 0; i < MAX_AVAILABLE; ++i) {
            if (item == items[i]) {
                if (used[i]) {
                used[i] = false;
                return true;
            } else
                return false;
            }
        }
        return false;
    }
}
```

获得一项前，每个线程必须从信号量获取许可，从而保证可以使用该项。该线程结束后，将项返回到池中并将许可返回到该信号量，从而允许其他线程获取该项。注意，调用 acquire() 时无法保持同步锁，因为这会阻止将项返回到池中。信号量封装所需的同步，以限制对池的访问，这同维持该池本身一致性所需的同步是分开的。 

将信号量初始化为 1，使得它在使用时最多只有一个可用的许可，从而可用作一个相互排斥的锁。这通常也称为二进制信号量，因为它只能有两种状态：一个可用的许可，或零个可用的许可。按此方式使用时，二进制信号量具有某种属性（与很多 Lock 实现不同），即可以由线程释放“锁”，而不是由所有者（因为信号量没有所有权的概念）。在某些专门的上下文（如死锁恢复）中这会很有用。 

此类的构造方法可选地接受一个公平 参数。当设置为 false 时，此类不对线程获取许可的顺序做任何保证。特别地，闯入 是允许的，也就是说可以在已经等待的线程前为调用 acquire() 的线程分配一个许可，从逻辑上说，就是新线程将自己置于等待线程队列的头部。当公平设置为 true 时，信号量保证对于任何调用获取方法的线程而言，都按照处理它们调用这些方法的顺序（即先进先出；FIFO）来选择线程、获得许可。注意，FIFO 排序必然应用到这些方法内的指定内部执行点。所以，可能某个线程先于另一个线程调用了 acquire，但是却在该线程之后到达排序点，并且从方法返回时也类似。还要注意，非同步的 tryAcquire 方法不使用公平设置，而是使用任意可用的许可。 

通常，应该将用于控制资源访问的信号量初始化为公平的，以确保所有线程都可访问资源。为其他的种类的同步控制使用信号量时，非公平排序的吞吐量优势通常要比公平考虑更为重要。 

此类还提供便捷的方法来同时 acquire 和释放多个许可。小心，在未将公平设置为 true 时使用这些方法会增加不确定延期的风险。 

#### 6.6 Exchanger

public class Exchanger<V> extends Object

可以在对中对元素进行配对和交换的线程的同步点。每个线程将条目上的某个方法呈现给 exchange 方法，与伙伴线程进行匹配，并且在返回时接收其伙伴的对象。Exchanger 可能被视为 SynchronousQueue 的双向形式。Exchanger 可能在应用程序（比如遗传算法和管道设计）中很有用。 

用法示例：以下是重点介绍的一个类，该类使用 Exchanger 在线程间交换缓冲区，因此，在需要时，填充缓冲区的线程获取一个新腾空的缓冲区，并将填满的缓冲区传递给腾空缓冲区的线程。 

```java
class FillAndEmpty {
    Exchanger<DataBuffer> exchanger = new Exchanger<DataBuffer>();
    DataBuffer initialEmptyBuffer = ... a made-up type
    DataBuffer initialFullBuffer = ...

    class FillingLoop implements Runnable {
        public void run() {
            DataBuffer currentBuffer = initialEmptyBuffer;
            try {
                while (currentBuffer != null) {
                    addToBuffer(currentBuffer);
                    if (currentBuffer.isFull())
                        currentBuffer = exchanger.exchange(currentBuffer);
                }
            } catch (InterruptedException ex) { ... handle ... }
        }
    }

    class EmptyingLoop implements Runnable {
        public void run() {
            DataBuffer currentBuffer = initialFullBuffer;
            try {
                while (currentBuffer != null) {
                takeFromBuffer(currentBuffer);
                if (currentBuffer.isEmpty())
                    currentBuffer = exchanger.exchange(currentBuffer);
                }
            } catch (InterruptedException ex) { ... handle ...}
        }
    }

    void start() {
        new Thread(new FillingLoop()).start();
        new Thread(new EmptyingLoop()).start();
    }
}
```

### 7、性能调优

#### 7.1 比较各种互斥技术

使用Lock通常会比使用synchronized要高效很多，而且synchronized的开销看起来变化范围很大，而Lock相对比较一致。

这是否意味着不使用synchronized呢？有两个因素要考虑：

首先，实际中，被互斥的部分比较大，因此在这些方法体中花费的时间的百分比可能会明显大于进入和退出互斥的开销，这样也就湮灭了提高互斥速度带来的好处。当在进行性能调优时，可以尝试各种不同的方法并观察它们造成的影响。

其次，synchronized关键词的代码的可读性要比lock-try/finally-unlock方法的代码的可读性高。因此，以synchronized关键字入手，只有在性能调优时才替换为Lock对象这种做法，比较具有实际意义。

Atomic对象只有在非常简单的情况下才有用，这些情况包括只有一个要被修改的Atomic对象，并且这个对象对立于其他所有的对象。更安全的做法是：以更加传统的互斥方式入手，只有在性能方面的需求能够明确指示时，才替换Atomic。

#### 7.2 免锁容器

免锁容器的通用策略是：对容器的修改可以与读取操作同时发生，只要读取者只能看到完成修改的结果即可。修改时在容器数据结构的某个部分的一个单独的副本上执行的，并且这个副本在修改过程中是不可视的。只有当修改完成时，被修改的结构才会自动的与主数据结构进行交换，之后读取者就可以看到这个修改了。





