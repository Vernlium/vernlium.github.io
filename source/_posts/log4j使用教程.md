title: "log4j使用教程"
date: 2015-06-28 15:54:26
tags: log4j
---
## 入门实例

下载log4j的jar包，目前的最新版本是1.2.17，[下载地址](http://logging.apache.org/log4j/1.2/)。

新建项目，新建lib文件夹，拷贝log4j-1.2.17.jar到lib下，导入到项目中。

新建class Log4JTest，代码如下：
    
    package com.anan.log;
    import org.apache.log4j.Logger;

    public class Log4JTest {
        private static final Logger log = Logger.getLogger(Log4JTest.class);

        public static void main(String[] args) {
            log.debug("this is debug message");
            log.error("this is error message");
            log.info("this is INFO message");
            log.warn("this  is warnning message");
        }
    }

src下新建文件log4j.properties,内容如下：

    log4j.rootCategory=DEBUG, stdout, file1

    log4j.appender.stdout=org.apache.log4j.ConsoleAppender
    log4j.appender.stdout.layout=org.apache.log4j.PatternLayout
    log4j.appender.stdout.layout.ConversionPattern=[QC] %p [%t] %C.%M(%L) - %m%n

    log4j.appender.file1=org.apache.log4j.DailyRollingFileAppender
    log4j.appender.file1.File=D:\\logs\\qc.log
    log4j.appender.file1.layout=org.apache.log4j.PatternLayout
    log4j.appender.file1.layout.ConversionPattern=%d-[TS] %p %t %c - %m%n

运行程序，控制台中得到如下输出：

    [QC] DEBUG [main] com.anan.log.Log4JTest.main(13) - this is debug message
    [QC] ERROR [main] com.anan.log.Log4JTest.main(14) - this is error message
    [QC] INFO [main] com.anan.log.Log4JTest.main(15) - this is INFO message
    [QC] WARN [main] com.anan.log.Log4JTest.main(16) - this  is warnning message

同时在D:\logs\文件夹下可以看到qc.log文件，其内容和上面输出的内容类似。

从上面的例子可以看出，log4j正常运行了，下面对配置文件和使用方法进行说明。

##  说明

### log4j.rootCategory=DEBUG, stdout

此句为将等级为DEBUG的日志信息输出到stdout这个目的地，stdout的定义在下面的代码，可以任意起名。等级可分为OFF、FATAL、ERROR、WARN、INFO、DEBUG、ALL.Log4j建议只使用四个级别，优先级从高到低分别是ERROR、WARN、INFO、DEBUG。

DEBUG：显示ERROR、WARN、INFO、DEBUG；

INFO： 显示ERROR、WARN、INFO；

WARN：显示ERROR、WARN；

ERROR：只显示ERROR；

可以通过修改DEBUG分别为另外三个值，看一下输出结果的区别。

### log4j.appender.stdout=org.apache.log4j.ConsoleAppender

此句为定义名为stdout的输出端是哪种类型，可以是

- org.apache.log4j.ConsoleAppender（控制台），
- org.apache.log4j.FileAppender（文件），
- org.apache.log4j.DailyRollingFileAppender（每天产生一个日志文件），
- org.apache.log4j.RollingFileAppender（文件大小到达指定尺寸的时候产生一个新的文件）
- org.apache.log4j.WriterAppender（将日志信息以流格式发送到任意指定的地方）

此处指定为输出端为控制台。

### log4j.appender.stdout.layout=org.apache.log4j.PatternLayout

此句为定义名为stdout的输出端的layout是哪种类型，可以是

- org.apache.log4j.HTMLLayout（以HTML表格形式布局），
- org.apache.log4j.PatternLayout（可以灵活地指定布局模式），
- org.apache.log4j.SimpleLayout（包含日志信息的级别和信息字符串），
- org.apache.log4j.TTCCLayout（包含日志产生的时间、线程、类别等等信息）

### log4j.appender.stdout.layout.ConversionPattern=[QC] %p [%t] %C.%M(%L) - %m%n

如果使用pattern布局就要指定的打印信息的具体格式ConversionPattern，打印参数如下：

%m 输出代码中指定的消息；

%M 输出打印该条日志的方法名；

%p 输出优先级，即DEBUG，INFO，WARN，ERROR，FATAL；

%r 输出自应用启动到输出该log信息耗费的毫秒数；

%c 输出所属的类目，通常就是所在类的全名；

%t 输出产生该日志事件的线程名；

%n 输出一个回车换行符，Windows平台为"rn”，Unix平台为"n”；

%d 输出日志时间点的日期或时间，默认格式为ISO8601，也可以在其后指定格式，比如：%d{ yyy-MM-dd HH:mm:ss,SSS}，输出类似：2002-10-18 22:10:28,921；

%l 输出日志事件的发生位置，及在代码中的行数；

[QC]是log信息的开头，可以为任意字符，一般为项目简称。

### log4j.appender.file1=org.apache.log4j.DailyRollingFileAppender

定义名为file1的输出端的类型为每天产生一个日志文件。

    log4j.appender.file1.File=D:\\logs\\qc.log

此句为定义名为file1的输出端的文件名为D:\\logs\\qc.log可以自行修改。


## 详解

### 定义配置文件

Log4j支持两种配置文件格式，一种是XML格式的文件，一种是Java特性文件log4j.properties（键=值）。项目中一般使用后一种方法。下面将介绍使用log4j.properties文件作为配置文件的方法:

①、配置根Logger
Logger 负责处理日志记录的大部分操作。

其语法为：

    log4j.rootLogger = [ level ] , appenderName, appenderName, …

其中，level 是日志记录的优先级，分为OFF、FATAL、ERROR、WARN、INFO、DEBUG、ALL或者自定义的级别。Log4j建议只使用四个级别，优先级从高到低分别是ERROR、WARN、INFO、DEBUG。通过在这里定义的级别，您可以控制到应用程序中相应级别的日志信息的开关。比如在这里定义了INFO级别，只有等于及高于这个级别的才进行处理，则应用程序中所有DEBUG级别的日志信息将不被打印出来。ALL:打印所有的日志，OFF：关闭所有的日志输出。 appenderName就是指定日志信息输出到哪个地方。可同时指定多个输出目的地。

②、配置日志信息输出目的地 Appender

Appender 负责控制日志记录操作的输出。

其语法为：

    log4j.appender.appenderName = fully.qualified.name.of.appender.class
    log4j.appender.appenderName.option1 = value1
    …
    log4j.appender.appenderName.optionN = valueN

这里的appenderName为在①里定义的，可任意起名。

其中，Log4j提供的appender有以下几种：
- org.apache.log4j.ConsoleAppender（控制台），
- org.apache.log4j.FileAppender（文件），
- org.apache.log4j.DailyRollingFileAppender（每天产生一个日志文件），
- org.apache.log4j.RollingFileAppender（文件大小到达指定尺寸的时候产生一个新的文件），可通过log4j.appender.R.MaxFileSize=100KB设置文件大小，还可通过log4j.appender.R.MaxBackupIndex=1设置为保存一个备份文件。
- org.apache.log4j.WriterAppender（将日志信息以流格式发送到任意指定的地方）

例如：log4j.appender.stdout=org.apache.log4j.ConsoleAppender
定义一个名为stdout的输出目的地，ConsoleAppender为控制台。

③、配置日志信息的格式（布局）Layout

Layout 负责格式化Appender的输出。

其语法为：

    log4j.appender.appenderName.layout = fully.qualified.name.of.layout.class
    log4j.appender.appenderName.layout.option1 = value1
    …
    log4j.appender.appenderName.layout.optionN = valueN

其中，Log4j提供的layout有以下几种：

- org.apache.log4j.HTMLLayout（以HTML表格形式布局），
- org.apache.log4j.PatternLayout（可以灵活地指定布局模式），
- org.apache.log4j.SimpleLayout（包含日志信息的级别和信息字符串），
- org.apache.log4j.TTCCLayout（包含日志产生的时间、线程、类别等等信息）

格式化日志

Log4J采用类似C语言中的printf函数的打印格式格式化日志信息，打印参数如下：
- %m 输出代码中指定的消息；
- %M 输出打印该条日志的方法名；
- %p 输出优先级，即DEBUG，INFO，WARN，ERROR，FATAL；
- %r 输出自应用启动到输出该log信息耗费的毫秒数；
- %c 输出所属的类目，通常就是所在类的全名；
- %t 输出产生该日志事件的线程名；
- %n 输出一个回车换行符，Windows平台为"rn”，Unix平台为"n”；
- %d 输出日志时间点的日期或时间，默认格式为ISO8601，也可以在其后指定格式，比如：%d{yyyy-MM-dd HH:mm:ss,SSS}，输出类似：2002-10-18 22:10:28,921；
- %l 输出日志事件的发生位置，及在代码中的行数。

### 运用在代码中

在需要输出日志信息的类中做如下的三个工作：

1、导入所有需的Logger类：

```java
import org.apache.log4j.Logger;
```

2、在自己的类中定义一个org.apache.log4j.Logger类的私有静态类成员：

```java 
private static final Logger log = Logger.getLogger(Log4JTest.class);
```

Logger.getLogger()方法的参数使用的是当前类的class。

3、使用Logger类的成员方法输出日志信息：

```java
log.debug("this is debug message");
log.error("this is error message");
log.info("this is INFO message");
log.warn("this  is warnning message");
```

## log4j类图

看看Log4J的类图：

![log4j类图](http://7xngpc.com1.z0.glb.clouddn.com/vernlog4j类图.png)
 
Logger - 日志写出器，供程序员输出日志信息 

Appender - 日志目的地，把格式化好的日志信息输出到指定的地方去 

ConsoleAppender - 目的地为控制台的Appender 

FileAppender - 目的地为文件的Appender 

RollingFileAppender - 目的地为大小受限的文件的Appender 

Layout - 日志格式化器，用来把程序员的logging request格式化成字符串 

PatternLayout - 用指定的pattern格式化logging request的Layout

总结一下，作为记录日志的工具，它至少应该包含如下几个组成部分(组件)： 

1. Logger 

记录器组件负责产生日志，并能够对日志信息进行分类筛选志应该被输出，什么样的日志应该被忽略。

2. Level 
    
    日志级别组件。 

3. Appender 
    
日志记录工具基本上通过 Appender 组件来输出到目ppender 实例就表示了一个输出的目的地。 
4. Layout 
    
Layout 组件负责格式化输出的日志信息，一个 Appenderayout。 

## 其他

我们在看其他项目的代码时有时候还会看到如下的Log类：

```java
import org.apache.commons.logging.Log;  
```

其引入的jar包为commons-logging.jar,这个jar包和log4j-xx.jar有什么关系呢？

我们应该知道，真正的记录日志的工具是 log4j 和 sun 公司提供的日志工具。而 commons-logging 把这两个(实际上，在 org.apache.commons.logging.impl 包下，commons-logging 仅仅为我们封装了 log4j 和 sun logger)记录日志的工具重新封装了一遍(Log4JLogger.java 和 Jdk14Logger.java)，可以认为 org.apache.commons.logging.Log 是个傀儡，它只是提供了对外的统一接口。因此我们只要能拿到 org.apache.commons.logging.Log，而不用关注到底使用的是 log4j 还是 sun logger。在项目中这样写： 

```java
// Run 是我们自己写的类，LogFactory 是一个专为提供 Log 的工厂(abstract class)  
private static final Log logger = LogFactory.getLog(Run.class);  
```

可是问题又来了，org.apache.commons.logging.Log 和 org.apache.log4j.Logger 这两个类，通过包名我们可以发现它们都是 apache 的项目，既然如下，为何要动如此大的动作搞两个东西(指的是 commons-logging 和 log4j)出来呢？事实上，在 sun 开发 logger 前，apache 项目已经开发了功能强大的 log4j 日志工具，并向 sun 推荐将其纳入到 jdk 的一部分，可是 sun 拒绝了 apache 的提议，sun 后来自己开发了一套记录日志的工具。可是现在的开源项目都使用的是 log4j，log4j 已经成了事实上的标准，但由于又有一部分开发者在使用 sun logger，因此 apache 才推出 commons-logging，使得我们不必关注我们正在使用何种日志工具。

在较新的代码中，我们还可以看到如下的Logger类：

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
private static final Logger log = LoggerFactory.getLogger(ClassName.class);
```

其jar包是slf4j-api-1.7.5.jar，这个jar包和上述的log4j-1.2.15.jar又有什么区别呢？

实际上，log4j 和 commons-logging 在 2007 年相继停止了更新，对于得到如此广泛应用的框架来说，这是个让人不安的事实。

SLF4J,简单日记门面(simple logging Facade for java),是为各种loging APIs提供一个简单统一的接口，从而使得最终用户能够在部署的时候配置自己希望的loging APIs实现。
准确的说，slf4j并不是一种具体的日志系统，而是一个用户日志系统的facade，允许用户在部署最终应用时方便的变更其日志系统。

SLF4J 作者就是 log4j 的作者 Ceki G&uuml;lc&uuml;，他宣称 SLF4J 比 log4j 更有效率，比 Apache Commons Logging (JCL) 简单、稳定。

可以说SLF4J和commons-logging类似，只不过SLF4J支持更多的logging APIs，细节方面有所改进。

在系统开发中，统一按照slf4j的API进行开发，在部署时，选择不同的日志系统包，即可自动转换到不同的日志系统上。比如：选择JDK自带的日志系统，则只需要将slf4j-api-1.5.10.jar和slf4j-jdk14-1.5.10.jar放置到classpath中即可，如果中途无法忍受JDK自带的日志系统了，想换成log4j的日志系统，仅需要用slf4j-log4j12-1.5.10.jar替换slf4j-jdk14-1.5.10.jar即可（ 当然也需要log4j的jar及 配置文件）。

slf4j门面原理

<img src="http://img.my.csdn.net/uploads/201304/07/1365322278_9117.jpg" />

目前，较新的代码中使用的都是slf4j，也推荐大家在以后的项目中使用slf4j。

## log4j在spring项目中的应用

。。。

### 【参考文献】
1. [http://www.iteye.com/topic/378077](http://www.iteye.com/topic/378077)
2. [百度百科log4j](http://baike.baidu.com/link?url=Wo0Jla0bb5jJeK5H4WeB2UCcYYBfzj1uJuG5DPndwQRl9R1oNkZwy6xRdTA8j8Zy47WCd8jzxMbG_fkZQDUfs_)
3. [http://blog.csdn.net/azheng270/article/details/2173430/](http://blog.csdn.net/azheng270/article/details/2173430/)
4. [http://sishuok.com/forum/blogPost/list/3740.html](http://sishuok.com/forum/blogPost/list/3740.html)
5. [http://www.cnblogs.com/eflylab/archive/2007/01/11/618001.html](http://www.cnblogs.com/eflylab/archive/2007/01/11/618001.html)