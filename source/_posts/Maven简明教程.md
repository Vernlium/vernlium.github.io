title: "Maven简明教程"
date: 2015-10-29 22:09:54
tags: Maven
categories: Maven 
description: 这是一个Maven简明教程，包括Maven入门和深入了解
---

## 1 Maven是什么

Maven是一个项目管理工具，它包含了一个项目对象模型 (Project Object Model)，一组标准集合，一个项目生命周期(Project Lifecycle)，一个依赖管理系统(Dependency Management System)，和用来运行定义在生命周期阶段(phase)中插件(plugin)目标(goal)的逻辑。

Maven的目标(官网上的)：

- Making the build process easy
- Providing a uniform build system
- Providing quality project information
- Providing guidelines for best practices development
- Allowing transparent migration to new features

这些概念都是虚的，还是要通过实践来体验。

## 2 Maven安装

### 2.1 安装JDK

JDK自行安装去吧，如果连JDK都没装还是不要学习maven了。

### 2.2 安装maven

去[Maven官网](maven.apache.org)下载maven，可以下载最新版的[mavne-3.3.3](http://www.us.apache.org/dist/maven/maven-3/3.3.3/binaries/apache-maven-3.3.3-bin.zip),需要注意的是maven3.3.3需要JDK1.7及以上版本。

下载之后将压缩包解压，放到要安装的目录下，比如C:\Program Files\apache-maven-3.3.3

### 2.3 配置环境变量

在环境变量中添加MAVEN_HOME变量，值为C:\Program Files\apache-maven-3.3.3，然后在Path中添加%MAVEN_HOME%/bin。

### 2.4 验证

在控制台中输入mvn，如果打印出如下信息，说明mvn安装成功了。

![验证是否安装成功](http://7xngpc.com1.z0.glb.clouddn.com/vern验证是否安装成功.png)

## 3 Maven入门实例

maven安装成功后，我们先来通过一个例子来认识他。

新建文件夹MavenProject，在下面按照如下目录创建maven项目HelloMaven：

```
    HelloMaven
       │  pom.xml
       │
       ├─src
          ├─main
          │  └─java
          │      └─com
          │          └─anan
          │              └─hellomaven
          │                      HelloMaven.java
          │
          └─test
              └─java
                  └─com
                      └─anan
                          └─hellomaven
                                  HelloMavenTest.java
```

HelloMaven.java文件的内容为：

```java
package com.anan.hellomaven;
    public class HelloMaven{
        public String sayHello(String name){
            return "Hello "+name;
    }
} 
```

HelloMavenTest.java文件的内容为：

```java
package com.anan.hellomaven;

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class HelloMavenTest{
    @Test
    public void testSayHello(){
        assertEquals("Hello maven",new HelloMaven().sayHello("maven"));
    }
}
```

pom.xml文件的内容为：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <!--
      modelVersion指定当前pom模型的版本，对于Maven2及Maven3来说，它只能是4.0.0
  -->
  <modelVersion>4.0.0</modelVersion>
  <!--
      groupId，artifactId和version这三个元素定义了一个项目基本的坐标,
      在Maven的世界，任何的jar、pom或者war都是以基于这些基本的坐标进行区分
  -->
  <!--
    groupId定义了项目属于哪个组，一般是公司网站反写+项目名
  -->
  <groupId>com.anan.hellomaven</groupId>
  <!--
    artifactId定义了当前Maven项目在组中唯一的ID，一般是项目名+模块名
  -->
  <artifactId>hellomaven-module1</artifactId>
  <!--
    项目当前版本，SNAPSHOT意为快照，说明该项目还处于开发中，是不稳定的版本
  -->
  <version>1.0-SNAPSHOT</version>
 
  <name>HelloMaven Project</name>
  <!--依赖的包-->
  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.7</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
```

上面的文件写完后保存，在项目根目录下按住Shift键右击，选择在此处打开命令窗口，输入命令：
    
`mvn compile`

执行命令可以看到如下信息，提示项目编译成功。

    [INFO] Compiling 1 source file to F:\MavenTest\HelloMaven\target\classes
    [INFO] -----------------------------------------------------------------
    [INFO] BUILD SUCCESS
    [INFO] -----------------------------------------------------------------
    [INFO] Total time: 3.304 s
    [INFO] Finished at: 2015-10-29T22:40:51+08:00
    [INFO] Final Memory: 11M/27M
    [INFO] ------------------------------------------------------------------

可以在项目根目录下看到多了一个target文件夹，里面的内容是..\HelloMaven\target\classes\com\anan\hellomaven\HelloMaven.class，也就是HelloMaven.java的编译文件。

继续执行命令：

`mvn test`

可以看到如下信息，提示项目测试成功。

    -------------------------------------------------------
     T E S T S
    -------------------------------------------------------
    Running com.anan.hellomaven.HelloMavenTest
    Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.102 sec
    Results :
    Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
    [INFO] -----------------------------------------------------------------
    [INFO] BUILD SUCCESS
    [INFO] -----------------------------------------------------------------
    [INFO] Total time: 5.315 s
    [INFO] Finished at: 2015-10-30T07:13:39+08:00
    [INFO] Final Memory: 12M/28M
    [INFO] -----------------------------------------------------------------

这时我们可以在..\HelloMaven\target\surefire-reports目录下看到com.anan.hellomaven.HelloMavenTest.txt文件，这个是测试的结果。

最后执行命令：

` mvn package`

支持命令看到如下信息，提示打包成功：

    -------------------------------------------------------
     T E S T S
    -------------------------------------------------------
    Running com.anan.hellomaven.HelloMavenTest
    Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.097 sec
    Results :
    Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
    [INFO]
    [INFO] --- maven-jar-plugin:2.4:jar (default-jar) @ hellomaven ---
    [INFO] Building jar: F:\MavenTest\HelloMaven\target\hellomaven-1.0-SNAPSHOT.jar
    [INFO] ------------------------------------------------------------------
    [INFO] BUILD SUCCESS
    [INFO] ------------------------------------------------------------------
    [INFO] Total time: 4.190 s
    [INFO] Finished at: 2015-10-30T07:16:38+08:00
    [INFO] Final Memory: 7M/19M
    [INFO] -------------------------------------------------------------------

可以在..\HelloMaven\target\目录下看到hellomaven-1.0-SNAPSHOT.jar文件，这个就是项目打包生成的jar包。

## 4 Eclipse整合Maven

在实际应用中，我们maven都是和IDE一起使用的，用起来很方便。现在eclipse的j2ee版本（4.0版本以后，Juno、Kepler、Luna、Mars这几个版本，之前的版本要自己安装插件才能使用）和IntelliJ IDEA这两个最流行的IDE都集成了maven，所以无需我们自己安装maven插件。（较老版本的安装maven插件也很简单，需要的自己网上搜一下，有很多）

我们这里使用eclipse这个IDE（现在IntelliJ IDEA是趋势啊，之前用来一段时间，确实好用，可是公司里面用eclipse，还是乖乖回到eclipse吧）。

需要注意的是，eclipse自带的maven是使用自带的maven版本，因此要设置为我们刚才安装的maven。打开window-preference-maven-Installations，添加刚才安装的maven路径，如下图所示。

![eclipse修改maven路径](http://7xngpc.com1.z0.glb.clouddn.com/vernchange-maven-path.png)

下面开始创建maven项目。

### 4.1 创建单个项目

选择File-New-Other，选择Maven Project，点击下一步；

![create-signle-project1](http://7xngpc.com1.z0.glb.clouddn.com/vernmaven-createpro-1.png)

Select project name and location这一步直接跳过，下一步；Select an Archetype，选择maven-archetype-quickstart，点击下一步。这里的archetype就相当于一个项目模板，这个模板中规定了源代码、测试源代码、编译目标等的路径，方便我们使用；

![create-signle-project2](http://7xngpc.com1.z0.glb.clouddn.com/vernmaven-createpro-2.png)

输入项目的groupId和artifactId，Package是项目中的包名，可以选择使用默认的，也可以重新输入；

![create-signle-project3](http://7xngpc.com1.z0.glb.clouddn.com/vernmaven-createpro-3.png)

创建完成后，我们可以在eclipse中看到项目app-module1，其结构如下：

![maven项目结构](http://7xngpc.com1.z0.glb.clouddn.com/vernmaven工程结构.png)

### 4.2 创建多项目

一般情况下，我们会将一个项目，分为多个子项目，这样便于项目开发和分工，这些子项目都是一个独立的maven工程，我们把这些子项目放在一个父项目下便于管理，下面我们就创建一个多项目聚合的maven项目。

大部分同学应该都做过基于ssh的web项目，我们现在就是用maven来创建一个这样的项目。

我们将这个项目分为如下四部分：

```
    ssh-parent
      |-- pom.xml
      |
      |-- ssh-domain（实体类，jar）
      |
      |-- ssh-dao（数据持久化子项目,jar）
      |
      |-- ssh-service（服务类子项目,jar）
      |
      |-- ssh-web（web前端子项目,war）
```

父项目也是maven项目，所以也有pom.xml文件，其他四个子项目中，前三个是普通maven项目，是用maven-archetype-quick来创建，ssh-web是web项目，所以要使用maven-archetype-webapp来创建。

先创建父项目，其方式和上面的单项目创建流程一致，groupId为com.anan.ssh，artifactId为ssh-parent。创建完成后，需要修改pom.xml中的packaging值为pom。

```
<groupId>com.anan.ssh</groupId>
<artifactId>ssh-parent</artifactId>
<version>0.0.1-SNAPSHOT</version>
<packaging>pom</packaging>
```

因为作为父项目，其packaging必须为pom。

下面创建四个子项目，需要注意的是，子项目创建时是创建Maven Module。

File-New-Other，选择Maven Module；

![mutl-create-pro-1](http://7xngpc.com1.z0.glb.clouddn.com/vernmult-pro-create-1.png)

填写Module Name并选择Parent Project，这里Module Name就是artifactId，groupId由父项目继承而来。

![mutl-create-pro-2](http://7xngpc.com1.z0.glb.clouddn.com/vernmult-pro-create-2.png)

ssh-dao、ssh-service和ssh-domain三个项目创建时使用的Archetype为maven-archetype-quickstart。

![mutl-create-pro-3](http://7xngpc.com1.z0.glb.clouddn.com/vernmult-pro-create-3.png)

ssh-web创建时使用的Archetype为maven-archetype-web。

![mutl-create-pro-4](http://7xngpc.com1.z0.glb.clouddn.com/vernmult-pro-create-4.png)

五个项目创建完成后，分别查看项目的pom.xml文件。在ssh-parent的pom中我们可以看到如下配置：

```
<modules>
  <module>ssh-domain</module>
  <module>ssh-dao</module>
  <module>ssh-service</module>
  <module>ssh-web</module>
</modules>
```

这段配置表示ssh-parent这个项目由四个子项目“聚合”而成。

四个子项目的pom中都可以看到如下配置：

```
<parent>
  <groupId>com.anan.ssh</groupId>
  <artifactId>ssh-parent</artifactId>
  <version>0.0.1-SNAPSHOT</version>
</parent>
```

这段配置表示项目的父项目为ssh-parent，其可以从父项目中继承一些配置。这些配置是创建项目后自动加入的。单个项目中则不会有类似的配置。

多个项目创建完成后，我们来进行多项目的编译和测试。

选中ssh-parent项目，右键run as -> maven build

![vernmult-pro-create-5](http://7xngpc.com1.z0.glb.clouddn.com/vernmult-pro-create-6.png)

在goal中输入，clean package：

![vernmult-pro-create-6](http://7xngpc.com1.z0.glb.clouddn.com/vernmult-pro-create-5.png)

可以在控制台看到如下输出：

    [INFO] Reactor Summary:
    [INFO] 
    [INFO] ssh-parent ..................................... SUCCESS [  0.005 s]
    [INFO] ssh-domain ..................................... SUCCESS [  6.280 s]
    [INFO] ssh-dao ........................................ SUCCESS [  0.939 s]
    [INFO] ssh-service .................................... SUCCESS [  1.034 s]
    [INFO] ssh-web Maven Webapp ........................... SUCCESS [  0.776 s]
    [INFO] -------------------------------------------------------------------
    [INFO] BUILD SUCCESS
    [INFO] -------------------------------------------------------------------
    [INFO] Total time: 9.305 s
    [INFO] Finished at: 2015-11-22T12:33:11+08:00
    [INFO] Final Memory: 17M/42M
    [INFO] -------------------------------------------------------------------

我们可以看到四个子项目都进行了编译，多个项目一起编译方便了很多。

在这个项目中，我们提到了一些概念，比如：聚合、继承等，会在后文中进行详述。

## 5 pom.xml详解

通过上面的例子，我们可能会有一个疑问：我们的项目并没有指定项目的源代码的位置，maven编译时是如何找到这些文件的位置的呢？这就要从pom.xml文件讲起。

就像Java中的类隐式继承Object一样，对于pom.xml来说，它隐式继承超级POM。针对Maven3来说，该超级POM位于maven-model-builder-VERSION.jar包中（该jar包位于maven根目录/lib下)解压该jar包，可以在maven-model-builder-VERSION/org/apache/maven/model目录中找到pom-4.0.0.xml，即超级POM。

该超级POM的配置成为了Maven提倡的约定。该文件的具体内容如下：

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <repositories>
    <repository>
      <id>central</id>
      <name>Central Repository</name>
      <url>https://repo.maven.apache.org/maven2</url>
      <layout>default</layout>
      <snapshots>
        <enabled>false</enabled>
      </snapshots>
    </repository>
  </repositories>

  <pluginRepositories>
    <pluginRepository>
      <id>central</id>
      <name>Central Repository</name>
      <url>https://repo.maven.apache.org/maven2</url>
      <layout>default</layout>
      <snapshots>
        <enabled>false</enabled>
      </snapshots>
      <releases>
        <updatePolicy>never</updatePolicy>
      </releases>
    </pluginRepository>
  </pluginRepositories>

  <build>
    <directory>${project.basedir}/target</directory>
    <outputDirectory>${project.build.directory}/classes</outputDirectory>
    <finalName>${project.artifactId}-${project.version}</finalName>
    <testOutputDirectory>${project.build.directory}/test-classes<   /testOutputDirectory>
    <sourceDirectory>${project.basedir}/src/main/java</sourceDirectory>
    <scriptSourceDirectory>${project.basedir}/src/main/scripts<   /scriptSourceDirectory>
    <testSourceDirectory>${project.basedir}/src/test/java<    /testSourceDirectory>
    <resources>
      <resource>
        <directory>${project.basedir}/src/main/resources</directory>
      </resource>
    </resources>
    <testResources>
      <testResource>
        <directory>${project.basedir}/src/test/resources</directory>
      </testResource>
    </testResources>
    <pluginManagement>
      <!-- NOTE: These plugins will be removed from future versions of the    super POM -->
      <!-- They are kept for the moment as they are very unlikely to    conflict wilifecycle mappings (MNG-4453) -->
      <plugins>
        <plugin>
          <artifactId>maven-antrun-plugin</artifactId>
          <version>1.3</version>
        </plugin>
        <plugin>
          <artifactId>maven-assembly-plugin</artifactId>
          <version>2.2-beta-5</version>
        </plugin>
        <plugin>
          <artifactId>maven-dependency-plugin</artifactId>
          <version>2.8</version>
        </plugin>
        <plugin>
          <artifactId>maven-release-plugin</artifactId>
          <version>2.3.2</version>
        </plugin>
      </plugins>
    </pluginManagement>
  </build>
  <reporting>
    <outputDirectory>${project.build.directory}/site</outputDirectory>
  </reporting>
  <profiles>
    <!-- NOTE: The release profile will be removed from future versions of    the supPOM -->
    <profile>
      <id>release-profile</id>
      <activation>
        <property>
          <name>performRelease</name>
          <value>true</value>
        </property>
      </activation>
      <build>
        <plugins>
          <plugin>
            <inherited>true</inherited>
            <artifactId>maven-source-plugin</artifactId>
            <executions>
              <execution>
                <id>attach-sources</id>
                <goals>
                  <goal>jar</goal>
                </goals>
              </execution>
            </executions>
          </plugin>
          <plugin>
            <inherited>true</inherited>
            <artifactId>maven-javadoc-plugin</artifactId>
            <executions>
              <execution>
                <id>attach-javadocs</id>
                <goals>
                  <goal>jar</goal>
                </goals>
              </execution>
            </executions>
          </plugin>
          <plugin>
            <inherited>true</inherited>
            <artifactId>maven-deploy-plugin</artifactId>
            <configuration>
              <updateReleaseInfo>true</updateReleaseInfo>
            </configuration>
          </plugin>
        </plugins>
      </build>

    </profile>
  </profiles>
</project>
```

我们可以在这个pom中看到build标签下有这样的配置：

    <sourceDirectory>${project.basedir}/src/main/java</sourceDirectory>

这个就是用来配置源代码的位置，maven进行编译时就会到此目录下查找源代码，并将编译的文件放在target目录下。directory配置了编译文件目录的位置。

  <directory>${project.basedir}/target</directory>

在pom文件中，我们需要关注如下一些标签：

### 5.1 坐标

Maven坐标为各种构件引入了秩序，任何一个构件都必须明确定义自己的坐标，而一组Maven坐标是通过一些元素定义的，它们是groupId,artifactId,version,packaging,class-sifer。下面是一组坐标定义：
  
```
<groupId>com.mycompany.app</groupId>  
<artifactId>my-app</artifactId>  
<packaging>jar</packaging>  
<version>0.0.1-SNAPSHOT</version>  
```

下面讲解一下各个坐标元素：
 
**groupId** ：定义当前Maven项目隶属的实际项目。首先，Maven项目和实际项目不一定是一对一的关系。比如SpringFrameWork这一实际项目，其对应的Maven项目会有很多，如spring-core,spring-context等。这是由于Maven中模块的概念，因此，一个实际项目往往会被划分成很多模块。其次，groupId不应该对应项目隶属的组织或公司。原因很简单，一个组织下会有很多实际项目，如果groupId只定义到组织级别，而后面我们会看到，artifactId只能定义Maven项目（模块），那么实际项目这个层次将难以定义。最后，groupId的表示方式与Java包名的表达方式类似，通常与域名反向一一对应。
 
**artifactId** : 该元素定义当前实际项目中的一个Maven项目（模块），推荐的做法是使用实际项目名称作为artifactId的前缀。比如上例中的my-app。
 
**version** : 该元素定义Maven项目当前的版本
 
**packaging** ：定义Maven项目打包的方式，首先，打包方式通常与所生成构件的文件扩展名对应，如上例中的packaging为jar,最终的文件名为my-app-0.0.1-SNAPSHOT.jar。也可以打包成war, ear等。当不定义packaging的时候，Maven 会使用默认值jar
 
**classifier** : 该元素用来帮助定义构建输出的一些附件。附属构件与主构件对应，如上例中的主构件为my-app-0.0.1-SNAPSHOT.jar,该项目可能还会通过一些插件生成如my-app-0.0.1-SNAPSHOT-javadoc.jar,my-app-0.0.1-SNAPSHOT-sources.jar, 这样附属构件也就拥有了自己唯一的坐标。这一元素我们并不经常用到，因此可以忽略。

### 5.2 依赖dependencies

依赖会包含基本的groupId, artifactId,version等元素，根元素project下的dependencies可以包含一个或者多个dependency元素，以声明一个或者多个依赖。

下面详细讲解每个依赖可以包含的元素：
 
groupId,artifactId和version：依赖的基本坐标，对于任何一个依赖来说，基本坐标是最重要的，Maven根据坐标才能找到需要的依赖
 
type: 依赖的类型，对应于项目坐标定义的packaging。大部分情况下，该元素不必声明，其默认值是jar
 
- scope: 依赖的范围。
- optional: 标记依赖是否可选。
- exclusions: 用来排除传递性依赖，下面会进行详解。
 
大部分依赖声明只包含基本坐标。

#### 依赖范围

Maven在编译主代码的时候需要使用一套classpath,在编译和执行测试的时候会使用另一套classpath,实际运行项目的时候，又会使用一套classpath。依赖范围就是用来控制依赖与这三种classpath(编译classpath、测试classpath、运行classpath)的关系，Maven有以下几种依赖范围：
 
compile: 编译依赖范围。如果没有指定，就会默认使用该依赖范围。使用此依赖范围的Maven依赖，对于编译、测试、运行三种classpath都有效。
 
test: 测试依赖范围。使用此依赖范围的Maven依赖，只对于测试classpath有效，在编译主代码或者运行项目的使用时将无法使用此类依赖。典型的例子就是JUnit，它只有在编译测试代码及运行测试的时候才需要。
 
provided: 已提供依赖范围。使用此依赖范围的Maven依赖，对于编译和测试classpath有效，但在运行时无效。典型的例子是servlet-api，编译和测试项目的时候需要该依赖，但在运行项目的时候，由于容器已经提供，就不需要Maven重复地引入一遍。
 
runtime: 运行时依赖范围。使用此依赖范围的Maven依赖，对于测试和运行classpath有效，但在编译主代码时无效。典型的例子是JDBC驱动实现，项目主代码的编译只需要JDK提供的JDBC接口，只有在执行测试或者运行项目的时候才需要实现上述接口的具体JDBC驱动。
 
system: 系统依赖范围。该依赖与三种classpath的关系，和provided依赖范围完全一致。但是，使用system范围依赖时必须通过systemPath元素显式地指定依赖文件的路径。由于此类依赖不是通过Maven仓库解析的，而且往往与本机系统绑定，可能造成构建的不可移植，因此应该谨慎使用。systemPath元素可以引用环境变量，如：

```xml
<dependency>  
  <groupId>javax.sql</groupId>  
  <artifactId>jdbc-stdext</artifactId>  
  <version>2.0</version>  
  <scope></scope>  
  <systemPath>${java.home}/lib/rt.jar</systemPath>  
</dependency>       
```

import(Maven 2.0.9及以上): 导入依赖范围。该依赖范围不会对三种classpath产生实际的影响，稍后会介绍到。

### 5.3 仓库

#### 5.3.1 仓库简介

　　没有 Maven 时，项目用到的 .jar 文件通常需要拷贝到 /lib 目录，项目多了，拷贝的文件副本就多了，占用磁盘空间，且难于管理。Maven 使用一个称之为仓库的目录，根据构件的坐标统一存储这些构件的唯一副本，在项目中通过依赖声明，可以方便的引用构件。

#### 5.3.2 仓库的布局

构件都有唯一的坐标，Maven 根据坐标管理构件的存储。如以下对 spring-core-4.1.0 的存储：

![仓库布局](http://7xngpc.com1.z0.glb.clouddn.com/vernartifact-1.png)   文件路径对应了：groupId/artifactId/version/artifactId-version.packaging

#### 5.3.3 仓库的分类

　　Maven 仓库分为本地仓库和远程仓库，寻找构件时，首先从本地仓库找，找不到则到远程仓库找，再找不到就报错；在远程仓库中找到了，就下载到本地仓库再使用。中央仓库是 Maven 核心自带的远程仓库，默认地址：[http://repo1.maven.org/maven2](http://repo1.maven.org/maven2)。除了中央仓库，还有其它很多公共的远程仓库。私服是架设在本机或局域网中的一种特殊的远程仓库，通过私服可以方便的管理其它所有的外部远程仓库。

##### 5.3.3.1 本地仓库

Maven 本地仓库默认地址为：${user.home}/.m2/repository。

通过修改 %MAVEN_HOME%/conf/settings.xml （或者：${user.home}/.m2/settings.xml，针对当前用户（推荐））配置文件可以更改本地仓库的位置。这个文件的配置会在后文中细讲。

##### 5.3.3.2 中央仓库

安装完 Maven ，本地仓库几乎是空的，这时需要从远程仓库下载所需构件。Maven 配置了一个默认的远程仓库，即中央仓库，找到 %MAVEN_HOME%/lib/maven-model-builder-3.2.1.jar，打开 org/apache/maven/model/pom-4.0.0.xml 超级POM：

##### 5.3.3.3 在项目中添加其他远程仓库

当中央仓库找不到所需的构件时，我们可以配置 pom.xml ，添加其它的远程仓库。

```xml
<repositories>
   <repository>
       <id>Sonatype</id>
       <name>Sonatype Repository</name>
       <url>http://repository.sonatype.org/content/groups/public/</url>
       <layout>default</layout>
       <releases>
           <enabled>true</enabled>
       </releases>
       <snapshots>
           <enabled>false</enabled>
       </snapshots>
   </repository>
</repositories>
```

其中 id 必须唯一，若不唯一，如设置为 central 将覆盖中央仓库的配置。

#### 5.3.4 配置仓库的标签

下面两个标签分别配置远程仓库和插件远程仓库

```
<repositories>
  <repository>
    <id>central</id>
    <name>Central Repository</name>
    <url>https://repo.maven.apache.org/maven2</url>
    <layout>default</layout>
    <snapshots>
      <enabled>false</enabled>
    </snapshots>
  </repository>
</repositories>
<pluginRepositories>
  <pluginRepository>
    <id>central</id>
    <name>Central Repository</name>
    <url>https://repo.maven.apache.org/maven2</url>
    <layout>default</layout>
    <snapshots>
      <enabled>false</enabled>
    </snapshots>
    <releases>
      <updatePolicy>never</updatePolicy>
    </releases>
  </pluginRepository>
</pluginRepositories>
```

也可以通过setttings.xml文件配置仓库地址：

```
<profiles>
    <profile>
      <id>ChinaMaven</id>
      <repositories>
        <repository>
          <id>ChinaMaven</id>
          <name>ChinaMaven</name>
          <url>http://maven.net.cn/content/groups/public/</url>
          <layout>default</layout>
          <snapshotPolicy>never</snapshotPolicy>
        </repository>
      </repositories>
      <pluginRepositories>
        <pluginRepository>
          <id>pluginMaven</id>
          <name>pluginMaven</name>
          <url>http://maven.net.cn/content/groups/public/</url>
          <layout>default</layout>
          <snapshotPolicy>always</snapshotPolicy>
        </pluginRepository>
      </pluginRepositories>
    </profile>
</profiles>
<activeProfiles>
  <activeProfile>ChinaMaven</activeProfile>
</activeProfiles>
```

##### 镜像仓库

镜像仓库可以理解为仓库的副本，从仓库中可以找到的构件，从镜像仓库中也可以找到。比如针对中央仓库 http://repo1.maven.org/maven2 ，在中国有它的镜像仓库，这样我们直接访问镜像仓库，更快更稳定。

```
<settings>
  ...
  <mirrors>
      <mirror>
          <id>maven.net.cn</id>
          <name>central mirror in china</name>
          <url>http://maven.net.cn/content/groups/public</url>
          <mirrorOf>central</mirrorOf>    <!  表明为central中央仓库配置镜像仓库-->
      </mirror>
  </mirrors>
  ...
</settings>
```

其中，<mirrorOf> 指明了为哪个仓库配置镜像，可以使用通配符如：<mirrorOf>*</mirrorOf>，或者 <mirrorOf>repo1,repo2</mirrorOf> 等进行匹配。一旦配置了镜像，所有针对原仓库的访问将转到镜像仓库的访问，原仓库将不再能直接访问，即使镜像仓库不稳定或停用。在搭建私服的时候，我们通常为所有仓库设置镜像为私服地址，通过私服对所有仓库进行统一管理。

### 5.4 build标签

build标签中用于配置编译时使用的一些配置，比如源码位置、测试代码位置、编译结果输出位置、插件的使用等。build标签是比较重要的标签，可以包含很多的内容。

这一块的内容较多，就不展开讲了，刚开始时也不太用得到，等后面用到的时候再去查。

### 5.5 properties标签

此标签中可以定义一些常量，比如某个依赖的版本。

```
<properties>
  <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  <spring.version>1.2.6</spring.version>
</properties>
```

后面使用到依赖于spring的构件的时候，可以使用如下方式设置版本号。如果有多处使用此版本，只需在properties标签中修改，可以省事。

```
<dependency>
  <groupId>org.springframework</groupId>
  <artifactId>spring-core</artifactId>
  <version>${spring.version}</version>
</dependency>
```


[这篇博客](http://blog.csdn.net/sunzhenhua0608/article/details/32938533)中有一个比较全的pom文件，里面包含了几乎所有的pom标签，可以参考一下。

## 6 生命周期与插件

### 6.1 生命周期模型

在maven出现之前，项目构建的生命周期已经存在，软件开发人员每天都在对项目进行清理、编译、测试和部署。可能大家使用的方式有所不同。

Maven的生命周期就是对所有的构建过程进行抽象和统一。Maven从项目和构建工具中学习和反思，总结了一套高度完善的、易于扩展的生命周期。这个生命周期包含了项目的情理、初始化、编译、测试、打包、集成测试、验证、部署和站点生成等几乎所有构建步骤。

Maven的生命周期是抽象的，也就是说生命周期本身不做任何实际的工作，实际的任务由插件来完成。
Maven的生命周期有三套，且是相互独立的，它们分别是clean、default和site。clean生命周期的目的是清理项目，default生命周期的项目是构建项目，而site生命周期的目的是建立项目站点。

每个生命周期都包含一些阶段，这些阶段是有顺序的，并且后面的阶段依赖于前面的阶段。三套生命周期相互独立，不会对其他生命周期产生影响。

#### 6.1.1 Clean 生命周期
当我们执行 mvn post-clean 命令时，Maven 调用 clean 生命周期，它包含以下阶段。

- pre-clean
- clean
- post-clean

Maven 的 clean 目标（clean:clean）绑定到了 clean 生命周期的 clean 阶段。它的 clean:clean 目标通过删除构建目录删除了构建输出。所以当 mvn clean 命令执行时，Maven 删除了构建目录。

#### 6.1.2 Default (or Build) 生命周期

这是 Maven 的主要生命周期，被用于构建应用。包括下面的 23 个阶段。

|     生命周期阶段      |                    描述                             |
| --------------------- | --------------------------------------------------------------------- |  
| validate          | 检查工程配置是否正确，完成构建过程的所有必要信息是否能够获取到。        |
| initialize        | 初始化构建状态，例如设置属性。                       |
| generate-sources      | 生成编译阶段需要包含的任何源码文件。                    |
| process-sources       | 处理源代码，例如，过滤任何值（filter any value）。             |
| generate-resources    | 生成工程包中需要包含的资源文件。                      |
| process-resources     | 拷贝和处理资源文件到目的目录中，为打包阶段做准备。             |
| compile             | 编译工程源码。                               |
| process-classes       | 处理编译生成的文件，例如 Java Class 字节码的加强和优化。          |
| generate-test-sources | 生成编译阶段需要包含的任何测试源代码。                   |
| process-test-sources  | 处理测试源代码，例如，过滤任何值（filter any values)。          |
| test-compile        | 编译测试源代码到测试目的目录。                       |
| process-test-classes  |  处理测试代码文件编译后生成的文件。                    |
| test          | 使用适当的单元测试框架（例如JUnit）运行测试。               |
| prepare-package     | 在真正打包之前，为准备打包执行任何必要的操作。               |
| package         | 获取编译后的代码，并按照可发布的格式进行打包，例如 JAR、WAR 或者 E AR 文件。 |
| pre-integration-test  | 在集成测试执行之前，执行所需的操作。例如，设置所需的环境变量。       |
| integration-test      | 处理和部署必须的工程包到集成测试能够运行的环境中。             |
| post-integration-test |  在集成测试被执行后执行必要的操作。例如，清理环境。              |
| verify          | 运行检查操作来验证工程包是有效的，并满足质量要求。             |
| install         | 安装工程包到本地仓库中，该仓库可以作为本地其他工程的依赖。         |
| deploy          | 拷贝最终的工程包到远程仓库中，以共享给其他开发人员和工程。         |


有一些与Maven生命周期相关的重要概念需要说明：

当一个阶段通过 Maven 命令调用时，例如 mvn compile，只有该阶段之前以及包括该阶段在内的所有阶段会被执行。

不同的 maven 目标将根据打包的类型（JAR / WAR / EAR），被绑定到不同的 Maven 生命周期阶段。

#### 6.1.3 Site 生命周期
Maven Site 插件一般用来创建新的报告文档、部署站点等。

阶段：

- pre-site
- site
- post-site
- site-deploy

### 6.2 插件

Maven的核心文件很小，主要的任务都是由插件来完成。进入到：%本地仓库%/org/apache/maven/plugins，可以看到一些下载好的插件：

![plugins](http://7xngpc.com1.z0.glb.clouddn.com/vernplugins.png)

Maven官网上有更详细的官方插件列表：

![官方插件列表](http://7xngpc.com1.z0.glb.clouddn.com/vernplugin2.png)

#### 6.2.1 插件的目标（Plugin Goals）

一个插件通常可以完成多个任务，每一个任务就叫做插件的一个目标。如执行mvn install命令时，调用的插件和执行的插件目标如下：

![dd](http://7xngpc.com1.z0.glb.clouddn.com/vernplugingoals.png)

每个插件都有哪些个目标，官方文档有更详细的说明：[Maven Plugins](http://maven.apache.org/plugins/index.html)

### 6.3 将插件绑定到生命周期

　　Maven的生命周期是抽象的，实际需要插件来完成任务，这一过程是通过将插件的目标（goal）绑定到生命周期的具体阶段（phase）来完成的。如：将maven-compiler-plugin插件的compile目标绑定到default生命周期的compile阶段，完成项目的源代码编译：

![lifecyclebinding](http://7xngpc.com1.z0.glb.clouddn.com/vernsimple-project_lifecyclebinding.png)

值得注意的是，执行一个生命周期阶段时，这个阶段之前的阶段上绑定的插件都将被执行。

#### 6.3.1 内置的绑定

Maven对一些生命周期的阶段（phase）默认绑定了插件目标，因为不同的项目有jar、war、pom等不同的打包方式，因此对应的有不同的绑定关系，其中针对default生命周期的jar包打包方式的绑定关系如下：



#### 6.3.2 自定义绑定

用户可以根据需要将任何插件目标绑定到任何生命周期的阶段，如：将maven-source-plugin的jar-no-fork目标绑定到default生命周期的package阶段，这样，以后在执行mvn package命令打包项目时，在package阶段之后会执行源代码打包，生成如：ehcache-core-2.5.0-sources.jar形式的源码包。

```
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-source-plugin</artifactId>
            <version>2.2.1</version>
            <executions>
                <execution>
                    <id>attach-source</id>
                    <phase>package</phase><!-- 要绑定到的生命周期的阶段 -->
                    <goals>
                        <goal>jar-no-fork</goal><!-- 要绑定的插件的目标 -->
                    </goals>
                </execution>
            </executions>
        </plugin>
    </plugins>
    ……
</build>
```

##### 6.3.3配置插件

Maven插件高度易扩展，可以方便的进行自定义配置。如：配置maven-compiler-plugin插件编译源代码的JDK版本为1.7：

```
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-compiler-plugin</artifactId>
    <configuration>
        <source>1.7</source>
        <target>1.7</target>
    </configuration>
</plugin>
```

## 7 依赖、继承和聚合

### 7.1 依赖

#### 7.1.1 传递性依赖

项目A依赖项目B，项目B依赖项目C，则项目A传递依赖于项目C。

![依赖传递](http://7xngpc.com1.z0.glb.clouddn.com/vernsimple-project_depgraph.png)
 
图中project-a传递依赖于project-d和project-e。

上图中，我们可能会思考一个问题：如果project-b和project-c同时依赖于project-d，但是版本不同，这时project-a该依赖porject-d的哪个版本呢？

这就是依赖冲突的问题，解决依赖传递有两个原则：

- 最短路径原则
- 第一声明优先原则

![最短路径](http://7xngpc.com1.z0.glb.clouddn.com/vernshotpath.png)

如上图，a->c->e2.0这条路径短，所以a传递依赖于e2.0。

![第一声明优先](http://7xngpc.com1.z0.glb.clouddn.com/vernfirstdefine.png)

如上图，两条路径长度一致，在a的pom中，如果依赖b写在依赖c前面，则a依赖e1.0，反之a依赖e2.0。

#### 7.1.2 可选依赖

有时候我们不想让依赖传递，那么可配置该依赖为可选依赖，将元素optional设置为true即可,例如：

```
<dependency>  
  <groupId>commons-logging</groupId>   
  <artifactId>commons-logging</artifactId>   
  <version>1.1.1</version>   
  <optional>true<optional>  
</dependency>  
```

那么依赖该项目的另以项目将不会得到此依赖的传递
 
#### 7.1.3 排除依赖

当我们引入第三方jar包的时候，难免会引入传递性依赖，有些时候这是好事，然而有些时候我们不需要其中的一些传递性依赖。比如我们的项目依赖spring-core,而spring-core依赖commons-logging，我们不想引入传递性依赖commons-logging，我们可以使用exclusions元素声明排除依赖，exclusions可以包含一个或者多个exclusion子元素，因此可以排除一个或者多个传递性依赖。需要注意的是，声明exclusions的时候只需要groupId和artifactId，而不需要version元素，这是因为只需要groupId和artifactId就能唯一定位依赖图中的某个依赖。

```
<dependency>    
     <groupId>org.springframework</groupId>  
     <artifactId>spring-core</artifactId>  
     <version>4.10</version>  
     <exclusions>  
           <exclusion>      
                <groupId>commons-logging</groupId>          
                <artifactId>commons-logging</artifactId>  
           </exclusion>  
     </exclusions>  
</dependency>  
```
  
则我们的项目不依赖commons-loggins。

### 7.2 聚合

4.2的例子中我们已经讲到了聚合的概念，聚合就是为了将多个项目集合起来，一条命令可以执行多个项目的生命周期。

聚合以modules标签来配置。

```
<modules>
  <module>ssh-domain</module>
  <module>ssh-dao</module>
  <module>ssh-service</module>
  <module>ssh-web</module>
</modules>
```

需要注意的是作为聚合的父项目，其packaging的类型必须为pom。

### 7.3 继承

子项目的pom可以继承父项目的pom中的元素，这样就可以避免一些重复的配置，比如子项目中都依赖的包可以放到父项目中。继承通过parent标签配置。

```
<parent>
  <groupId>com.anan.ssh</groupId>
  <artifactId>ssh-parent</artifactId>
  <version>0.0.1-SNAPSHOT</version>
</parent>
```

pom中可被继承的元素有：

- groupId ：项目组 ID ，项目坐标的核心元素；  
- version ：项目版本，项目坐标的核心元素；  
- description ：项目的描述信息；  
- organization ：项目的组织信息；  
- inceptionYear ：项目的创始年份；  
- url ：项目的 url 地址  
- develoers ：项目的开发者信息；  
- contributors ：项目的贡献者信息；  
- distributionManagerment ：项目的部署信息；  
- issueManagement ：缺陷跟踪系统信息；  
- ciManagement ：项目的持续继承信息；  
- scm ：项目的版本控制信息；  
- mailingListserv ：项目的邮件列表信息；  
- properties ：自定义的 Maven 属性；  
- dependencies ：项目的依赖配置；  
- dependencyManagement ：醒目的依赖管理配置；  
- repositories ：项目的仓库配置；  
- build ：包括项目的源码目录配置、输出目录配置、插件配置、插件管理配置等；  
- reporting ：包括项目的报告输出目录配置、报告插件配置等。  

### 7.4 聚合与继承的关系

区别 ：
1.对于聚合模块来说，它知道有哪些被聚合的模块，但那些被聚合的模块不知道这个聚合模块的存在。
2.对于继承关系的父 POM来说，它不知道有哪些子模块继承与它，但那些子模块都必须知道自己的父 POM是什么。

共同点 ：
1.聚合 POM与继承关系中的父POM的 packaging都是pom
2.聚合模块与继承关系中的父模块除了 POM之外都没有实际的内容。
Maven聚合关系与继承关系的比较
注：在现有的实际项目中一个 POM既是聚合POM，又是父 POM，这么做主要是为了方便
 


ps:这个博客拖拖拉拉写了近一个月终于写完了，强迫症和拖延症太严重了，有些 地方写的也不好，主要就是拖太久了，有些概念不如当时看的时候清晰了。以后要改正，一段时间只做一件事，而且要做好。