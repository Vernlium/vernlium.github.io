title: "Java序列化"
date: 2015-07-22 21:12:43
tags: java序列化
---

我在看程序的时候，经常会看到一些类实现了"Serializable"接口，一直对这个接口很奇怪，不知道它的作用是什么，今天专门查了一下资料，总结如下。

## 1、为什么要序列化？

在程序中，创建一个java对象，这个对象会随着程序的终止而消亡。但是在一些情况下，如果对象能够在程序不运行的情况下仍能存在并保存其信息，将非常有用。这样，在下次运行程序时，该对象将被重建并且拥有的信息与在程序上次运行是它所拥有的信息相同。

## 2、序列化是干什么的？

简单说就是为了保存在内存中的各种对象的状态（也就是成员变量，不是方法），并且可以把保存的对象状态再读出来。

Java对象的序列化将那些实现了**Serializable**接口的对象转换成一个字节序列，并能够在以后将这个字节序列完全恢复为原来的对象。

序列化分为两大部分：序列化和反序列化。序列化是这个过程的第一部分，将数据分解成字节流，以便存储在文件中或在网络上传输。反序列化就是打开字节流并重构对象。对象序列化不仅要将基本数据类型转换成字节表示，有时还要恢复数据。恢复数据要求有恢复数据的对象实例。ObjectOutputStream中的序列化过程与字节流连接，包括对象类型和版本信息。反序列化时，JVM用头信息生成对象实例，然后将对象字节流中的数据复制到对象数据成员中。

## 3、什么情况下需要序列化

对象序列化的概念加入到语言中是为了支持两种主要特性：

- 远程方法调用（RMI），它使存活于其他计算机上的对象使用起来就像是存活着本地上一样。当向远程对象发送消息时，需要通过对象序列化来传输参数和返回值。
- Java Beans。使用一个Bean时，一般情况下在设计阶段对它的状态信息进行配置。这种状态信息必须保存下来，并在程序启动进行后期恢复，这种具体工具就由序列化完成的。


## 4、相关注意事项

- a）序列化时，只对对象的状态进行保存，而不管对象的方法；
- b）当一个父类实现序列化，子类自动实现序列化，不需要显式实现Serializable接口；
- c）当一个对象的实例变量引用其他对象，序列化该对象时也把引用对象进行序列化；
- d）并非所有的对象都可以序列化，,至于为什么不可以，有很多原因了,比如：
    + d.1.安全方面的原因，比如一个对象拥有private，public等field，对于一个要传输的对象，比如写到文件，或者进行rmi传输 等等，在序列化进行传输的过程中，这个对象的private等域是不受保护的。
    + d.2. 资源分配方面的原因，比如socket，thread类，如果可以序列化，进行传输或者保存，也无法对他们进行重新的资源分 配，而且，也是没有必要这样实现。

## 5、如何进行序列化？

要序列化一个对象，首先创建OutputStream对象，然后将其封装在一个ObjectOutputStream对象中。这时，只需调用writeObject()即可将对象序列化，并将其发送给OutputStream(对象序列化是基于字节的，所以要使用InputStream和OutputStream继承层次结构)。要进行反序列化，需要将一个InputStream封装在ObjectInputStream中，然后调用readObject()。最后获得一个引用，它指向一个向上转型的Object，所以必须向下转型才能直接设置它们。

要注意的是：只有实现了Serializable或Externalizable接口的类的对象才能被序列化，否则抛出异常。

实例如下：

Student类定义如下：

```java
import java.io.Serializable;  
   
public class Student implements Serializable  
{  
    private String name;  
    private char sex;  
    private int year;  
    private double gpa;  
     
    public Student()  {
        System.out.println("Student constructor"); 
    }  
    public Student(String name,char sex,int year,double gpa)  {  
        System.out.println("Student constructor with:"+name+" "+sex+" "+year+" "+gpa);
        this.name = name;   
        this.sex = sex;  
        this.year = year;  
        this.gpa = gpa;  
    }  
     
    public void setName(String name)  {  
        this.name = name;  
    }  
     
    public void setSex(char sex)  {  
        this.sex = sex;  
    }  
     
    public void setYear(int year)  {  
        this.year = year;  
    }  
     
    public void setGpa(double gpa)  {  
        this.gpa = gpa;  
    }  
     
    public String getName()  {  
        return this.name;  
    }  
      
    public char getSex()  {  
        return this.sex;  
    }  
     
    public int getYear()  {  
        return this.year;  
    }  
     
    public double getGpa()  {  
        return this.gpa;  
    }  
}  
```

把Student类的对象序列化到文件student.txt，并从该文件中反序列化，向console显示结果。代码如下：

```java
import java.io.*;  
  
public class UseStudent  
{  
    public static void main(String[] args)  
    {  
        Student st = new Student("Tom",'M',20,3.6);  
        File file = new File("student.txt");  
        try  
        {  
         file.createNewFile();  
        }  
        catch(IOException e)  
        {  
         e.printStackTrace();  
        }  
        try  
        {  
         //Student对象序列化过程  
         FileOutputStream fos = new FileOutputStream(file);  
         ObjectOutputStream oos = new ObjectOutputStream(fos);  
         oos.writeObject(st);  
         oos.flush();  
         oos.close();  
         fos.close();  
        
         //Student对象反序列化过程  
         FileInputStream fis = new FileInputStream(file);  
         ObjectInputStream ois = new ObjectInputStream(fis);  
         Student st1 = (Student) ois.readObject();  
         System.out.println("name = " + st1.getName());  
         System.out.println("sex = " + st1.getSex());  
         System.out.println("year = " + st1.getYear());  
         System.out.println("gpa = " + st1.getGpa());  
         ois.close();  
         fis.close();  
        }  
        catch(ClassNotFoundException e)  
        {  
         e.printStackTrace();  
        }  
        catch (IOException e)  
        {  
         e.printStackTrace();  
        }               
    }  
} 
```

结果如下所示：

```
name = Tom
sex = M
year = 20
gpa = 3.6
```

注意:在对一个Serializable对象进行还原的过程中，没有调用任何构造器，包括默认的构造器。整个对象都是通过从InputStream中取得数据恢复而来。

### 5.1、Externalizable

有时，要考虑特殊的安全问题，不希望对象的某一部分被序列化，或者一个对象被还原以后，某个子对象需要重新创建，而不必将该子对象序列化。

在这些特殊情况下，可以通过实现Externalizable接口——代替实现Serializable接口——来对序列化过程进行控制。这个Externalizable接口继承了Serializable接口，同时增添了两个方法：writeExternal()和readExternal()。这两个方法会在序列化和反序列化还原的过程中被自动调用，以便执行一些特殊操作。

```java
import java.io.*;
import static net.mindview.util.Print.*;

class Blip1 implements Externalizable {
  public Blip1() {
    print("Blip1 Constructor");
  }
  public void writeExternal(ObjectOutput out)
      throws IOException {
    print("Blip1.writeExternal");
  }
  public void readExternal(ObjectInput in)
     throws IOException, ClassNotFoundException {
    print("Blip1.readExternal");
  }
}

class Blip2 implements Externalizable {
  Blip2() {
    print("Blip2 Constructor");
  }
  public void writeExternal(ObjectOutput out)
      throws IOException {
    print("Blip2.writeExternal");
  }
  public void readExternal(ObjectInput in)
     throws IOException, ClassNotFoundException {
    print("Blip2.readExternal");
  }
}

public class Blips {
  public static void main(String[] args)
  throws IOException, ClassNotFoundException {
    print("Constructing objects:");
    Blip1 b1 = new Blip1();
    Blip2 b2 = new Blip2();
    ObjectOutputStream o = new ObjectOutputStream(
      new FileOutputStream("Blips.out"));
    print("Saving objects:");
    o.writeObject(b1);
    o.writeObject(b2);
    o.close();
    // Now get them back:
    ObjectInputStream in = new ObjectInputStream(
      new FileInputStream("Blips.out"));
    print("Recovering b1:");
    b1 = (Blip1)in.readObject();
    // OOPS! Throws an exception:
    //! print("Recovering b2:");
    //! b2 = (Blip2)in.readObject();
  }
} 
```

结果为：

```
/* Output:
Constructing objects:
Blip1 Constructor
Blip2 Constructor
Saving objects:
Blip1.writeExternal
Blip2.writeExternal
Recovering b1:
Blip1 Constructor
Blip1.readExternal
*///:~
```

例中没有恢复Blip2对象，因为这样做会导致一个异常。这是因为：Blip1的构造器是public，而Blip2的构造器却不是，这样就会在恢复时造成异常。

恢复b1后，会调用Blip1默认构造器。这与恢复一个Serializable对象不同。对于Serializable对象，对象完全以它存储的二进制位为基础来构造，而不调用构造器。而对于一个Externalizable对象，所有普通的默认构造器都会被调用（包括在字段定义时的初始化），然后调用readExternal()。必须主要到：所有默认的构造器都会被调用，才能使Externalizable对象产生正确的行为。

```java
//: io/Blip3.java
// Reconstructing an externalizable object.
import java.io.*;
import static net.mindview.util.Print.*

public class Blip3 implements Externalizable {
  private int i;
  private String s; // No initialization
  public Blip3() {
    print("Blip3 Constructor");
    // s, i not initialized
  }
  public Blip3(String x, int a) {
    print("Blip3(String x, int a)");
    s = x;
    i = a;
    // s & i initialized only in non-default constructor.
  }

  public String toString() { return s + i; }
  public void writeExternal(ObjectOutput out) throws IOException {
    print("Blip3.writeExternal");
    // You must do this:
    out.writeObject(s);
    out.writeInt(i);
  }

  public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
    print("Blip3.readExternal");
    // You must do this:
    s = (String)in.readObject();
    i = in.readInt();
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    print("Constructing objects:");
    Blip3 b3 = new Blip3("A String ", 47);
    print(b3);
    ObjectOutputStream o = new ObjectOutputStream(
      new FileOutputStream("Blip3.out"));
    print("Saving object:");
    o.writeObject(b3);
    o.close();
    // Now get it back:
    ObjectInputStream in = new ObjectInputStream(
      new FileInputStream("Blip3.out"));
    print("Recovering b3:");
    b3 = (Blip3)in.readObject();
    print(b3);
  }
} 
```

结果为：

```
/* Output:
Constructing objects:
Blip3(String x, int a)
A String 47
Saving object:
Blip3.writeExternal
Recovering b3:
Blip3 Constructor
Blip3.readExternal
A String 47
*///:~
```

字段s和i只在第二个构造器中初始化，而不是在默认的构造器中初始化。这意味着假如不在readExternal()中初始化s和i，s就会为null，而i就会为0。如果注释掉跟随于”You must do this“后面的两行代码，当对象被还原后,s是null,而i为0。

如果从Externalizable对象继承，通常需要调用基类版本的writeExternal()和readExternal()来为基类组件提供恰当的存储和恢复功能。

因此，为了正常运行，不仅需要在writeExternal()方法中将来自对象的重要信息写入，还必须在readExternal()方法中恢复数据。

### 5.2、transient关键字

当某个字段被声明为transient后，默认序列化机制就会忽略该字段。

```java
import java.util.concurrent.*;
import java.io.*;
import java.util.*;
import static net.mindview.util.Print.*;

public class Logon implements Serializable {
  private Date date = new Date();
  private String username;
  private transient String password;
  public Logon(String name, String pwd) {
    username = name;
    password = pwd;
  }
  public String toString() {
    return "logon info: \n   username: " + username +
      "\n   date: " + date + "\n   password: " + password;
  }
  public static void main(String[] args) throws Exception {
    Logon a = new Logon("Hulk", "myLittlePony");
    print("logon a = " + a);
    ObjectOutputStream o = new ObjectOutputStream(
      new FileOutputStream("Logon.out"));
    o.writeObject(a);
    o.close();
    TimeUnit.SECONDS.sleep(1); // Delay
    // Now get them back:
    ObjectInputStream in = new ObjectInputStream(
      new FileInputStream("Logon.out"));
    print("Recovering object at " + new Date());
    a = (Logon)in.readObject();
    print("logon a = " + a);
  }
} 
```

结果为：

```
/* Output: (Sample)
logon a = logon info:
   username: Hulk
   date: Sat Nov 19 15:03:26 MST 2005
   password: myLittlePony
Recovering object at Sat Nov 19 15:03:28 MST 2005
logon a = logon info:
   username: Hulk
   date: Sat Nov 19 15:03:26 MST 2005
   password: null
*///:~
```

password是transient的，所以不会被自动保存到磁盘。另外，自动序列化机制也不会尝试恢复它。当对象被恢复时，password就变成了null。

由于Externalizable对象在默认情况下不保存它们的任何字段，所以transient关键字只能和Serializable对象一起使用。

### 5.3、Externalizable的替代方法

实现Serializable接口，并添加（是添加，不是覆盖或实现）名为writeObject()和readObject()的方法。这样一旦对象被序列化或者被反序列化还原，就会自动的分别调用这两个方法。也就是所，只要提供这两个方法，就会使用它们而不是默认的序列化机制。

这些方法必须具有准确的方法特征签名：

```
private void writeObject(ObjectOutputStream stream) throws IOException;

private void readObject(ObjectInputStream stream) 
throws IOException,ClassNotFoundException;
```

在调用ObjectOutputStream.writeObject()时，会检查所传递的Serializable对象，看看是否实现了自己的writeObject()。如果是这样，就跳到正常的序列化过程并调用它的writeObject()。readObject()的情形与此相同。

在writeObject()内部，可以调用defaultWriteObject()来选择执行默认的writeObject()。类似的，在readObject()内部，可以调用defaultReadObject()。

```java
import java.io.*;

public class SerialCtl implements Serializable {
  private String a;
  private transient String b;
  public SerialCtl(String aa, String bb) {
    a = "Not Transient: " + aa;
    b = "Transient: " + bb;
  }

  public String toString() { return a + "\n" + b; }
  private void writeObject(ObjectOutputStream stream)
  throws IOException {
    stream.defaultWriteObject();
    stream.writeObject(b);
  }

  private void readObject(ObjectInputStream stream)
  throws IOException, ClassNotFoundException {
    stream.defaultReadObject();
    b = (String)stream.readObject();
  }

  public static void main(String[] args)
  throws IOException, ClassNotFoundException {
    SerialCtl sc = new SerialCtl("Test1", "Test2");
    System.out.println("Before:\n" + sc);
    ByteArrayOutputStream buf= new ByteArrayOutputStream();
    ObjectOutputStream o = new ObjectOutputStream(buf);
    o.writeObject(sc);
    // Now get it back:
    ObjectInputStream in = new ObjectInputStream(
      new ByteArrayInputStream(buf.toByteArray()));
    SerialCtl sc2 = (SerialCtl)in.readObject();
    System.out.println("After:\n" + sc2);
  }
} 
```

结果为：

```
/* Output:
Before:
Not Transient: Test1
Transient: Test2
After:
Not Transient: Test1
Transient: Test2
*///:~
```

这个例子证明transient字段并非由defaultWriteObject()方法保存，而transient字段必须在程序中明确保存和恢复。

如果要使用默认机制写入对象的非transient部分，那么必须调用defaultWriteObject()作为writeObject()中的第一个操作，并让defaultReadObject()作为readObject()中的第一个操作。

## 6、总结

1）Java序列化就是把对象转换成字节序列，而Java反序列化就是把字节序列还原成Java对象。

2）采用Java序列化与反序列化技术，一是可以实现数据的持久化，在MVC模式中很是有用；二是可以对象数据的远程通信。

### 参考文献

1.《Thinking in Java》

2.[http://blog.csdn.net/wangloveall/article/details/7992448](http://blog.csdn.net/wangloveall/article/details/7992448)

3.[http://www.blogjava.net/jiangshachina/archive/2012/02/13/369898.html](http://www.blogjava.net/jiangshachina/archive/2012/02/13/369898.html)