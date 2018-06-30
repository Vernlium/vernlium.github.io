title: "Pattern类的使用"
date: 2015-06-15 10:09:51
tags: Java
---
Pattern类的定义为：

```java
public final class Pattern extends Object implements Serializable
```

此类为正则表达式的编译表示形式。 

指定为字符串的正则表达式必须首先被编译为此类的实例。然后，可将得到的模式用于创建 Matcher 对象，依照正则表达式，该对象可以与任意字符序列匹配。执行匹配所涉及的所有状态都驻留在匹配器中，所以多个匹配器可以共享同一模式。 

因此，典型的调用顺序是 

```java
Pattern p = Pattern.compile("a*b");
Matcher m = p.matcher("aaaaab");
boolean b = m.matches();
```

在仅使用一次正则表达式时，可以方便地通过此类定义 matches 方法。此方法编译表达式并在单个调用中将输入序列与其匹配。语句 

```
boolean b = Pattern.matches("a*b", "aaaaab");
```

等效于上面的三个语句，尽管对于重复的匹配而言它效率不高，因为它不允许重用已编译的模式。 
此类的实例是不可变的，可供多个并发线程安全使用。Matcher类的实例用于此目的则不安全。 


实例：

```java
public class DirList {
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        File path = new File(".");
        String regex = ".*java$"
        String[] list;
        list = path.list(new FilenameFilter(){
            private Pattern pattern = Pattern.compile(regex);
            public boolean accept(File dir, String name) {
                return pattern.matcher(name).matches();
            }
        });
        Arrays.sort(list,String.CASE_INSENSITIVE_ORDER);
        for(String dirItem: list)
            System.out.println(dirItem);
    }
}
```

该类的作用是获取 当前目录下所有以java结尾的文件。
list()方法会回调accept()方法，这种结构称为回调，更具体的说这是一个“策略模式”的例子。因为list()实现了基本的功能，而且按照FilenameFilter的形式提供了这个策略，以便完善list()在提供服务时所需的算法。
因为list()接受FilenameFilter对象作为参数，这意味着我们可以传递实现了FilenameFilter接口的任何类的对象，用以选择list()方法的行为方式。策略的目的就是提供了代码行为的灵活性。
accept()会使用一个正则表达式的matcher对象，在查看此正则表达式regex是否匹配这个文件的名字。通过使用accept(),list()方法会返还一个数组。