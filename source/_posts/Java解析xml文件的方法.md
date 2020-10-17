title: "Java解析xml文件的方法"
date: 2015-06-17 15:20:02
tags: xml
categories: java
---
本文介绍xml.

## 什么是 XML?

- XML指可扩展标记语言（EXtensible Markup Language）
- XML 是一种标记语言，很类似 HTML 
- XML 的设计宗旨是传输数据，而非显示数据 
- XML 标签没有被预定义。您需要自行定义标签。 
- XML 被设计为具有自我描述性。 
- XML 是 W3C 的推荐标准 
- XML 与 HTML 的主要差异
- XML 不是 HTML 的替代。

###XML 和 HTML 为不同的目的而设计：

- XML 被设计为传输和存储数据，其焦点是数据的内容。
- HTML 被设计用来显示数据，其焦点是数据的外观。
- HTML 旨在显示信息，而 XML 旨在传输信息。

XML是一种通用的数据交换格式，它与平台、语言、系统无关，给数据集成与交互带来了极大的方便。对XML本身的语法与技术细节，可以参阅[w3c官方网站](http://www.w3c.org)上相关的技术文献，包括：DOM(Document Object Model)、DTD(Document Type Definition)、SAX(Simple API for XML)、XSD(Xml Schema Definition)、XSLT(Extensible Stylesheet Language Transformation)。

### XML 文档形成一种树结构
XML文档必须包含根元素。该元素是所有其他元素的父元素。

XML文档中的元素形成了一棵文档树。这棵树从根部开始，并扩展到树的最底端。

所有元素均可拥有子元素：

```xml
<root>
    <child>
        <subchild>.....</subchild>
    </child>
</root>
```

父、子以及同胞等术语用于描述元素之间的关系。父元素拥有子元素。相同层级上的子元素成为同胞（兄弟或姐妹）。

所有元素均可拥有文本内容和属性（类似 HTML 中）。
实例

![domtree](http://7xngpc.com1.z0.glb.clouddn.com/vernct_nodetree1.gif)

上图表示下面的 XML 中的一本书：
    
```xml
<bookstore>
    <book category="COOKING">
        <title lang="en">Everyday Italian</title> 
        <author>Giada De Laurentiis</author> 
        <year>2005</year> 
        <price>30.00</price> 
    </book>
    <book category="CHILDREN">
        <title lang="en">Harry Potter</title> 
        <author>J K. Rowling</author> 
        <year>2005</year> 
        <price>29.99</price> 
    </book>
    <book category="WEB">
        <title lang="en">Learning XML</title> 
        <author>Erik T. Ray</author> 
        <year>2003</year> 
        <price>39.95</price> 
    </book>
</bookstore>
```

例子中的根元素是`<bookstore>` 。文档中的所有 `<book>` 元素都被包含在 `<bookstore>` 中。

`<book>`元素有 4 个子元素：`<title>`、`<author>`、`<year>`、`<price>`。


XML在不同语言里解析方式都是一样的，只不过实现的语法不同。基本的解析方式有两种，一种是DOM，另一种是SAX。DOM是基于XML文档树结构的解析，SAX是基于事件流的解析。
本文介绍Java解析xml文档的几种方法。

##DOM解析xml文档
为xml文档的已解析版本定义了一组接口。解析器读入整个文档，然后构建一个驻留内存的DOM树结构，然后可以使用DOM接口来操作这个树结构。
*优点*：整个文档树在内存中，便于操作；支持删除、修改、重新排列等多种功能。
*缺点*：将整个文档调入内存（包括无用的节点），如果文档过大，则占用过多空间且浪费时间。
*适用场合*：一旦解析了文档还需多次访问这些数据，硬件资源充足（内存、CPU）。

解析代码如下：

```java
public class DomTest {
    public void parseXmlPath(String xmlPath) throws Exception{
        parseXmlFile(new File(xmlPath));
    }
    private void parseXmlFile(File file) throws Exception{
        //获取DocumentBuilder工厂
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        //从DocumentBuilder工厂中获取DocumentBuilder对象
        DocumentBuilder db = dbf.newDocumentBuilder();
        //从DocumentBuilder对象中获取Document对象
        Document doc = db.parse(file);
        //从Document对象中获取根元素对象
        //Element 接口表示 HTML 或 XML 文档中的一个元素。
        //元素可能有与它们相关的属性；由于 Element 接口继承自 Node，
        //所以可以使用一般 Node 接口属性 attributes 来获得元素所有属性的集合。
        //Element 接口上有通过名称获得 Attr 对象或通过名称获得属性值的方法。
        //在 XML 中（其中的属性值可能包含实体引用），应该获得 Attr 对象来检查表示属性值的可能相当杂的子树。
        Element root = doc.getDocumentElement();
        parseElement(root);
    }
    private void parseElement(Element element){
        String tagName = element.getNodeName();
        //获取当前元素的子节点
        NodeList childern = element.getChildNodes();
        System.out.println("<" + tagName +">");
        //实现 NamedNodeMap 接口的对象用于表示可以通过名称访问的节点的集合。
        NamedNodeMap map = element.getAttributes();
        if(map != null){
            for(int i = 0;i<map.getLength();i++){
                Attr attr = (Attr) map.item(i);
                String attrName = attr.getName();
                String attrValue = attr.getValue();
                System.out.println("  " + attrName +"=\""+attrValue+"\"");
            }
        }
        for(int i=0;i<childern.getLength();i++){
            Node node = childern.item(i);
            short nodeType = node.getNodeType();
            if(nodeType == Node.ELEMENT_NODE){
                parseElement((Element)node);
           }else if(nodeType == Node.TEXT_NODE){
                System.out.println(node.getNodeName()+":"+node.getNodeValue());
            }else if(nodeType == Node.COMMENT_NODE){
                System.out.print("<!--");
                Comment comment = (Comment)node;
                String data = comment.getText();
                System.out.print(data);
                System.out.print("-->");
            }
        }
        System.out.println("</"+tagName + ">");
    }
}
```

##SAX解析XML文档
为解决DOM的问题，出现了SAX。SAX，事件驱动。当解析器发现元素开始、元素结束、文本、文档的开始或结束等时，发送事件，编写响应这些事件的代码，保存数据。
*优点*：不用事先调入整个文档，占用资源少；SAX解析器代码比DOM解析器代码小。
*缺点*：不是持久的；事件过后，若没有保存数据，那么数据丢失；无状态；从事件中只能得到文本，但不知道给文本属于哪个元素。
*适用场合*：Applet；只需要XML文档的少量内容，很少回头访问；机器内存小。

代码如下：

##DOM4J解析xml文档
DOM4J是一个非常优秀的Java Xml API，具有性能优异、功能强大和易于使用的特点，同时它也是一个开放源代码的软件。现在很多Java项目中都使用DOM4J读写xml文件，很多Spring项目也是。
