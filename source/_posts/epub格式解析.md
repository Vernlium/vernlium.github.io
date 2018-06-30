title: "epub格式解析"
date: 2015-06-10 10:07:54
tags: epub
---

EPUB 2 was initially standardized in 2007 as a successor format to the Open eBook Publication Structure or "OEB", which was originally developed in 1999. A maintenance release, EPUB 2.0.1, approved in 2010, was the final release in the EPUB 2 branch.

In October, 2011, EPUB 3 superseded EPUB 2 when EPUB 3.0 was approved as a final Recommended Specification. A maintenance release, EPUB 3.0.1, was approved as a Final Recommended specification and became the current version of EPUB in June, 2014.

## 简介

epub格式是一种常见的电子书格式，其优点是体积小、与设备无关，在任何尺寸的屏幕上都能自动排版，因此比较流行。

epub格式建立在EPUB标准的基础之上。目前EPUB标准的最新版本是3.0.1,EPUB 3较其前一版本EPUB2有较大改动。EPUB 2在2007年发布，2010年的EPUB 2.0.1是EPUB 2的最后一个版本。EPUB 3在2011年10月发布，并被IDOP组织认可为最终推荐规范，EPUB 3取代了EPUB 2。2014年7月，EPUB 3.0.1发布。

先来看一下EPUB 2.0.1

>EPUB 2.0.1

>EPUB 2.0.1 is a maintenance release of EPUB 2. Its development was chartered in 2009, and the final standard was approved by the IDPF Membership as a Recommended Specification in May, 2010.

>EPUB 2.0.1 is defined by three open standard specifications, the Open Publication Structure (OPS), Open Packaging Format (OPF) and Open Container Format (OCF)

>EPUB 2.0.1 was superseded by  EPUB 3.0 in October, 2011. As of June 2014 the latest released version of EPUB is EPUB 3.0.1. EPUB 2.x is now considered obsolete and is no longer under active maintenance.

EPUB 2.0.1包含三个部分，也就是三个标准：

- 内容容器标准（a content container standard：开放容器格式，Open Container Format - OCF):OCF定义了把一组文件集合打包进一个ZIP压缩文件的规则。
- 打包标准（a packaging atandard：开放打包格式，Open Packaging Format - OPF):此标准定义了把一个OPS出版物的不同组件组合在一起的机制和提供了电子出版物的附加的结构和语义。OPF的作用如下：
  - 描述和引用电子出版物的所有组件（例如：markup files、images、navigation structures）
  - 提供出版级别元数据
  - 指定出版物的线性阅读顺序
  - 提供了一种指定描述全局导航结构（NCX）的机制
- 内容审定标准（a content markup standard：开放出版结构，Open publication Structure - OPS):OPS提供了一种表示电子出版物内容的标准。

EPUB 3较EPUB 2有较大改进，除了OCF其他两个标准都进行了修改，并添加了一个新的标准。

>EPUB 3, the third major release of the standard, consists of a set of four specifications, each defining an important component of an overall EPUB Publication:

>- EPUB Publications 3.0 [Publications30], which defines publication-level semantics and overarching conformance requirements for EPUB Publications.

>- EPUB Content Documents 3.0 [ContentDocs30], which defines profiles of XHTML, SVG and CSS for use in the context of EPUB Publications.

>- EPUB Open Container Format (OCF) 3.0 [OCF3], which defines a file format and processing model for encapsulating a set of related resources into a single-file (ZIP) EPUB Container.

>- EPUB Media Overlays 3.0 [MediaOverlays30], which defines a format and a processing model for synchronization of text and audio.

EPUB 3由四部分组成：

- EPUB Publications 3.0：取代OPF2.0.1。
- EPUB Content Documents 3.0：取代OPS2.0.1。
- EPUB Open Container Format (OCF) 3.0：继承自OCF2.0.1。
- EPUB Media Overlays 3.0：定义文字和音频同步的格式和处理模式。

#### EPUB3和EPUB2比较

| Area          | EPUB 3 Specification                  | EPUB 2.0.1 Specification |
| ------------- |---------------------------------------| -------------------------------|
| Overview      | EPUB 3 Overview | (throughout) |
| Publication-level Specification & Package Docs      | EPUB Publications 3.0      |   Open Packaging Format 2.0.1 |
| EPUB Navigation Documents |   EPUB Content Documents 3.0   | N/A (NCX referenced as DAISY specification)   |
| Media Overlays | EPUB Media Overlays 3.0     | N/A   |
| Container packaging | EPUB Open Container Format 3.0     |  Open Container Format 2.0.1  |
|Changes from previous version |  EPUB 3 Changes from EPUB 2.0.1     | (throughout)   |

## 文件组成

一个未经加密处理的epub电子书由以下三部分组成：

1. META-INF（文件夹，有一个文件container.xml）
2. OEBPS（文件夹，包含images文件夹、很多xhtml文件、*.css文件和content.opf文件）
3. mimetype

### 文件mimetype

每一个epub电子书均包含一个名为mimtype的文件，且内容不变，用以说明epub的文件格式。文件内容为：

    application/epub+zip

### 目录：META-INF

META-INF用于存放容器信息，默认情况下改目录包含一个文件，即container.xml，文件内容如下：

```xml
<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
	<rootfiles>
		<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
	</rootfiles>
</container>
```

container.xml文件的主要功能用于告诉阅读器，电子书的根文件（rootfile）的路径和打开格式，一般来说，该containerxml文件也不需要任何修改，除非改变了根文件的路径和文件名称。
除了container.xml文件之外，OCF还规定了以下几个文件：

- manifest.xml 		文件列表
- metadata.xml 		元数据
- signatures.xml 	数字签名 
- encryption.xml 	加密
- rights.xml 		权限管理

这些目录是可选的

### 目录：OEBPS

OEPBS目录用于存放OPF文档、CSS文件、NCX文档。

#### OPF文件（★）

OPF文档是epub的核心文件，且是一个标准的xml文件，依据OPF规范，此文件的根元素为`<package>`

    <package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uuid_id">

其内容主要由五部分组成：

##### 1.`<metadata>`

元数据信息，此信息是书籍的出版信息，由两个子元素组成。

(1)`<dc-metadata>`,其元素构成采用[dubline core(DC)](http://baike.baidu.com/link?url=zgnNQm9IxPVmr5-BxhvoipgAPEfQiZf-fMGa0ZEhyF7qpuTUVXVDpfKCacvkrhNX7icjg7TDPtgEvc5Rx0HllK)的15项核心元素，包括：

- `<dc-title>`：标题
- `<dc-creator>`：责任者
- `<dc-subject>`：主题词或关键词
- `<dc-descributor>`：内容描述
- `<dc-date>`：日期
- `<dc-type>`：类型
- `<dc-publisher>`：出版者
- `<dc-contributor>`：发行者
- `<dc-format>`：格式
- `<dc-identifier>`：标识信息
- `<dc-source>`：来源信息
- `<dc-language>`：语言
- `<dc-relation>`：相关资料
- `<dc-coverage>`：覆盖范围
- `<dc-rights>`：权限描述

(2)`<x-metadata>`

扩展元素。如果有些信息在上述元素中无法描述，则在此元素中进行扩展。

例如：

```xml
<metadata xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance
xmlns:opf="http://www.idpf.org/2007/opf" xmlns:dcterms="http://purl.org/dc/terms/
xmlns:calibre="http://calibre.kovidgoyal.net/2009/metadata
xmlns:dc="http://purl.org/dc/elements/1.1/">
	<dc:title>1984</dc:title>
	<dc:creator>[英] 乔治·奥威尔</dc:creator>
	<dc:subject>ibook.178.com</dc:subject>
	<dc:language>zh-cn</dc:language>
	<dc:date>2010-12-30</dc:date>
	<dc:contributor>ibook.178.com [http://ibook.178.com]</dc:contributor>
	<dc:type>普通图书</dc:type>
	<dc:format>Text/html(.xhtml,.html)</dc:format>
	<meta name="cover" content="cover-image"/>
</metadata>
```

##### 2.`<manifest>`

文件列表，列出书籍出版的所有文件，但是不包括：mimetype、container.xml、content.opf，由一个子元素构成

    <item id="" href="" media-type="">

其中

- id:文件的id号
- href:文件的相对路径
- media-type:文件的媒体类型

例如：

```xml
<manifest>
	<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml" />
	<item href="cover.xhtml" id="cover" media-type="application/xhtml+xml"/>
	<item href="copyright.xhtml" id="copyright" media-type="application/xhtml+xml"/>
	<item href="catalog.xhtml" id="catalog" media-type="application/xhtml+xml"/>
	<item href="chap0.xhtml" id="chap0" media-type="application/xhtml+xml"/>
</manifest>
```

##### 3.`<spine toc="ncx">`

脊骨，其主要功能是提供书籍的线性阅读次序。由一个子元素构成：

    <itemref idref="copyright">

其中
idref:即参照manifest列出的id
例如：

    <spine toc="ncx">
    	<itemref idref="cover" />
    	<itemref idref="copyright" />
	</spine>

##### 4.`<guide>`

指南，一次列出电子书的特定页面，例如封面、目录、序言等，属性值指向文件保存地址。一般情况下，epub电子书可以不用该元素。

例如：

```xml
<guide>
	<reference href="cover.xhtml" type="text" title="封面"/>
	<reference href="catalog.xhtml" type="text" title="目录"/>
</guide>
```

##### 5.`<tour>`

导读，可以根据不同的读者水平或阅读目的，按一定的次序，选择电子书中的部分页面组成导读。一般情况下，epub电子书可以不用该元素。

#### NCX文件（★）

NCX文件是epub电子书的又一个核心文件，用于制作电子书的目录，其文件的命名通常为toc.ncx。ncx文件也是一个xml文件。
ncx代表“Navigation Center eXtended”，意思大致就是导航文件，这个文件与目录有直接的关系。

.ncx文件中最主要的节点是navMap。navMap节点是由许多navPoint节点组成的。而navPoint节点则是由navLabel、content两个子节点组成。

- navPoint节点中，playOrder属性定义当前项在目录中显示的次序。navLabel子节点中的text节点定义了每个目录的名字。

- content子节点的src属性定义了对应每个章节的文件的具体位置。

nvaPoint节点可以嵌套，就是书籍的目录是层级目录。

下面是一个toc.ncx文件的实例。

```xml
<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
	<head>
		<meta content="178_0" name="dtb:uid"/>
		<meta content="2" name="dtb:depth"/>
		<meta content="0" name="dtb:totalPageCount"/>
		<meta content="0" name="dtb:maxPageNumber"/>
	</head>
	<docTitle>
		<text>1984</text>
	</docTitle>
	<docAuthor>
		<text>[英] 乔治·奥威尔</text>
	</docAuthor>
	<navMap>
		<navPoint id="catalog" playOrder="0">
			<navLabel>
				<text>目录</text>
			</navLabel>
			<content src="catalog.xhtml"/>
		</navPoint>
		<navPoint id="chap0" playOrder="1">
			<navLabel>
				<text>前言</text>
			</navLabel>
			<content src="chap0.xhtml"/>
		</navPoint>
		<navPoint id="chap1" playOrder="2">
			<navLabel>
				<text>　　第一部</text>
			</navLabel>
			<content src="chap1.xhtml"/>
		</navPoint>
		<navPoint id="chap2" playOrder="3">
			<navLabel>
				<text>　　第1节</text>
			</navLabel>
			<content src="chap2.xhtml"/>
		</navPoint>
		<navPoint id="chap3" playOrder="4">
			<navLabel>
				<text>　　第2节</text>
			</navLabel>
			<content src="chap3.xhtml"/>
		</navPoint>
		<navPoint id="chap4" playOrder="5">
			<navLabel>
				<text>　　第3节</text>
			</navLabel>
			<content src="chap4.xhtml"/>
		</navPoint>
	</navMap>
</ncx>
```    

看到这里，可能会有有人觉得.opf文件与.ncx文件有一点重复：.opf文件的item节点中的href属性描述了各个章节文件的位置与顺序，.ncx文件中的conten节点中的src属性也描述了各个章节文件的位置与顺序。其实他们还是有区别的，区别就在于，.opf文件定义的是读者在顺序阅读时会用到的章节文件与它们的顺序，.ncx文件则定义的是目录中会用到的章节文件与它们的顺序。如果存在某些附件性质的内容被希望在目录中出现，但却不希望在读者顺序阅读的时候出现时，那么就可以通过对.opf文件和.ncx文件进行不同的设置来达到这个目的。当然，大部分的时候，.opf与.ncx这两个文件的内容基本是重合的。
