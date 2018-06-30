title: "FBReader源码分析"
date: 2015-06-25 15:40:49
tags: epub
---
## FBReader自定义文件格式
FBReader程序自己定义的三种文件格式类以及资源文件、epub文件又都对应着哪类文件格式。

FBReader的自定义文件格式类分别在org.geometerplus.zlibrary.core.filesystem包与org.amse.ys.zip包里面。

org.geometerplus.zlibrary.core.filesystem包里面，

**ZLFile**类是基类，ZLResourceFile、ZLPhysicalFile、ZLArchiveFile是ZLFile类的子类，ZLZipEntryFile是ZLArchiveFile的子类。

- **ZLResourceFile**类专门用来处理资源文件，这一章中要解析的assets文件夹下的资源文件都可以ZLResourceFile类来处理。
- **ZLPhysicalFile**类专门用来处理普通文件，eoub文件就可以用一个ZLPhysicalFile类来代表。
- **ZLZipEntryFile**类用来处理epub文件内部的xml文件，这个类会在第五章“epub文件处理 -- 解压epub文件”中出现。

这三个文件类都实现了getInputStream抽象方法，不用的文件类会通过这个方法获得针对当前文件类的字节流类。
