---
title: 新机器hexo博客恢复
date: 2019-07-01 08:06:38
tags: hexo
---

换了新电脑后，又要恢复一遍hexo博客，是件很麻烦的事情。好在我之前已经对之前对博客环境进行了备份，只需要恢复即可。没想到对是，由于新换对是macos系统，恢复起来也费了不少功夫。

### 博客备份

首先，需要对之前电脑对博客环境进行备份，参考如下博客：

http://stevenshi.me/2017/05/07/hexo-backup/

更换电脑后，对博客进行恢复。

### 博客恢复

#### 1、git、nodejs等软件安装

参考：https://hexo.io/docs/#Requirements

#### 2、hexo安装

首先在本地建立自己的博客文件夹，比如 `/Users/anan/01_blog/hexo`，命令行进入该文件夹内，开始安装 hexo：

```
$npm install hexo-cli -g //注: -g表示全局安装
$npm install hexo-server -g //注： hexo3.0之后server模块是独立的,需要单独安装
```

完成之后可以通过命令查看hexo是否安装成功：

```
　$hexo -v
```


#### 3、备份博客拉取

当环境建立好后，在自己的博客文件夹下`/Users/anan/01_blog/hexo`,执行命令：
　　
```
　$hexo init //初始化，自动下载搭建网站所需的所有文件
　$npm install //安装依赖包，执行完之后，会多一个node_modules目录
```

然后删除如下文件，因为这些文件已经在github进行了备份。
　
```
$rm _config.yml db.json package.json 
$rm -rf scaffolds/ themes/ source/
$rm .npmignore
```

之后添加远程仓库分支至本地：
　
```
$git init
$git remote add origin https://github.com/yourusername/yourusername.github.io.git
```

查看远程仓库所有分支：
```
$git branch -r
```

这一步不知道为什么我的没有任何显示，直接执行如下命令：

```
$git pull origin hexo
```

### 常见错误

#### hexo sh: highlight_alias.json: Permission denied

使用npm安装hexo:

```
npm install -g hexo-cli
```

可能会报下面这种错：

```
 sh:highlight_alias.json:Permission denied
```

这是由于权限问题，在安装之前执行下面的命令：

```
npm config set unsafe-perm true
```

#### hexo g报错：Error: ENOENT: no such file or directory, scandir '/Users/anan/01_blog/hexo/node_modules/node-sass/vendor'

```
ERROR Plugin load failed: hexo-renderer-sass
Error: ENOENT: no such file or directory, scandir '/Users/anan/01_blog/hexo/node_modules/node-sass/vendor'
    at Object.readdirSync (fs.js:790:3)
    at Object.getInstalledBinaries (/Users/anan/01_blog/hexo/node_modules/node-sass/lib/extensions.js:130:13)
    at foundBinariesList (/Users/anan/01_blog/hexo/node_modules/node-sass/lib/errors.js:20:15)
    at foundBinaries (/Users/anan/01_blog/hexo/node_modules/node-sass/lib/errors.js:15:5)
    at Object.module.exports.missingBinary (/Users/anan/01_blog/hexo/node_modules/node-sass/lib/errors.js:45:5)
    at module.exports (/Users/anan/01_blog/hexo/node_modules/node-sass/lib/binding.js:15:30)
    at Object.<anonymous> (/Users/anan/01_blog/hexo/node_modules/node-sass/lib/index.js:14:35)
    at Module._compile (internal/modules/cjs/loader.js:776:30)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:787:10)
    at Module.load (internal/modules/cjs/loader.js:653:32)
    at tryModuleLoad (internal/modules/cjs/loader.js:593:12)
    at Function.Module._load (internal/modules/cjs/loader.js:585:3)
    at Module.require (internal/modules/cjs/loader.js:690:17)
    at require (internal/modules/cjs/helpers.js:25:18)
    at Object.<anonymous> (/Users/anan/01_blog/hexo/node_modules/hexo-renderer-sass/lib/renderer.js:4:12)
    at Module._compile (internal/modules/cjs/loader.js:776:30)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:787:10)
    at Module.load (internal/modules/cjs/loader.js:653:32)
    at tryModuleLoad (internal/modules/cjs/loader.js:593:12)
    at Function.Module._load (internal/modules/cjs/loader.js:585:3)
    at Module.require (internal/modules/cjs/loader.js:690:17)
    at require (/Users/anan/01_blog/hexo/node_modules/hexo/lib/hexo/index.js:219:21)
    at /Users/anan/01_blog/hexo/node_modules/hexo-renderer-sass/index.js:4:20
    at fs.readFile.then.script (/Users/anan/01_blog/hexo/node_modules/hexo/lib/hexo/index.js:232:12)
    at tryCatcher (/Users/anan/01_blog/hexo/node_modules/hexo/node_modules/bluebird/js/release/util.js:16:23)
    at Promise._settlePromiseFromHandler (/Users/anan/01_blog/hexo/node_modules/hexo/node_modules/bluebird/js/release/promise.js:517:31)
    at Promise._settlePromise (/Users/anan/01_blog/hexo/node_modules/hexo/node_modules/bluebird/js/release/promise.js:574:18)
    at Promise._settlePromise0 (/Users/anan/01_blog/hexo/node_modules/hexo/node_modules/bluebird/js/release/promise.js:619:10)
```

解决办法：

```
sudo npm rebuild node-sass
```

