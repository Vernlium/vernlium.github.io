---
title: pytorch_Resnet学习
date: 2020-10-08 22:15:30
tags: [pytorch, Resnet, deep_learning]
---

本文介绍使用pytorch运行Resnet网络的推理，及分析resnet的实现源码。

## pytorch Resnet网络学习

### 执行resnet

Pytorch中实现了常用的经典网络，并提供了预训练好的模型，我们可以直接加载模型并直接执行推理。代码如下：

``` python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnet152 = models.resnet152(pretrained=True)
# pretrained=True 设置从训练好的模型中加载参数，否则是未训练的模型

# resnet101.eval()
# resnet152.eval()
# resnet18.eval()
resnet50.eval() # 设置为推理模式

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import os
os.path.join("./")
from imagenet_1000_labels import imagenet_labels

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)

def img_infer(img_path):
    """
    image inference
    """
    # open image and show
    img = Image.open(img_path)
    plt.imshow(img) # 显示图片
    plt.axis('off') # 不显示坐标轴
    plt.show()

	  # image preprocess, resize to (224, 224)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

	  # do inference
    y = resnet50(x).cpu()
    y = torch.squeeze(y)

	  # get top5 vale
    softmax = torch.nn.Softmax(dim=0)
    top5_value, top5_index = torch.topk(softmax(y), 5, 0, True, True)

    print("top 5 value:")
    for i in range(5):
        value = top5_value[i].item()
        index = top5_index[i].item()

        label = imagenet_labels[index]
        print("[" + label + "]'s value is " + str(value))


img_path = "./cat1.jpeg"
img_infer(img_path)
img_path = "./goldfish.jpeg"
img_infer(img_path)

img_path = "./hotdog.jpeg"
img_infer(img_path)
```

> 注： 代码中用到的几个文件如下：
> - imagenet_1000_labels ： 结果对应的标签
> - 3幅测试图片:
>     - [cat1](https://user-images.githubusercontent.com/11350907/95468438-18828380-09b1-11eb-874e-f52c400fddfe.jpeg)
>     - [goldfish](https://user-images.githubusercontent.com/11350907/95468458-1d473780-09b1-11eb-8376-044bd3de5078.jpeg)
>     - [hotdog](https://user-images.githubusercontent.com/11350907/95468473-21735500-09b1-11eb-81e5-4f91e09fbc44.jpeg)

效果如下图所示：

![image](https://user-images.githubusercontent.com/11350907/95469511-5207be80-09b2-11eb-8b3f-9274888173ef.png)
![image](https://user-images.githubusercontent.com/11350907/95469570-62b83480-09b2-11eb-887a-001a2efdbca2.png)

### Resnet网络分析

Resnet网络的[论文](https://arxiv.org/pdf/1512.03385v1.pdf)中定义的网络的结构如下图：

![image](https://user-images.githubusercontent.com/11350907/95470222-1cafa080-09b3-11eb-88b8-bb9742300aaa.png)

Resnet的创新之处就在于，首次提出了残差结构（跨层连接，即H(x)=F(x)+x），解决了网络过深而导致的梯度消失的问题，可以使网络更深。

![image](https://user-images.githubusercontent.com/11350907/95472276-67cab300-09b5-11eb-889b-6f8393ca111e.png)

通过残差结构，使得网络可以更深。注意下图中第2列和第3列的差异，就是在于引入了残差结构。

![resnet_paper_figure3](https://user-images.githubusercontent.com/11350907/95470746-bd05c500-09b3-11eb-8235-8a158be4e238.png)

> 注：图中的虚线表示要先做一个下采样再进行连接。

Resnet18和Resnet50的结构如下图所示：



### Resnet代码分析

pytorch中实现resnet的代码入口为：

```python
def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

可以看到：resnet34和resnet50的差异在于block参数的不同。我们从论文的网络结构中也可以看到，resnet18/34使用的结构为两个3*3的卷积，而resnet50及以上用的是1/*1/3/*3/1/*1 3个卷积。

Resnet类的定义为：

```python

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
```

注意`_make_layer`函数，resnet18/34的layer1是没有`downsample`的，而resnet50及以上的layer1有`downsample`。


`BasicBlock`的定义为：

```python

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

残差结构就是通过 `out += identity` 来实现的。

`Bottleneck`的定义如下，和论文中用的卷积是一致的。

```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

这个代码结合上面的图看起来还是很清晰的。