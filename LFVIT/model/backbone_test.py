import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
}

#resnet18，也就是两个3x3卷积
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
    # inplanes：输入通道数
    # planes：输出通道数

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        #中间部分省略
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #为后续相加保存输入
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            #遇到降尺寸或者升维的时候要保证能够相加
            identity = self.downsample(x)

        out += identity#论文中最核心的部分，resnet的简洁和优美的体现
        out = self.relu(out)

        return out

# resnet50 1*1 + 3*3 + 1*1
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print("conv1=>bn1=>relu=>", out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # print("conv2=>bn2=>relu=>", out.size())

        out = self.conv3(out)
        out = self.bn3(out)

        # print("conv3=>bn3=>", out.size())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    

def resnet50_backbone(**kwargs):
    # Constructs a ResNet-50 model_hyper.
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    # load pre-trained weights
    # save_model = model_zoo.load_url(model_urls['resnet50'])
    save_model = torch.load('/Users/mianmaokuchuanma/database/model/resnet50.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    j, k = 0, 0
    model_list1 = list(model_dict.keys())
    model_list2 = list(state_dict.keys())
    while 1:
        if j >= len(model_list1) or k >= len(model_list2):
            break
        layer1, layer2 = model_list1[j], model_list2[k]
        if "num_batches_tracked" in layer1:
            j += 1
            layer1 = model_list1[j]
        print(layer1, model_dict[layer1].shape, end=' ')
        print(layer2, state_dict[layer2].shape)
        j += 1
        k += 1
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    return model


def resnet34_backbone(**kwargs):
    # Constructs a ResNet-18 model_hyper.
    model = ResNetBackbone(BasicBlock, [3, 4, 6, 3], **kwargs)

    # load pre-trained weights
    # save_model = model_zoo.load_url(model_urls['resnet50'])
    save_model = torch.load('/Users/mianmaokuchuanma/database/model/resnet34.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and "num_batches_tracked" not in k}
    model_list1 = list(model_dict.keys())
    model_list2 = list(state_dict.keys())

    j, k = 0, 0
    while 1:
        if j >= len(model_list1) or k >= len(model_list2):
            break
        layer1, layer2 = model_list1[j], model_list2[k]
        if "num_batches_tracked" in layer1:
            j += 1
            layer1 = model_list1[j]
        print(layer1, model_dict[layer1].shape, end=' ')
        print(layer2, state_dict[layer2].shape)
        j += 1
        k += 1

    model_dict.update(state_dict)
    model.load_state_dict(model_dict, False)

    return model


def resnet101_backbone(**kwargs):
    # Constructs a ResNet-50 model_hyper.
    model = ResNetBackbone(Bottleneck, [3, 4, 23, 3], **kwargs)

    # load pre-trained weights
    # save_model = model_zoo.load_url(model_urls['resnet50'])
    save_model = torch.load('./model/resnet101.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # model_list1 = list(model_dict.keys())
    # model_list2 = list(state_dict.keys())
    # minlen = min(len(model_list1), len(model_list2))
    # for i in range(minlen):
    #     # print(model_list1[i], end=' ')
    #     print(model_list2[i])
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model