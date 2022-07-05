import torch.nn as nn
import torch
import math

# resnet 50
# [1*1 + 3*3 + 1*1]
class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
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

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

# resnet18 / 34
# [3*3 + 3*3]
class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, replace_stride_with_dilation=[False, False, False]):
        # TODO 大写的win号
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # if replace_stride_with_dilation is None:
        #     replace_stride_with_dilation = [False, False, False]

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation

        # TODO 这个dilation又怎么算嘞
        # if dilate:
        #     self.dilation *= stride

        # TODO 又为什么要下采样嘞
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=previous_dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('x', x.dtype)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print("layer1=>", x.size())
        x = self.layer2(x)
        # print("layer2=>", x.size())
        x = self.layer3(x)
        # print("layer3=>", x.size())
        x = self.layer4(x)
        # print("layer4=>", x.size())

        return x


def resnet50_backbone(**kwargs):
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)

    # load pre-trained weights
    save_model = torch.load('./dataset/model/resnet50.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model


def resnet18_backbone(**kwargs):
    model = ResNetBackbone(BasicBlock, [2, 2, 2, 2], **kwargs)

    # load pre-trained weights
    save_model = torch.load('path/to/model')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model