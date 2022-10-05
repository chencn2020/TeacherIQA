import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet50_backbone(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
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
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        bottom = self.layer4(x3)

        return x, x1, x2, x3, bottom


class TripleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TripleConv, self).__init__()
        hide_ch = out_ch // 2
        self.TripleConv = nn.Sequential(
            nn.Conv2d(in_ch, hide_ch, 3, padding=1, groups=hide_ch),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, hide_ch, 3, padding=1, groups=hide_ch),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, out_ch, 3, padding=1, groups=hide_ch),
        )

    def forward(self, x):
        return self.TripleConv(x)


class Inception(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inception, self).__init__()

        out = out_ch // 4
        hide_ch = out // 2

        self.p1 = nn.Sequential(
            nn.Conv2d(in_ch, out, kernel_size=1),
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(in_ch, hide_ch, 1),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, out, 5, padding=2, groups=hide_ch),
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_ch, hide_ch, 1),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, hide_ch, 3, padding=1, groups=hide_ch),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, out, 1)
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, out, kernel_size=1),
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)

        return torch.cat((p1, p2, p3, p4), dim=1)


class InceptionConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InceptionConv, self).__init__()
        self.inception = nn.Sequential(
            Inception(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.tripleConv = nn.Sequential(
            TripleConv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inceptionRes = self.inception(x)
        return self.tripleConv(inceptionRes)


class MC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MC, self).__init__()
        self.MCConv = InceptionConv(in_ch, out_ch)

    def pad(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1

    def forward(self, encoderFeature, decoderFeature):
        encoderFeature = self.pad(encoderFeature, decoderFeature)
        merge = torch.cat([encoderFeature, decoderFeature], dim=1)
        return self.MCConv(merge)


class TeacherNetwork(nn.Module):
    def __init__(self, out_ch, pretrainedResnet=True):
        super(TeacherNetwork, self).__init__()

        self.resNet = resnet50_backbone(pretrained=pretrainedResnet)

        self.bottom = InceptionConv(2048, 2048)

        # up
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, 2)
        self.upMC1 = MC(1024 * 2, 1024)  # 14 * 14

        self.up2 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.upMC2 = MC(512 * 2, 512)  # 28 * 28

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upMC3 = MC(256 * 2, 256)  # 56 * 56

        self.up4 = nn.ConvTranspose2d(256, 64, 2, 2)  # 112 * 112
        self.upMC4 = MC(64 * 2, 64)  # 56 * 56

        self.up5 = nn.ConvTranspose2d(64, 64, 2, 2)  # 224 * 224
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # down
        encoderFeature0, encoderFeature1, encoderFeature2, encoderFeature3, bottom = self.resNet(x)
        bottom = self.bottom(bottom)

        # up
        decoderFeature1 = self.up1(bottom)  # 14 * 14
        upMC1 = self.upMC1(decoderFeature1, encoderFeature3)

        decoderFeature2 = self.up2(upMC1)
        upMC2 = self.upMC2(decoderFeature2, encoderFeature2)  # 28 * 28

        decoderFeature3 = self.up3(upMC2)
        upMC3 = self.upMC3(decoderFeature3, encoderFeature1)  # 56 * 56

        decoderFeature4 = self.up4(upMC3)
        upMC4 = self.upMC4(decoderFeature4, encoderFeature0)  # 56 * 56

        up5 = self.up5(upMC4)
        out = self.out(up5)

        return nn.Sigmoid()(out), bottom, [upMC1, upMC2, upMC3]


if __name__ == '__main__':
    net = TeacherNetwork(3)
    inputs = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    output = net(inputs)
    print(output[0].size())
