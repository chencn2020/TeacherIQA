import torch.nn as nn
import torch
from model import teacherNetwork as TN


class selfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(selfAttention, self).__init__()

        self.qConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.kConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.vConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inFeature):
        bs, C, w, h = inFeature.size()
        
        proj_query = self.qConv(inFeature).view(bs, -1, w * h).permute(0, 2, 1)
        proj_key = self.kConv(inFeature).view(bs, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.vConv(inFeature).view(bs, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, w, h)

        out = self.gamma * out + inFeature

        return out

class enc(nn.Module):
    def __init__(self, ch):
        super(enc, self).__init__()
        self.KI = selfAttention(ch)

    def forward(self, KL, distortionKL):
        fusionKL = torch.cat((KL, distortionKL), dim=1)
        KIRes = self.KI(fusionKL)
        return KIRes

class StudentNetwork(nn.Module):
    def __init__(self):
        super(StudentNetwork, self).__init__()
        
        self.tn = TN.TeacherNetwork(3).cuda()

        self.bottomConv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.upMC1Conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )

        self.upMC2Conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4, 4),
        )

        self.upMC3Conv = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8, 8),
        )

        self.enc1 = enc(512)
        self.enc2 = enc(1024)
        self.enc3 = enc(2048)

        self.iqaScore = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1),  # 24 24
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),  # 1 8 8
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),  # 1 8 8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, kernel_size=1, stride=1),  # 1 8 8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=7, stride=1),  # 1 10 10
        )

    def forward(self, img):
        _, semanticKL, [distortionKL1, distortionKL2, distortionKL3] = self.tn(img)
        resNetBottomFeature = self.bottomConv(semanticKL)  # n, 32, 28, 28

        distortionKL1 = self.upMC1Conv(distortionKL1)  # n, 32, 28, 28
        distortionKL2 = self.upMC2Conv(distortionKL2)  # n, 32, 28, 28
        distortionKL3 = self.upMC3Conv(distortionKL3)  # n, 32, 28, 28

        attention1 = self.enc1(resNetBottomFeature, distortionKL1)
        attention2 = self.enc2(attention1, distortionKL2)
        attention3 = self.enc3(attention2, distortionKL3)

        return self.iqaScore(attention3).view(img.shape[0])

if __name__ == '__main__':
    net = StudentNetwork().cuda()
    inputs = torch.zeros((1, 3, 224, 224), dtype=torch.float32).cuda()
    output = net(inputs)
    print(output.size())
