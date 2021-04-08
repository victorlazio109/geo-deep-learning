import torch.nn as nn
import torch
from torchvision import models
from torch.nn import functional as F

# Context Module Implementation as described here: https://arxiv.org/abs/1511.07122

BN_MOMENTUM = 0.01


def dilated_conv(n_convs, in_channels, out_channels, dilation):
    layers = []
    for i in range(n_convs):
        layers.append(nn.ZeroPad2d(dilation))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class RefUnet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv0 = nn.Conv2d(in_ch, in_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(32)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(32 + 16, 16, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(16)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(16, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))  # scale:1
        hx = self.pool1(hx1)  # scale:1/2

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)  # scale:1/4

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)  # scale:1/8

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.upscore2(hx4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual



class RNet(nn.Module):
    def __init__(self, num_bands, num_classes, inference=None):
        super().__init__()
        self.inference = inference
        resnet = models.resnet18(pretrained=True)
        newconv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_bands > 3:
            newconv1.weight.data[:, 3:num_bands, :, :].copy_(resnet.conv1.weight.data[:, 0:num_bands - 3, :, :])
        else:
            newconv1.weight.data[:, 0:num_bands, :, :].copy_(resnet.conv1.weight.data[:, 0:num_bands, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(32, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=0.01), nn.ReLU())
        self.clf = nn.Conv2d(16, num_classes, kernel_size=1)
        self.ctx = nn.Sequential(dilated_conv(n_convs=2, in_channels=16, out_channels=16, dilation=1),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=2),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=4),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=8),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16,
                                              dilation=16),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=1),
                                 nn.Conv2d(16, 16, kernel_size=1, padding=0))

        self.Ref = RefUnet(num_classes)

        initialize_weights(self.dec, self.clf, self.ctx, self.Ref)

    def forward(self, x):
        # x_size = x.size()
        x = self.layer0(x)  # scale:1/2, 32
        x = self.layer1(x)  # scale:1/2, 64
        x = self.layer2(x)  # scale:1/4, 128
        x = self.layer3(x)  # scale:1/8, 256
        x = self.layer4(x)  # scale:1/8, 512
        x = self.head(x)

        out = self.dec(x)
        out = self.ctx(out)
        # out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
        out_seg = self.clf(out)
        out_rf = self.Ref(out_seg)
        if self.inference:
            return out_rf
        else:
            return out_rf, out_seg


class RNet50_WRN(nn.Module):
    def __init__(self, num_bands, num_classes, inference=None):
        super().__init__()
        self.inference = inference
        resnet = models.wide_resnet50_2(pretrained=True)
        newconv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_bands > 3:
            newconv1.weight.data[:, 3:num_bands, :, :].copy_(resnet.conv1.weight.data[:, 0:num_bands - 3, :, :])
        else:
            newconv1.weight.data[:, 0:num_bands, :, :].copy_(resnet.conv1.weight.data[:, 0:num_bands, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # self.head = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False),
        #                           nn.BatchNorm2d(512, momentum=0.95),
        #                           nn.ReLU())
        # self.dec = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(256, momentum=0.01), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(128, momentum=0.01), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(16, momentum=0.01), nn.ReLU(),)
        # nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
        # nn.BatchNorm2d(16, momentum=0.01), nn.ReLU())

        # self.head = nn.Sequential(nn.BatchNorm2d(2048, momentum=0.95),
        #                           nn.ReLU(),
        #                           nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False))

        self.dec = nn.Sequential(nn.BatchNorm2d(2048, momentum=0.01),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=2, stride=2),

                                 nn.BatchNorm2d(1024, momentum=0.01),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),

                                 nn.BatchNorm2d(512, momentum=0.01),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),

                                 nn.BatchNorm2d(256, momentum=0.01),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=2, stride=2), )

        # nn.BatchNorm2d(32, momentum=0.01),
        # nn.ReLU(),
        # nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2))

        self.ctx = nn.Sequential(dilated_conv(n_convs=2, in_channels=64, out_channels=64, dilation=1),
                                 dilated_conv(n_convs=1, in_channels=64, out_channels=64, dilation=2),
                                 dilated_conv(n_convs=1, in_channels=64, out_channels=64, dilation=4),
                                 dilated_conv(n_convs=1, in_channels=64, out_channels=64, dilation=8),
                                 dilated_conv(n_convs=1, in_channels=64, out_channels=64,
                                              dilation=16),
                                 dilated_conv(n_convs=1, in_channels=64, out_channels=64, dilation=1))

        self.clf = nn.Conv2d(64, num_classes, kernel_size=1)

        self.Ref = RefUnet(num_classes)

        initialize_weights(self.dec, self.clf, self.ctx, self.Ref)

    def forward(self, x):
        x = self.layer0(x)  # scale:1/2, 32
        # print(x.size())
        x = self.layer1(x)  # scale:1/2, 64
        # print(x.size())
        x = self.layer2(x)  # scale:1/4, 128
        # print(x.size())
        x = self.layer3(x)  # scale:1/8, 256
        # print(x.size())
        x = self.layer4(x)  # scale:1/8, 512
        # print(x.size())
        # x = self.head(x)
        # print(x.size())

        out = self.dec(x)
        # print(out.shape)
        out = self.ctx(out)
        # out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
        out_seg = self.clf(out)
        out_rf = self.Ref(out_seg)
        if self.inference:
            return out_rf
        else:
            return out_rf, out_seg


class RNet_32(nn.Module):
    def __init__(self, num_bands, num_classes, inference=None):
        super().__init__()
        self.inference = inference
        resnet = models.resnet34(pretrained=True)
        newconv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_bands > 3:
            newconv1.weight.data[:, 3:num_bands, :, :].copy_(resnet.conv1.weight.data[:, 0:num_bands - 3, :, :])
        else:
            newconv1.weight.data[:, 0:num_bands, :, :].copy_(resnet.conv1.weight.data[:, 0:num_bands, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(32, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=0.01), nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
                                 nn.BatchNorm2d(16, momentum=0.01), nn.ReLU())
        self.clf = nn.Conv2d(16, num_classes, kernel_size=1)
        self.ctx = nn.Sequential(dilated_conv(n_convs=2, in_channels=16, out_channels=16, dilation=1),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=2),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=4),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=8),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16,
                                              dilation=16),
                                 dilated_conv(n_convs=1, in_channels=16, out_channels=16, dilation=1),
                                 nn.Conv2d(16, 16, kernel_size=1, padding=0))

        self.Ref = RefUnet(num_classes)

        initialize_weights(self.dec, self.clf, self.ctx, self.Ref)

    def forward(self, x):
        # x_size = x.size()
        x = self.layer0(x)  # scale:1/2, 32
        # print(x.size())
        x = self.layer1(x)  # scale:1/2, 64
        # print(x.size())
        x = self.layer2(x)  # scale:1/4, 128
        # print(x.size())
        x = self.layer3(x)  # scale:1/8, 256
        # print(x.size())
        x = self.layer4(x)  # scale:1/8, 512
        # print(x.size())
        x = self.head(x)
        # print(x.size())

        out = self.dec(x)
        # print(out.shape)
        out = self.ctx(out)
        # out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
        out_seg = self.clf(out)
        out_rf = self.Ref(out_seg)
        if self.inference:
            return out_rf
        else:
            return out_rf, out_seg
