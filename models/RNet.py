import torch.nn as nn
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
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class RNet(nn.Module):
    def __init__(self, num_bands, num_classes):
        super().__init__()

        # self.encoder = smp.encoders.get_encoder('resnet18', in_channels=num_bands, depth=5, weights=None)
        # self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
        #                           nn.BatchNorm2d(128),
        #                           nn.ReLU())
        # self.dec = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(128, momentum=BN_MOMENTUM), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(32, momentum=BN_MOMENTUM), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(16, momentum=BN_MOMENTUM), nn.ReLU(),
        #                          nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
        #                          nn.BatchNorm2d(16, momentum=BN_MOMENTUM), nn.ReLU())
        # self.clf = nn.Conv2d(16, num_classes, kernel_size=1)
        # self.ctx = nn.Sequential(dilated_conv(n_convs=2, in_channels=num_classes, out_channels=num_classes, dilation=1),
        #                          dilated_conv(n_convs=1, in_channels=num_classes, out_channels=num_classes, dilation=2),
        #                          dilated_conv(n_convs=1, in_channels=num_classes, out_channels=num_classes, dilation=4),
        #                          dilated_conv(n_convs=1, in_channels=num_classes, out_channels=num_classes, dilation=8),
        #                          dilated_conv(n_convs=1, in_channels=num_classes, out_channels=num_classes,
        #                                       dilation=16),
        #                          dilated_conv(n_convs=1, in_channels=num_classes, out_channels=num_classes, dilation=1),
        #                          nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0))
        #
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

        initialize_weights(self.dec, self.clf, self.ctx)

    def forward(self, x):
        # x_enc = self.encoder(x)
        # x_h = self.head(x_enc[-1])
        # x_d = self.dec(x_h)
        # x_c = self.clf(x_d)
        # x_out = self.ctx(x_c)
        x_size = x.size()
        x = self.layer0(x)  # scale:1/2, 32
        x = self.layer1(x)  # scale:1/2, 64
        x = self.layer2(x)  # scale:1/4, 128
        x = self.layer3(x)  # scale:1/8, 256
        x = self.layer4(x)  # scale:1/8, 512
        x = self.head(x)

        out = self.dec(x)
        out = self.ctx(out)
        # out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
        out = self.clf(out)

        return out

# def dilated_conv(n_convs, in_channels, out_channels, dilation):
#     layers = []
#     for i in range(n_convs):
#         layers.append(nn.ZeroPad2d(dilation))
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation))
#         layers.append(nn.ReLU(inplace=True))
#     return nn.Sequential(*layers)
#
#
# class contextModuleS(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         self.ctx1_s = dilated_conv(n_convs=2, in_channels=in_channels, out_channels=out_channels, dilation=1)
#         self.ctx2_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=2)
#         self.ctx3_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=4)
#         self.ctx4_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=8)
#         self.ctx5_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=16)
#         self.ctx7_s = dilated_conv(n_convs=1, in_channels=in_channels, out_channels=out_channels, dilation=1)
#         self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         ctx1_s = self.ctx1_s(x)
#         ctx2_s = self.ctx2_s(ctx1_s)
#         ctx3_s = self.ctx3_s(ctx2_s)
#         ctx4_s = self.ctx4_s(ctx3_s)
#         ctx5_s = self.ctx5_s(ctx4_s)
#         ctx7_s = self.ctx7_s(ctx5_s)
#         conv_s = self.conv_s(ctx7_s)
#
#         return conv_s
#
#
# class contextModuleL(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#
#         self.ctx0_l = dilated_conv(n_convs=1, in_channels=in_ch, out_channels=out_ch, dilation=1)
#         self.ctx1_l = dilated_conv(n_convs=1, in_channels=in_ch, out_channels=out_ch * 2, dilation=1)
#         self.ctx2_l = dilated_conv(n_convs=1, in_channels=in_ch * 2, out_channels=out_ch * 4, dilation=2)
#         self.ctx3_l = dilated_conv(n_convs=1, in_channels=in_ch * 4, out_channels=out_ch * 8, dilation=4)
#         self.ctx4_l = dilated_conv(n_convs=1, in_channels=in_ch * 8, out_channels=out_ch * 16, dilation=8)
#         self.ctx5_l = dilated_conv(n_convs=1, in_channels=in_ch * 16, out_channels=out_ch * 32, dilation=16)
#         self.ctx7_l = dilated_conv(n_convs=1, in_channels=in_ch * 32, out_channels=out_ch * 32, dilation=1)
#         self.conv_l = nn.Conv2d(in_ch * 32, out_ch, kernel_size=1, padding=0)
#
#     def forward(self, x):
#         ctx0_l = self.ctx0_l(x)
#         ctx1_l = self.ctx1_l(ctx0_l)
#         ctx2_l = self.ctx2_l(ctx1_l)
#         ctx3_l = self.ctx3_l(ctx2_l)
#         ctx4_l = self.ctx4_l(ctx3_l)
#         ctx5_l = self.ctx5_l(ctx4_l)
#         ctx7_l = self.ctx7_l(ctx5_l)
#         conv_l = self.conv_l(ctx7_l)
#
#         return conv_l
#
#
# class ContextualUnet(nn.Module):
#
#     def __init__(self, num_bands, num_channels):
#         super().__init__()
#
#         model = smp.Unet(encoder_name="resnext50_32x4d",
#                          encoder_weights="imagenet",
#                          encoder_depth=5,
#                          in_channels=num_bands,
#                          classes=num_channels,
#                          activation=None)
#         # self.e_ch = model.encoder.out_channels[-1]
#         self.d_ch = model.decoder.blocks[-1].conv2[0].out_channels
#         self.model = model
#         self.ctx_s = contextModuleS(num_bands, num_bands)
#         self.ctx_l = contextModuleL(self.d_ch, self.d_ch)
#
#     def forward(self, x):
#
#         s_ctx = self.ctx_s(x)
#         num_features = self.model.encoder(s_ctx)
#         # num_features[-1] = s_ctx
#         decoder_output = self.model.decoder(*num_features)
#         l_ctx = self.ctx_l(decoder_output)
#         masks = self.model.segmentation_head(l_ctx)
#
#         return masks
