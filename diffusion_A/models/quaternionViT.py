# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuaternionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(QuaternionConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride=stride,
                                         padding=padding)
        self.conv_imag1 = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride=stride,
                                         padding=padding)
        self.conv_imag2 = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride=stride,
                                         padding=padding)
        self.conv_imag3 = nn.Conv2d(in_channels // 4, out_channels // 4, kernel_size, stride=stride,
                                        padding=padding)

    def quaternion_multiply(self, q1, q2):
        a1, b1, c1, d1 = q1
        a2, b2, c2, d2 = q2
        real = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
        imag1 = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
        imag2 = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
        imag3 = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
        return real, imag1, imag2, imag3

    def forward(self, x):
        x_real, x_imag1, x_imag2, x_imag3 = x.chunk(4, dim=1)
        #print('x_real',x_real.shape)
        # Real part convolutions
        conv_real = self.conv_real(x_real)
        #print('conv_real',conv_real.shape)
        # Imaginary part convolutions
        conv_imag1 = self.conv_imag1(x_imag1)
        #print('conv_imag1', conv_imag1.shape)
        conv_imag2 = self.conv_imag2(x_imag2)
        conv_imag3 = self.conv_imag3(x_imag3)

        # Quaternion multiplication (x_real, x_imag1, x_imag2, x_imag3) and (conv_real, conv_imag1, conv_imag2, conv_imag3)
        q1 = (x_real, x_imag1, x_imag2, x_imag3)
        q2 = (conv_real, conv_imag1, conv_imag2, conv_imag3)
        real, imag1, imag2, imag3 = self.quaternion_multiply(q1,q2)
        out = torch.cat((real, imag1, imag2, imag3), dim=1)
        return out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        out = x * channel_att * spatial_att
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()
        self.quaternion_conv = QuaternionConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x_quaternion = torch.cat((x,), dim=1)
        features = self.quaternion_conv(x_quaternion)
        features = self.cbam(features)
        features = features.view(batch_size, channels, height, width)
        return features

# if __name__ == '__main__':
#     q1 = random_tensor = torch.randn(82, 1024, 8, 8)
#     print('q1',q1.shape)
#     feature_extractor = FeatureExtractor(in_channels=1024, out_channels=1024)
#     q1 = feature_extractor(q1)
#     print('q2',q1.shape)
