import paddle
from paddle import nn as nn
import math
import x2paddle
from x2paddle import torch2paddle
from paddleseg.cvlibs import param_init


class ConvLayer(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.module1 = nn.Sequential(
            nn.ReLU(), 
            nn.Conv2D(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias_attr=False))

    def forward(self, x):
        return self.module1(x)


class ResidualUnit(nn.Layer):

    def __init__(self, num_features):
        super(ResidualUnit, self).__init__()
        self.module2 = nn.Sequential(
            ConvLayer(num_features, num_features),
            ConvLayer(num_features, num_features))

    def forward(self, h0, x):
        return h0 + self.module2(x)


class RecursiveBlock(nn.Layer):

    def __init__(self, in_channels, out_channels, U):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.h0 = ConvLayer(in_channels, out_channels)
        self.ru = ResidualUnit(out_channels)

    def forward(self, x):
        h0 = self.h0(x)
        x = h0
        for i in range(self.U):
            x = self.ru(h0, x)
        return x


class DRRN(nn.Layer):

    def __init__(self, B, U, num_channels=1, num_features=128):
        super(DRRN, self).__init__()
        self.rbs = nn.Sequential(*[RecursiveBlock(num_channels if i == 0 else
            num_features, num_features, U) for i in range(B)])
        self.rec = ConvLayer(num_features, num_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, paddle.nn.Conv2D):
                param_init.normal_init(m.weight.data,mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                
                if m.bias is not None:
                    torch2paddle.constant_init_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = self.rbs(x)
        x = self.rec(x)
        x += residual
        return x
