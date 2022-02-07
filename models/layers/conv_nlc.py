# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
直接用conv1d的nlc格式，如此卷积或注意力不用频繁转置
'''
import paddle
from paddle import nn
from models.utils import _single
from paddle import Tensor
import paddle.nn.functional as F
from paddle.nn.initializer import XavierNormal,Constant
xavier_normal_=XavierNormal()
zeros_=Constant(value=0.)

class ConvNLC(nn.Layer):
    """1D convolution over an input of shape (batch x length x channel),neither convolution nor attention requires frequent transposes
       Speed up around 20%!
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvNLC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.weight = self.create_parameter(shape=[out_channels, in_channels,self.kernel_size[0]]) #mck
        self.bias = self.create_parameter(shape=[out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal_(self.weight)
        zeros_(self.bias)

    def conv_nlc(self, input: Tensor):
        ''' input: [N L C]'''
        return F.conv1d(input,self.weight,self.bias,padding=self.padding[0],data_format='NLC')

    def forward(self, input: Tensor):
        return self.conv_nlc(input)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
