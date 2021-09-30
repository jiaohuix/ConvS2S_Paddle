# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
未完成：conv_tbc c++算子
'''
import paddle
from paddle import nn
from models.utils import _single
from paddle import Tensor
import paddle.nn.functional as F
from paddle.nn.initializer import XavierNormal,Constant
xavier_normal_=XavierNormal()
zeros_=Constant(value=0.)

class ConvTBC(nn.Layer):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)

        # self.weight = self.create_parameter(shape=[self.kernel_size[0], in_channels, out_channels]) #mck
        self.weight = self.create_parameter(shape=[out_channels, in_channels,self.kernel_size[0]]) #mck
        self.bias = self.create_parameter(shape=[out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal_(self.weight)
        zeros_(self.bias)

    # 待修正，需要不出tbc算子！，但是可以不用在encoder、decoder里显示转换bct
    def conv_tbc(self, input: Tensor):
        # return paddle.conv_tbc(
        #     input.contiguous(), self.weight, self.bias, self.padding[0]
        # )
        # input:tbc->bct
        return F.conv1d(input.transpose((1,2,0)),self.weight,self.bias,padding=self.padding[0]).transpose((2,0,1))
        # return F.conv1d(input,self.weight,self.bias,padding=self.padding[0])

    def forward(self, input: Tensor):
        return self.conv_tbc(input)


    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
