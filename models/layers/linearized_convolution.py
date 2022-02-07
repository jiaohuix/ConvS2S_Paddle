# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import paddle
import paddle.nn.functional as F
import collections
from .conv_tbc import ConvTBC
from .conv_nlc import ConvNLC
from typing import Dict, Optional
from paddle import Tensor

class LinearizedConvolutionV1(ConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """
    Cache = collections.namedtuple("Cache", ["input_buffer"])

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._linearized_weight = None
        # self.register_backward_hook(self._clear_linearized_weight) # 这个干嘛的？？？？？？？？？

    def forward(self, input, cache= None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Time x Batch x Channel during training
            Batch x Time x Channel during inference
        """
        assert cache is None or isinstance(cache,self.Cache),'cache must be none or instance of Cache'
        if cache is None:
            output = self.conv_tbc(input)
            if self.kernel_size[0] > 1 and self.padding[0] > 0:
                # remove future timesteps added by padding
                output = output[: -self.padding[0], :, :] #tbc
                # output = output[:, : , :-self.padding[0]] #bct
            return output

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        bsz = input.shape[0]  # input: bsz x len x dim
        if kw > 1:
            # input = input.data # 取无梯度的值tensor，以后考虑把梯度关掉
            input_buffer = cache.input_buffer # 与cache的input_buffer共享,修改了input_buffer,cache中的也会改变
            if not (input_buffer==0).all(): # 非初始化为0
                # shift buffer
                input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
            # append next input
            input_buffer[:, -1, :] = input[:, -1, :]
            input = input_buffer
        with paddle.no_grad():
            output = F.linear(input.reshape((bsz,-1)), weight, self.bias).reshape((bsz, 1, -1))
        return output,cache

    def gen_cache(self,memory):
        ''' 按memory的bsz产生cache '''
        bsz,k,dim=memory.shape[0],self.kernel_size[0],self.in_channels
        if self.kernel_size[0]>1:
            buffer=paddle.zeros(shape=[bsz,k,dim],dtype=memory.dtype)
            buffer.stop_gradient = True
        else:
            buffer=None
        return self.Cache(input_buffer=buffer)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # [k in out]->[k*in out] conv_tbc
            # weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            # [out in k]->[k*in out] conv1d
            weight = self.weight.transpose((2,1,0)) # 注意！[k*in out]!= [in*k out]
            # assert weight.shape == [self.out_channels, kw, self.in_channels]
            assert weight.shape == [kw, self.in_channels,self.out_channels]
            # return weight.reshape((self.out_channels,-1))
            return weight.reshape((-1, self.out_channels))
        return self._linearized_weight


class LinearizedConvolution(ConvNLC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """
    Cache = collections.namedtuple("Cache", ["input_buffer"])

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._linearized_weight = None
        # self.register_backward_hook(self._clear_linearized_weight) # 这个干嘛的？？？？？？？？？

    def forward(self, input, cache= None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Batch x Time x Channel during training and inference
        """
        assert cache is None or isinstance(cache,self.Cache),'cache must be none or instance of Cache'
        if cache is None:
            output = self.conv_nlc(input)
            if self.kernel_size[0] > 1 and self.padding[0] > 0:
                # remove future timesteps added by padding
                # output = output[: -self.padding[0], :, :] #tbc
                output = output[:, :-self.padding[0],:] # nlc
            return output

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        bsz = input.shape[0]  # input: bsz x len x dim
        if kw > 1:
            input_buffer = cache.input_buffer # 与cache的input_buffer共享,修改了input_buffer,cache中的也会改变
            if not (input_buffer==0).all(): # 非初始化为0
                # shift buffer
                input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
            # append next input
            input_buffer[:, -1, :] = input[:, -1, :]
            input = input_buffer
        with paddle.no_grad():
            output = F.linear(input.reshape((bsz,-1)), weight, self.bias).reshape((bsz, 1, -1))
        return output,cache

    def gen_cache(self,memory):
        ''' 按memory的bsz产生cache '''
        bsz,k,dim=memory.shape[0],self.kernel_size[0],self.in_channels
        if self.kernel_size[0]>1:
            buffer=paddle.zeros(shape=[bsz,k,dim],dtype=memory.dtype)
            buffer.stop_gradient = True
        else:
            buffer=None
        return self.Cache(input_buffer=buffer)

    def _get_linearized_weight(self): # 这里待会必须研究！
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # [k in out]->[k*in out] conv_tbc
            # weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            # [out in k]->[k*in out] conv1d
            weight = self.weight.transpose((2,1,0)) # 注意！[k*in out]!= [in*k out]
            # assert weight.shape == [self.out_channels, kw, self.in_channels]
            assert weight.shape == [kw, self.in_channels,self.out_channels]
            # return weight.reshape((self.out_channels,-1))
            return weight.reshape((-1, self.out_channels))
        return self._linearized_weight

    # def _get_linearized_weight(self): # 这里待会必须研究！
    #     if self._linearized_weight is None:
    #         kw = self.kernel_size[0]
    #         # [k in out]->[k*in out] conv_tbc
    #         # weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
    #         # [out in k]->[k*in out] conv1d
    #         # [out in k]->[in * k ,out] conv1d
    #         weight = self.weight.transpose((1,2,0))
    #         # assert weight.shape == [self.out_channels, kw, self.in_channels] # 1
    #         # assert weight.shape == [kw, self.in_channels,self.out_channels] # 2
    #         assert weight.shape == [ self.in_channels,kw,self.out_channels] # 3
    #         # return weight.reshape((self.out_channels,-1))
    #         return weight.reshape((-1, self.out_channels))
    #     return self._linearized_weight