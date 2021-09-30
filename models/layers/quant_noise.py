# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import paddle
import paddle.nn as nn

def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Layer
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don'align_norm register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2D))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.shape[1] % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module._kernel_size == [1, 1]: # 这里就不对
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module._kernel_size[0] * module._kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.shape[1]
                out_features = weight.shape[0]

                # split weight matrix into blocks and randomly drop selected blocks
                mask = paddle.zeros(shape=[in_features // block_size * out_features])
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).reshape((-1, in_features))

            else: # 卷积
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod._kernel_size == [1, 1]:
                    mask = paddle.zeros(shape=[int(in_channels // block_size * out_channels)])
                    mask.bernoulli_(p)
                    ### ????????????
                    mask = mask.repeat_interleave(block_size, -1).reshape((-1, in_channels))
                else:
                    mask = paddle.zeros(shape=weight.shape)
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod._kernel_size[0], mod._kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                paddle.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0) # data是获取没有梯度信息的tensor而不是var

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


'''
kernel_size-> _kernel_size  ()->[]
mask
bernoulli_
to
hook
masked_fill
weight.data
repeat
repeat_interleave
'''