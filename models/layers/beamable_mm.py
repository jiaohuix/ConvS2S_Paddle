# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

'''
fold 和unfold的算子,尚未完成。突然发现只要unfold，且只用来reshape
已完成unfold1d(只能axis=0)
'''
# dim [a,b,c],->如果axis=0,shape=[a//beam,b,c,beam]
def unfold1d_asis0(tensor,beam):
    ''' tensor :[bsz ... nhu] '''
    src_shape=tensor.shape
    tgt_shape=src_shape+[beam]
    tgt_shape[0]=src_shape[0]//beam
    # bsz x 1 x nhu --> bsz/beam x beam x nhu
    bsz=tensor.shape[0]
    tensor=tensor.reshape((bsz,-1)).unsqueeze((0,1)) # 1 x 1 x bsz x nhu
    nhu=tensor.shape[-1]
    tensor = F.unfold(tensor, kernel_sizes=[beam, nhu], strides=beam)  # 1 x beam*nhu x bsz//beam   axis=1【1 x bsz*beam x nhu//beam】不用transpose
    tensor = tensor.transpose((0, 2, 1)) # 1 x bsz//beam x beam*nhu*channel(1)
    tensor = tensor.reshape((bsz // beam, beam,nhu)) #  bsz/beam x beam x nhu
    tensor = tensor.transpose((0,2,1)).reshape(tgt_shape)
    return tensor

# 3dim的aixs=1未通过。
def unfold1d(tensor,beam,axis=0):
    ''' tensor :[bsz ... nhu] '''
    # tensor 2dim ,kernerl ,tmp_shape 3处需要按axis变化，明天需要测试axis0 1在2dim和多维的效果是否一样
    assert axis==0 or axis==1 ,'axis=0 or 1'
    src_shape=tensor.shape
    tgt_shape=src_shape+[beam]
    tgt_shape[axis]=src_shape[axis]//beam

    # bsz x ... x nhu --> bsz/beam x beam x nhu 【axis=0】
    bsz,nhu=tensor.shape[0],tensor.shape[-1]
    tmp_shape=[bsz,-1] if axis==0 else  [-1,nhu]
    tensor=tensor.reshape(tmp_shape).unsqueeze((0,1)) # 1 x 1 x bsz x nhu # 这也要变@@@@@@@@@@@@
    nhu=tensor.shape[-1]
    kernel_size=[beam,nhu] if axis==0 else [bsz,beam]
    tensor = F.unfold(tensor, kernel_sizes=kernel_size, strides=beam)  # 1 x beam*nhu x bsz//beam   axis=1【1 x bsz*beam x nhu//beam】不用transpose
    tmp_shape = [bsz ,beam, nhu// beam] # axis=1
    if axis==0:
        tensor = tensor.transpose((0, 2, 1)) # 1 x bsz//beam x beam*nhu*channel(1)
        tmp_shape=[bsz//beam,beam,nhu]
    tensor = tensor.reshape(tmp_shape) #  bsz/beam x beam x nhu
    tensor = tensor.transpose((0,2,1))
    tensor=tensor.reshape(tgt_shape)
    return tensor

class BeamableMM(nn.Layer):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """

    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if (
            not self.training  # eval时候才会开启
            and self.beam_size is not None  # test mode
            and input1.dim() == 3  # beam size is set
            and input1.shape[1]  # only support batched input
            == 1  # single time step update
        ):
            bsz,nhu,beam = input1.shape[0],input1.shape[-1], self.beam_size

            # bsz x 1 x nhu --> bsz/beam x beam x nhu
            input1=unfold1d_asis0(input1[:,0,:],beam) # bsz/beam x nhu x beam
            input1 = input1.transpose((0,2,1)) #1 5 128
            # bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
            input2=unfold1d_asis0(input2,beam)[:, :, :, 0]
            # input2 = input2.unfold(0, beam, beam)[:, :, :, 0]

            # use non batched operation if bsz = beam
            if input1.shape[0] == 1:
                output = paddle.mm(input1[0, :, :], input2[0, :, :])
            else:
                output = input1.bmm(input2)
            return output.reshape((bsz, 1, -1))
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

if __name__ == '__main__':
    beam=BeamableMM(beam_size=5)
    beam.eval()
    print(beam.training)
    x1=paddle.randn((5,1,128)) # tgt # 1 128 5
    x2=paddle.randn((5,128,20)) # src # 1 128 20 5
    import time
    t1=time.time()
    o=beam(x1,x2)
    t2=time.time()
    print(o.shape)
    print(f'exe:{(t2 - t1) * 1000} ms')

    # 测试unfold1d 2dim
    # x=torch.randn(3,4) # [1 4 2]  / [3,2,2]
    # x2=paddle.to_tensor(x.numpy())
    # print(x.unfold(1,2,2))
    # print(unfold1d(x2,2,axis=1)) # axis=0 or 1

    # 测试unfold1d 3dim
    # x=torch.randn(3,5,4) # [1 5 4 2] / [3 2 4 2]
    # x2=paddle.to_tensor(x.numpy())
    # print(x.unfold(1,2,2))
    # print(unfold1d(x2,2,axis=1)) # axis=0 or 1