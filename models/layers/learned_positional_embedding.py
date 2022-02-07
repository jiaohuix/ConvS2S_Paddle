# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models import utils
from paddle import Tensor
from paddle.nn.initializer import Constant

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx) # 继承父类并初始化
        self.onnx_trace = False
        if self._padding_idx is not None:
            self.max_positions = self._num_embeddings - self._padding_idx - 1
        else:
            self.max_positions = self._num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None, # 默认，不用管
    ):
        """Input is expected to be of size [bsz x seqlen]. 输入的是一个batch的词id序列"""
        assert (positions is None) or (
            self._padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None: # 增量
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn'align_norm work in some cases when exporting to ONNX
                const_=Constant(value=int(self._padding_idx + input.shape[1]))
                positions=paddle.zeros((1,1))
                const_(positions)
                positions=positions.astype(paddle.int64)
            else: # 正常嵌入
                positions = utils.make_positions(input, self._padding_idx)

        return F.embedding(
            positions,
            self.weight,
            # padding_idx=self._padding_idx,
            sparse=self._sparse
            # 下面是torch的，用的是默认设置max_norm=None，在这个情况下和paddle一致
            # self.max_norm, # None
            # self.norm_type, #2
            # self.scale_grad_by_freq, # false
        )
