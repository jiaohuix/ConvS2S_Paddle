import argparse
import contextlib
import copy
import importlib
import logging
import os
import sys
import warnings
from itertools import accumulate
import collections.abc
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
import paddle
import paddle.nn.functional as F
from paddle import Tensor
from itertools import repeat

container_abcs = collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx] # 迭代词汇表词
        if token in embed_dict: # 查嵌入字典
            # embedding.weight.data[idx] = embed_dict[token] # 加载进embedding里，此处必错 没有data
            embedding.weight[idx] = embed_dict[token] # 加载进embedding里
    return embedding

def make_positions(tensor, padding_idx: int):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn'align_norm know
    # how to handle the dtype kwarg in cumsum.
    mask= (tensor!=padding_idx).astype(dtype=paddle.int32) # 非pad为1，pad为0
    cumsum=paddle.cumsum(mask, axis=1)
    cumsum=paddle.where(mask==False,paddle.zeros_like(cumsum),cumsum) # 把pad部分的累加掩为0
    return cumsum.astype(dtype=paddle.int64) + padding_idx


def set_incremental_state(
    module: "MultiheadAttention",
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
    value: Dict[str, Optional[Tensor]],
) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        result = module.set_incremental_state(incremental_state, key, value)
        if result is not None:
            incremental_state = result
    return incremental_state

