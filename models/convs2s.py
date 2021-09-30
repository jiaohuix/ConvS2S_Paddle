# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import paddle
import logging
import numpy as np
import time
import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from models.layers import BeamableMM, AdaptiveSoftmax, PaddleseqDropout, \
    LearnedPositionalEmbedding, LinearizedConvolution, GradMultiply
from models import utils, EncoderDecoderModel, PaddleEncoder, PaddleDecoder
from paddle.fluid.layers.utils import map_structure

# from models import utils
logger = logging.getLogger("ConvS2S")
normal_ = Normal(0, 0.1)
zeros_ = Constant(value=0.)


class ConvS2SModel(nn.Layer):
    """
    A fully convolutional models, i.e. a convolutional encoder and a
    convolutional decoder, as described in `"Convolutional Sequence to Sequence
    Learning" (Gehring et al., 2017) <https://arxiv.org/abs/1705.03122>`_.

    Args:
        encoder (ConvS2SEncoder): the encoder
        decoder (ConvS2SDecoder): the decoder

    The Convolutional models provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.fconv_parser
        :prog:
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_src_positions,
                 max_tgt_positions,
                 encoder_layers,  # convolutions config
                 decoder_layers,
                 encoder_embed_dim,
                 decoder_embed_dim,
                 decoder_out_embed_dim,  # decoder conv完后把通道转成嵌入维度
                 dropout=0.1,
                 decoder_attention=True,
                 share_embed=False,
                 pad_id=1,
                 bos_id=0,
                 eos_id=2,
                 ):
        super().__init__()

        self.encoder = ConvS2SEncoder(
            src_vocab_size=src_vocab_size,
            embed_dim=encoder_embed_dim,
            max_positions=max_src_positions,
            convolutions=eval(encoder_layers),
            dropout=dropout,
            pad_id=pad_id,
        )
        self.decoder = ConvS2SDecoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=decoder_embed_dim,
            out_embed_dim=decoder_out_embed_dim,
            max_positions=max_tgt_positions,
            convolutions=eval(decoder_layers),
            attention=decoder_attention,
            dropout=dropout,
            share_embed=share_embed,
            positional_embeddings=True,
            adaptive_softmax_cutoff=None,
            adaptive_softmax_dropout=0.0,
            pad_id=1,
        )
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.encoder.num_attention_layers = sum(layer is not None for layer in self.decoder.attention)

    def forward(self, src_tokens, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder models.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any models-specific outputs
        """
        encoder_out = self.encoder(src_tokens, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any models-specific outputs
        """
        encoder_out = self.encoder(src_tokens, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the models."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class ConvS2SEncoder(PaddleEncoder):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.

    Args:
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(
            self,
            src_vocab_size,
            embed_dim=512,
            max_positions=1024,
            convolutions=((512, 3),) * 20,
            dropout=0.1,
            pad_id=1,
    ):
        super(ConvS2SEncoder, self).__init__(src_vocab_size, pad_id)
        self.dropout_module = PaddleseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.num_attention_layers = None
        self.vocab_size = src_vocab_size
        self.pad_id = pad_id
        self.embed_tokens = Embedding(self.vocab_size, embed_dim, self.pad_id)  # 词典V*768
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, self.pad_id)  # 1024*768
        # extend_conv_spec在spec为（c,k）时默认添加为（c,k,1）
        convolutions = extend_conv_spec(convolutions)  # (512,3,1)*20 #ochannel,kernel,residual
        in_channels = convolutions[0][0]  # 512
        # 把嵌入维度768转通道数512
        self.fc1 = Linear(embed_dim, in_channels)  # 768 512  要改！ok
        self.projections = nn.LayerList()  # 在conv时提升通道数
        self.convolutions = nn.LayerList()
        self.residuals = []
        # 构建conv和res
        layer_in_channels = [in_channels]  # 记录conv的所有输入通道数
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]  # 1的话岂不是最后一个通道，就是上一层的输出数,512
            self.projections.append(
                Linear(residual_dim, out_channels)  # 如果输入维度不等于输出维度，则把上一个输入映射到输出数。都是512则不会遇到这种情况
                if residual_dim != out_channels
                else None
            )
            # 填充
            if kernel_size % 2 == 1:  # 3时填充1
                padding = kernel_size // 2
            else:
                padding = 0  # 1时填充0
            # 门控1d卷积
            self.convolutions.append(  # 512 1024
                # 修改：用nlc替代tbc,卷积和注意力不需要频繁转置
                ConvNLC(
                    in_channels,
                    out_channels * 2,  # c->2c，经glu，变回c
                    kernel_size,
                    dropout=dropout,
                    padding=padding,  # 词数不变
                )
            )
            self.residuals.append(residual)  # 1或0（是否不一样？0不一样似乎）
            in_channels = out_channels  # 当前输出通道数变下一层输入数
            layer_in_channels.append(out_channels)
        # 把通道数2048转回嵌入维度768
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens):
        # x [bsz step embed_dim]
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`

        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **src_mask** (ByteTensor): the positions of
                  not padded elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        input_embedding = x  # 和输出z作残差连接为z+e

        # project to size of convolution
        x = self.fc1(x)
        # used to mask padding in input # 注意！变成是否是非pad，pad部分为0，否则1
        # mask！！！！！有可能要改哦
        src_mask = paddle.cast(src_tokens != self.pad_id,
                               dtype=paddle.get_default_dtype()).unsqueeze([-1])  # ->  B x T x 1
        src_mask.stop_gradient = True
        if not (src_tokens == self.pad_id).any():
            src_mask = None

        # B x T x C -> T x B x C
        # x = x.transpose((1, 0, 2))

        residuals = [x]  # 放输入的x
        # temporal convolutions
        for proj, conv, res_layer in zip(
                self.projections, self.convolutions, self.residuals
        ):
            if res_layer > 0:  # 1或0，作为索引
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)  # 如果维度不同（proj！=none），需要用proj投影
            else:
                residual = None  # 0表示不需要残差

            if src_mask is not None:
                x = x * src_mask  # 把pad的嵌入向量变为0

            x = self.dropout_module(x)
            # 卷积
            kernel_size = 3
            if kernel_size % 2 == 1:  # 奇数
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, [0, 0, padding_l, padding_r, 0, 0, ], data_format='NLC')
                x = conv(x)  # bsz 1024 step
            x = F.glu(x, axis=2)  # step bsz 512
            # 残差连接
            if residual is not None:
                x = (x + residual) * (0.5 ** 0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        # x = x.transpose((1, 0, 2))
        # project back to size of embedding
        x = self.fc2(x)

        if src_mask is not None:
            x = x * src_mask  # 把pad的特征向量变为0
            src_mask = src_mask.squeeze([-1])  # 莫把bsz=1时bsz挤掉

        # scale gradients (this only affects backward, not forward)
        if self.num_attention_layers is not None:
            scale = 1.0 / (2.0 * self.num_attention_layers)  # 缩放梯度
            x = GradMultiply.apply(x, scale)

        # add output to input embedding for attention 把输出加输入
        y = (x + input_embedding) * (0.5 ** 0.5)

        return {
            "encoder_out": (x.transpose((0, 2, 1)), y),  # ncl,nlc
            "src_mask": src_mask,  # B x T
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions


class AttentionLayer(nn.Layer):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)
        self.bmm = bmm if bmm is not None else paddle.bmm

    def forward(self, x, target_embedding, encoder_out, src_mask):  # tgt_embed是tgt的嵌入（未经卷积）
        residual = x  # conv输出（嵌入维度768转512），与attn结果最res
        # attention
        x = (self.in_projection(x) + target_embedding) * (0.5 ** 0.5)  # h [bsz tgt_len 768]
        x = self.bmm(x, encoder_out[0])  # h与z作attn  [bsz len_tgt len_src]

        # don'align_norm attend over padding
        if src_mask is not None:  # B src_len (pad为0，其他为1)
            x = x + (src_mask.unsqueeze((1)) - 1.0) * 1e9  # (pad为-inf，其他为0) [bsz tgt_len src_len]
            # FP16 support: cast to float and back

        # softmax over last dim
        x = F.softmax(x, axis=-1)  # [bsz len_tgt len_src] attn
        attn_scores = x
        # weighted sum over z+e->c
        x = self.bmm(x, encoder_out[1])  # x=attn与z+e进行加权求和得到c [bsz len_tgt 768]

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].shape[1]  # src_len(含pad)
        if src_mask is None:
            x = x * (s ** 0.5)  # 如果没有mask，x用关于长度s的缩放
        else:  # bsz src_len,x [2*5 1 512]  10 20
            s = s - paddle.cast(src_mask == 0, dtype=x.dtype).sum(axis=1, keepdim=True)  # bsz*1 求各句子非pad数
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())  # size不变 [bsz len_tgt 768],缩放了下

        # project back  [h+c]
        x = (self.out_projection(x) + residual) * (0.5 ** 0.5)  # 转回conv channel  [bsz len_tgt 512]
        return x, attn_scores

    #### 已经实现，需要unfold算子
    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace paddle.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_sublayer("bmm", BeamableMM(beamable_mm_beam_size))


class ConvS2SDecoder(PaddleDecoder):
    """Convolutional decoder"""

    def __init__(
            self,
            tgt_vocab_size,
            embed_dim=512,
            out_embed_dim=256,
            max_positions=1024,
            convolutions=((512, 3),) * 20,
            attention=True,
            dropout=0.1,
            share_embed=False,
            positional_embeddings=True,
            adaptive_softmax_cutoff=None,
            adaptive_softmax_dropout=0.0,
            pad_id=1,
    ):
        super(ConvS2SDecoder, self).__init__(tgt_vocab_size, pad_id)
        self.dropout_module = PaddleseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.need_attn = True
        self.vocab_size = tgt_vocab_size
        self.pad_id = pad_id

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError(
                "Attention is expected to be a list of booleans of "
                "length equal to the number of layers."
            )

        self.embed_tokens = Embedding(self.vocab_size, embed_dim, self.pad_id)
        self.embed_positions = (
            PositionalEmbedding(
                max_positions,
                embed_dim,
                self.pad_id,
            )
            if positional_embeddings
            else None
        )

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.LayerList()
        self.convolutions = nn.LayerList()
        self.attention = nn.LayerList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )
            self.convolutions.append(
                LinearizedConv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    padding=(kernel_size - 1),
                    dropout=dropout,
                )
            )
            self.attention.append(
                AttentionLayer(out_channels, embed_dim) if attention[i] else None
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(
                self.vocab_size,
                in_channels,
                adaptive_softmax_cutoff,
                dropout=adaptive_softmax_dropout,
            )
        else:
            self.fc2 = Linear(in_channels, out_embed_dim)
            if share_embed:
                assert out_embed_dim == embed_dim, (
                    "Shared embed weights implies same dimensions "
                    " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
                )
                self.fc3 = nn.Linear(out_embed_dim, self.vocab_size)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(out_embed_dim, self.vocab_size, dropout=dropout)

    def forward(
            self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        if encoder_out is not None:
            src_mask = encoder_out["src_mask"]
            encoder_out = encoder_out["encoder_out"]

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]  # 取最后一列，增量翻译。按说beam里面一次输入一列来着？？？???修改beamdecoder的step就行了！

        x = self._embed_tokens(prev_output_tokens, incremental_state)
        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = self.dropout_module(x)
        target_embedding = x  # 初始就是嵌入维度768.conv通道是转换的

        # project to size of convolution # N x L x C
        x = self.fc1(x)
        # train: -> T x B x C | inference:BTC
        # x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for i, (proj, conv, attention, res_layer) in enumerate(
                zip(self.projections, self.convolutions, self.attention, self.residuals)):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = self.dropout_module(x)
            if incremental_state is None:
                x = conv(x, incremental_state)
            else:
                cache = incremental_state[i % len(incremental_state)]  # 取第i个cache,当k<1时候无cache,拿前面的伪造下(不会用到)
                x, cache = conv(x, cache)
                incremental_state[i % len(incremental_state)] = cache  # 若<1,cache传进去并原封不动传回来
            x = F.glu(x, axis=2)
            # if is_diff(x.numpy(), out_dic[f'conv_{i}'], msg=f'conv{i}'): print(f'conv{i} diff')

            # attention
            if attention is not None:
                # x = self._transpose_if_training(x, incremental_state) #BTC =NLC
                # 注意力分数需不需要重新排？
                x, attn_scores = attention(
                    x, target_embedding, encoder_out, src_mask
                )
                if not self.training and self.need_attn:  # inference才会又avg attn score
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores = avg_attn_scores + attn_scores

                # x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * (0.5 ** 0.5)
            residuals.append(x)
        # T x B x C -> B x T x C
        # x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.fc3(x)
        return (x, avg_attn_scores) if incremental_state is None else (x, avg_attn_scores, incremental_state)

    def gen_caches(self, memory):
        ''' 用memory的bsz创建13个[bsz,3,in_channel]的cache,K<=1无cache'''
        caches = [layer.gen_cache(memory) for layer in self.convolutions if layer.kernel_size[0] > 1]
        return caches  # 就是increment state

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return (
            self.embed_positions.max_positions
            if self.embed_positions is not None
            else float("inf")
        )

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass # 增量推理
            tokens = tokens[:, -1:]  # 取token最后一列
        return self.embed_tokens(tokens)


class ConvS2SBeamSearchDecoder(nn.decode.BeamSearchDecoder):
    """
    This layer is a subclass of `BeamSearchDecoder` to make
    beam search adapt to Transformer decoder.

    Args:
        cell (`ConvS2SDecoder`):
            An instance of `ConvS2SDecoder`.
        start_token (int):
            The start token id.
        end_token (int):
            The end token id.
        beam_size (int):
            The beam width used in beam search.
        var_dim_in_state (int):
            Indicate which dimension of states is variant.
    """

    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(ConvS2SBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, c):
        # Init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        c = paddle.transpose(c,
                             list(range(var_dim_in_state, len(c.shape))) +
                             list(range(0, var_dim_in_state)))
        c = paddle.reshape(
            c, [0] * (len(c.shape) - var_dim_in_state
                      ) + [self.batch_size * self.beam_size] +
               [int(size) for size in c.shape[-var_dim_in_state + 2:]])
        c = paddle.transpose(
            c,
            list(range((len(c.shape) + 1 - var_dim_in_state), len(c.shape))) +
            list(range(0, (len(c.shape) + 1 - var_dim_in_state))))
        return c

    def _split_batch_beams_with_var_dim(self, c):
        var_dim_size = paddle.shape(c)[self.var_dim_in_state]
        c = paddle.reshape(
            c, [-1, self.beam_size] +
               [int(size)
                for size in c.shape[1:self.var_dim_in_state]] + [var_dim_size] +
               [int(size) for size in c.shape[self.var_dim_in_state + 1:]])
        return c

    @staticmethod
    def tile_beam_merge_with_batch(t, beam_size):  # 铺出beam （包含了多个数据，把他们拆开分别添加beam）
        r"""
        Tiles the batch dimension of a tensor. Specifically, this function takes
        a tensor align_norm shaped `[batch_size, s0, s1, ...]` composed of minibatch
        entries `align_norm[0], ..., align_norm[batch_size - 1]` and tiles it to have a shape
        `[batch_size * beam_size, s0, s1, ...]` composed of minibatch entries
        `align_norm[0], align_norm[0], ..., align_norm[1], align_norm[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Args:
            t (list|tuple):
                A list of tensor with shape `[batch_size, ...]`.
            beam_size (int):
                The beam width used in beam search.

        Returns:
            Tensor:
                A tensor with shape `[batch_size * beam_size, ...]`, whose
                data type is same as `align_norm`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerBeamSearchDecoder

                align_norm = paddle.rand(shape=[10, 10])
                TransformerBeamSearchDecoder.tile_beam_merge_with_batch(align_norm, beam_size=4)
        """
        return map_structure(
            lambda x: nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(x, beam_size), t)

    def step(self, time, inputs, states, **kwargs):
        # Steps for decoding.
        # Compared to RNN, ConvS2S has 3D data at every decoding step
        inputs = paddle.reshape(inputs, [-1, 1])  # beam*bsz x 1
        # 获取cell_states并合并bsz和beam （就是incremental_states）[bsz*beam,3,in_channels]
        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)
        # cell_outputs:[bsz*beam,1,vocab_size]
        cell_outputs, _, next_cell_states = self.cell(prev_output_tokens=inputs, encoder_out=kwargs['encoder_out'],
                                                      incremental_state=cell_states)  # encoder_out由kwargs传进来了

        # Steps for beam search.
        # Squeeze to adapt to BeamSearchDecoder which use 2D logits,cell_outputs:[bsz*beam,vocab_size]
        cell_outputs = map_structure(
            lambda x: paddle.squeeze(x, [1]) if len(x.shape) == 3 else x,
            cell_outputs)
        # split bsz and beam, cell_outputs:[bsz,beam,vocab_size]; next_cell_states:[bsz,beam,3,in_channels]*13
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)
        # start beam search,
        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,  # [bsz beam vocab]
            next_cell_states=next_cell_states,  # [cache...]
            beam_state=states)
        # [bsz beam]  [bsz beam]
        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)
        return (beam_search_output, beam_search_state, next_inputs, finished)


class InferConvS2SModel(ConvS2SModel):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_src_positions,
                 max_tgt_positions,
                 encoder_layers,  # convolutions config
                 decoder_layers,
                 encoder_embed_dim,
                 decoder_embed_dim,
                 decoder_out_embed_dim,  # decoder conv完后把通道转成嵌入维度
                 dropout=0.1,
                 decoder_attention=True,
                 share_embed=False,
                 pad_id=1,
                 bos_id=0,
                 eos_id=2,
                 beam_size=5,
                 max_out_len=256,
                 output_time_major=False,
                 beam_search_version='v2',
                 **kwargs):  # alpha,rel_len
        args = dict(locals())
        args.pop("self")
        args.pop("__class__", None)
        self.beam_size = args.pop("beam_size")
        self.max_out_len = args.pop("max_out_len")
        self.output_time_major = args.pop("output_time_major")
        self.dropout = dropout
        self.beam_search_version = args.pop('beam_search_version')
        kwargs = args.pop("kwargs")
        if self.beam_search_version == 'v2':
            self.alpha = kwargs.get("alpha", 0.6)
            self.rel_len = kwargs.get("rel_len", False)
        # self.max_out_len=None # 使用max_src_len+20
        super(InferConvS2SModel, self).__init__(**args)

        cell = self.decoder
        self.beam_decode = ConvS2SBeamSearchDecoder(
            cell, eos_id, eos_id, beam_size, var_dim_in_state=2)

    @paddle.no_grad()
    def forward(self, src_tokens):
        r"""
        The ConvS2S forward method.

        Args:
            src_tokens (Tensor):
                The ids of source sequence words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.

        Returns:
            Tensor:
                An int64 tensor shaped indicating the predicted ids. Its shape is
                `[batch_size, seq_len, beam_size]` or `[seq_len, batch_size, beam_size]`
                according to `output_time_major`.
        """
        if self.beam_search_version == 'v1':
            # Run encoder
            enc_output = self.encoder(src_tokens)

            # Init states (caches) for transformer, need to be [updated] according to selected beam, incremental_state:[bsz,3,in_channels]
            incremental_state = self.decoder.gen_caches(
                enc_output['encoder_out'][0])  # bsz,需要在initialize里面expand为bsz*beam
            # [bsz,s0,...]->[bsz*beam,s0,...]
            src_mask, encoder_out = ConvS2SBeamSearchDecoder.tile_beam_merge_with_batch(
                (enc_output["src_mask"], enc_output["encoder_out"]), self.beam_size)  # mask 和enc_out要扩展beam

            enc_output = {
                "encoder_out": encoder_out,
                "src_mask": src_mask,  # B x T
            }

            # Run decode
            rs, _ = nn.decode.dynamic_decode(
                decoder=self.beam_decode,
                inits=incremental_state,
                output_time_major=self.output_time_major,
                max_step_num=self.max_out_len,
                is_test=True,
                encoder_out=enc_output)  # **kwargs

            return rs
        elif self.beam_search_version == 'v2':
            # 1.搜索
            finished_seq, finished_scores = self.beam_search_v2(
                src_tokens, self.beam_size, self.max_out_len, self.alpha)
            # 2.重排顺序
            if self.output_time_major:
                finished_seq = finished_seq.transpose([2, 0, 1])  # bsz beam time ->[batch_size, time, beam_size]`
            else:
                finished_seq = finished_seq.transpose([0, 2, 1])  # bsz beam time ->[time, batch_size, beam_size]

            return finished_seq

    def beam_search_v2(self, src_tokens, beam_size=4, max_len=None, alpha=0.6):
        """
        维护alive和finish两个队列，各自放beam个，有3个核心函数
        Beam search with the alive and finished two queues, both have a beam size
        capicity separately. It includes `grow_topk` `grow_alive` `grow_finish` as
        steps.
        1. `grow_topk` selects the top `2*beam_size` candidates to avoid all getting
        EOS. # 选出2beam个
        2. `grow_alive` selects the top `beam_size` non-EOS candidates as the inputs
        of next decoding step. # 筛选出beam个
        3. `grow_finish` compares the already finished candidates in the finished queue
        and newly added finished candidates from `grow_topk`, and selects the top
        `beam_size` finished candidates.
        """

        def expand_to_beam_size(tensor, beam_size):  # [bsz beam len]
            tensor = paddle.unsqueeze(tensor, axis=1)
            tile_dims = [1] * len(tensor.shape)  # 其他维度不变
            tile_dims[1] = beam_size  # 第二维度复制beam次
            return paddle.tile(tensor, tile_dims)

        def merge_beam_dim(tensor):  # 合并bsz和beam
            shape = tensor.shape
            return paddle.reshape(tensor,
                                  [shape[0] * shape[1]] + list(shape[2:]))  # [bsz*beam len]

        ### 1.Run encoder ###
        enc_output = self.encoder(src_tokens)

        # constant number
        inf = float(1. * 1e7)
        batch_size = enc_output['encoder_out'][0].shape[0]  # [bsz dim len]
        src_len = enc_output['encoder_out'][0].shape[2]
        max_len = (src_len + 20) if max_len is None else (  # max_len没设置默认是<=src_len+20,比最长的长20（会不会是解码的时候也是短句先行）
            src_len + max_len if self.rel_len else max_len)  # max_len有的情况下，如果是相对长度，则<=src_len+max_len；否则就算绝对不超过max_len

        ### 2.initialize states of beam search ### (不作修改)
        ## init for the alive
        initial_log_probs = paddle.assign(
            np.array(
                [[0.] + [-inf] * (beam_size - 1)], dtype="float32"))  # []*n为元素复制n份，所以是beam个元素，第一个0，后面全是-inf
        alive_log_probs = paddle.tile(initial_log_probs, [batch_size, 1])  # 结果两个维度，0复制bsz份，1维维持1份， [bsz beam]
        # seq存放id，先给出bos [bsz beam 1]
        alive_seq = paddle.tile(
            paddle.cast(
                paddle.assign(np.array([[[self.bos_id]]])), src_tokens.dtype),
            [batch_size, beam_size, 1])  # 2维维持不变，1个bos元素；复制bsz 和beam份，结果： [bsz beam 1]

        # finish
        ## init for the finished ##
        finished_scores = paddle.assign(
            np.array(
                [[-inf] * beam_size], dtype="float32"))  # [1 beam]个-inf
        finished_scores = paddle.tile(finished_scores, [batch_size, 1])  # [bsz beam]
        # seq放id [bsz beam 1]
        finished_seq = paddle.tile(
            paddle.cast(
                paddle.assign(np.array([[[self.bos_id]]])), src_tokens.dtype),
            [batch_size, beam_size, 1])
        finished_flags = paddle.zeros_like(finished_scores)  # 结束标志，可能1就退出

        ### 3.initialize inputs and states of convs2s decoder ###
        ## init inputs for decoder, shaped `[batch_size*beam_size, ...]`
        # 新增decoder输入
        tgt_word = paddle.reshape(alive_seq[:, :, -1],
                                  [batch_size * beam_size, 1])
        # expand:[bsz beam len]->merge:[bsz*beam len]
        src_mask = merge_beam_dim(expand_to_beam_size(enc_output["src_mask"], beam_size))
        z = merge_beam_dim(
            expand_to_beam_size(enc_output['encoder_out'][0], beam_size))  # encoder output z: [bsz*beam dim src_len]
        z_e = merge_beam_dim(expand_to_beam_size(enc_output['encoder_out'][1],
                                                 beam_size))  # #encoder output z plus embedding: [bsz*beam src_len dim]

        enc_output = {
            "encoder_out": (z, z_e),
            "src_mask": src_mask,  # B x T
        }
        ## init states (caches) for transformer, need to be updated according to selected beam
        caches = self.decoder.gen_caches(enc_output['encoder_out'][
                                             0])  # [bsz*beam k in_channel] 妙!不用expand的原因cache随enc的维度，enc已经expand为[bsz*beam ...]

        # 根据topk，调整convs2s decoder的cache的顺序（还拓展为2beam）
        def update_states(caches, topk_coordinates, beam_size, batch_size):
            new_caches = []
            for cache in caches:
                input_buffer = gather_2d(
                    cache.input_buffer,
                    topk_coordinates,
                    beam_size,
                    batch_size,
                    need_unmerge=True)
                new_caches.append(
                    LinearizedConvolution.Cache(input_buffer=input_buffer))  # buffer的维度变成了2倍，而cache不能替换，只能修改元素，所以要重新生成
            return new_caches

        def get_topk_coordinates(beam_idx, beam_size, batch_size,
                                 dtype="int64"):
            batch_pos = paddle.arange(
                batch_size * beam_size, dtype=dtype) // beam_size
            batch_pos = paddle.reshape(batch_pos, [batch_size, beam_size])
            topk_coordinates = paddle.stack([batch_pos, beam_idx], axis=2)
            return topk_coordinates

        def gather_2d(tensor_nd,
                      topk_coordinates,
                      beam_size,
                      batch_size,
                      need_unmerge=False):
            # 1.split bsz and beam
            new_tensor_nd = paddle.reshape(
                tensor_nd,
                shape=[batch_size, beam_size] +  # 如果需要拆分（unmerge）的话，拆bsz 和beam
                      list(tensor_nd.shape[1:])) if need_unmerge else tensor_nd  # 否则维持不变
            # re order
            topk_seq = paddle.gather_nd(new_tensor_nd, topk_coordinates)
            return merge_beam_dim(topk_seq) if need_unmerge else topk_seq  # 3.merge bsz and beam

        def early_finish(alive_log_probs, finished_scores,
                         finished_in_finished):
            max_length_penalty = np.power(((5. + max_len) / 6.), alpha)
            lower_bound_alive_scores = alive_log_probs[:,
                                       0] / max_length_penalty
            lowest_score_of_fininshed_in_finished = paddle.min(
                finished_scores * finished_in_finished, 1)
            lowest_score_of_fininshed_in_finished += (
                                                             1. - paddle.max(finished_in_finished, 1)) * -inf
            bound_is_met = paddle.all(
                paddle.greater_than(lowest_score_of_fininshed_in_finished,
                                    lower_bound_alive_scores))

            return bound_is_met

        def grow_topk(i, logits, alive_seq, alive_log_probs, states):
            """
            This function takes the current alive sequences, and grows them to topk
            sequences where k = 2*beam.
            """
            logits = paddle.reshape(logits, [batch_size, beam_size, -1])
            candidate_log_probs = paddle.log(F.softmax(logits, axis=2))
            log_probs = paddle.add(candidate_log_probs,
                                   alive_log_probs.unsqueeze(-1))

            # Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
            # https://arxiv.org/abs/1609.08144.
            length_penalty = paddle.pow(5.0 + (i + 1.0) / 6.0, alpha)
            curr_scores = log_probs / length_penalty
            flat_curr_scores = paddle.reshape(curr_scores, [batch_size, -1])

            topk_scores, topk_ids = paddle.topk(
                flat_curr_scores, k=beam_size * 2)
            if topk_ids.dtype != alive_seq.dtype:
                topk_ids = paddle.cast(topk_ids, dtype=alive_seq.dtype)

            topk_log_probs = topk_scores * length_penalty

            topk_beam_index = topk_ids // self.tgt_vocab_size
            topk_ids = topk_ids % self.tgt_vocab_size

            topk_coordinates = get_topk_coordinates(
                topk_beam_index,
                beam_size * 2,
                batch_size,
                dtype=alive_seq.dtype)
            topk_seq = gather_2d(alive_seq, topk_coordinates, beam_size,
                                 batch_size)
            topk_seq = paddle.concat(
                [
                    topk_seq, paddle.reshape(topk_ids,
                                             list(topk_ids.shape[:]) + [1])
                ],
                axis=2)
            # update_states
            states = update_states(states, topk_coordinates, beam_size,
                                   batch_size)
            eos = paddle.full(
                shape=paddle.shape(topk_ids),
                dtype=alive_seq.dtype,
                fill_value=self.eos_id)
            topk_finished = paddle.cast(paddle.equal(topk_ids, eos), "float32")

            # topk_seq: [batch_size, 2*beam_size, i+1]
            # topk_log_probs, topk_scores, topk_finished: [batch_size, 2*beam_size]
            return topk_seq, topk_log_probs, topk_scores, topk_finished, states

        def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished,
                       states):
            """
            Given sequences and scores, will gather the top k=beam size sequences
            """
            curr_scores += curr_finished * -inf
            _, topk_indexes = paddle.topk(curr_scores, k=beam_size)
            if topk_indexes.dtype != curr_seq.dtype:
                topk_indexes = paddle.cast(topk_indexes, dtype=curr_seq.dtype)

            topk_coordinates = get_topk_coordinates(
                topk_indexes, beam_size, batch_size, dtype=curr_seq.dtype)
            alive_seq = gather_2d(curr_seq, topk_coordinates, beam_size,
                                  batch_size)

            alive_log_probs = gather_2d(curr_log_probs, topk_coordinates,
                                        beam_size, batch_size)
            states = update_states(states, topk_coordinates, beam_size * 2,
                                   batch_size)

            return alive_seq, alive_log_probs, states

        def grow_finished(finished_seq, finished_scores, finished_flags,
                          curr_seq, curr_scores, curr_finished):
            """
            Given sequences and scores, will gather the top k=beam size sequences.
            """
            # finished scores
            finished_seq = paddle.concat(
                [
                    finished_seq, paddle.full(
                    shape=[batch_size, beam_size, 1],
                    dtype=finished_seq.dtype,
                    fill_value=self.eos_id)
                ],
                axis=2)
            curr_scores += (1. - curr_finished) * -inf
            curr_finished_seq = paddle.concat([finished_seq, curr_seq], axis=1)
            curr_finished_scores = paddle.concat(
                [finished_scores, curr_scores], axis=1)
            curr_finished_flags = paddle.concat(
                [finished_flags, curr_finished], axis=1)
            _, topk_indexes = paddle.topk(curr_finished_scores, k=beam_size)
            if topk_indexes.dtype != curr_seq.dtype:
                topk_indexes = paddle.cast(topk_indexes, dtype=curr_seq.dtype)

            topk_coordinates = get_topk_coordinates(
                topk_indexes, beam_size, batch_size, dtype=curr_seq.dtype)
            finished_seq = gather_2d(curr_finished_seq, topk_coordinates,
                                     beam_size, batch_size)
            finished_scores = gather_2d(curr_finished_scores, topk_coordinates,
                                        beam_size, batch_size)
            finished_flags = gather_2d(curr_finished_flags, topk_coordinates,
                                       beam_size, batch_size)

            return finished_seq, finished_scores, finished_flags

        # 类似step函数
        def inner_loop(i, tgt_word, alive_seq, alive_log_probs, finished_seq,
                       finished_scores, finished_flags, caches):  # 注意caches要改 xbeam？调整顺序？
            ## 1.run decode ##
            logits, avg_attn_scores, caches = self.decoder(
                prev_output_tokens=tgt_word,
                encoder_out=enc_output,
                incremental_state=caches)
            ## 2.grow ##
            topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(
                i, logits, alive_seq, alive_log_probs, caches)
            alive_seq, alive_log_probs, states = grow_alive(
                topk_seq, topk_scores, topk_log_probs, topk_finished, states)
            caches = states
            finished_seq, finished_scores, finished_flags = grow_finished(
                finished_seq, finished_scores, finished_flags, topk_seq,
                topk_scores, topk_finished)
            ## 3.get last token for next step ##
            tgt_word = paddle.reshape(alive_seq[:, :, -1],
                                      [batch_size * beam_size, 1])
            return (i + 1, tgt_word, alive_seq, alive_log_probs, finished_seq,
                    finished_scores, finished_flags, caches)

        def is_not_finish(i, tgt_word, alive_seq, alive_log_probs, finished_seq,
                          finished_scores, finished_flags, caches):
            return paddle.greater_than(
                i < max_len,
                # early_finish
                early_finish(alive_log_probs, finished_scores, finished_flags))

        _, tgt_word, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, caches = paddle.static.nn.while_loop(
            is_not_finish,  # cond
            inner_loop, [  # body
                paddle.zeros(shape=[1]), tgt_word, alive_seq, alive_log_probs,
                finished_seq, finished_scores, finished_flags, caches
            ])

        # (gongenlei) `paddle.where` doesn't support broadcast, so we need to use `paddle.unsqueeze` # where不支持广播，手动扩张到该维度，然后乘
        # and `paddle.tile` to make condition.shape same as X.shape. But when converting dygraph
        # to static  graph, `paddle.tile` will raise error.
        finished_flags = paddle.cast(finished_flags, dtype=finished_seq.dtype)
        neg_finished_flags = 1 - finished_flags
        finished_seq = paddle.multiply(
            finished_seq, finished_flags.unsqueeze(-1)) + paddle.multiply(  # 把后面掩码为0
            alive_seq, neg_finished_flags.unsqueeze(-1))  # 把后面掩为1，然后两相加后左边是token，后面1是pad
        finished_scores = paddle.multiply(
            finished_scores,
            paddle.cast(
                finished_flags, dtype=finished_scores.dtype)) + paddle.multiply(
            alive_log_probs,
            paddle.cast(
                neg_finished_flags, dtype=alive_log_probs.dtype))
        return finished_seq, finished_scores


def extend_conv_spec(convolutions):
    """
    添加默认配置（残差）
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception(
                "invalid number of parameters in convolution spec "
                + str(spec)
                + ". expected 2 or 3"
            )
    return tuple(extended)


def Embedding(num_embeddings, embedding_dim, pad_id):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=pad_id)
    normal_(m.weight)
    zeros_(m.weight[pad_id])
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, pad_id):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, pad_id)  # 这里要改，forward支持incremental 【增量推理】
    normal_(m.weight)
    zeros_(m.weight[pad_id])
    return m


def Linear(in_features, out_features, dropout=0.0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    std = ((1 - dropout) / in_features) ** 0.5
    normal_ = Normal(0, std)
    normal_(m.weight)
    zeros_(m.bias)
    return nn.utils.weight_norm(m, dim=1)  # dim1,对out features拆分


# decoder中对kernel做了weight_norm
def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = ((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels)) ** 0.5
    normal_ = Normal(0, std)
    normal_(m.weight)
    zeros_(m.bias)
    return nn.utils.weight_norm(m, dim=0)


def ConvNLC(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from models.layers import ConvNLC
    m = ConvNLC(in_channels, out_channels, kernel_size, **kwargs)
    std = ((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels)) ** 0.5
    normal_ = Normal(0, std)
    normal_(m.weight)
    zeros_(m.bias)
    return nn.utils.weight_norm(m, dim=0)  # 也是对out_channels


def _create_convs2s(variant, is_test, pretrained_path, args):
    if not is_test:
        del args['beam_size'], args['max_out_len']
        model = ConvS2SModel(**args)
        print(f'Train model {variant} created!')
    else:
        print(f'Infer model {variant} created!')
        model = InferConvS2SModel(**args)

    if pretrained_path is not None:
        state = paddle.load(pretrained_path)
        model.set_dict(state)
        print(f'Pretrained weight load from:{pretrained_path}!')
    return model


def base_architecture(args):
    args["dropout"] = args.get("dropout", 0.1)
    args["encoder_embed_dim"] = args.get("encoder_embed_dim", 512)
    args["encoder_layers"] = args.get("encoder_layers", "[(512, 3)] * 20")
    args["decoder_embed_dim"] = args.get('decoder_embed_dim', 512)
    args["decoder_layers"] = args.get("decoder_layers", "[(512, 3)] * 20")
    args["decoder_out_embed_dim"] = args.get("decoder_out_embed_dim", 256)
    args["decoder_attention"] = args.get("decoder_attention", True)
    args["share_embed"] = args.get("share_embed", False)
    return args


cfgs = ['src_vocab_size', 'tgt_vocab_size', 'max_src_positions', 'max_tgt_positions']


def convs2s_iwslt_de_en(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(encoder_layers="[(256, 3)] * 4", decoder_layers="[(256, 3)] * 3",
                      encoder_embed_dim=256, decoder_embed_dim=256,
                      decoder_out_embed_dim=256, **kwargs)
    model_args = base_architecture(model_args)
    model = _create_convs2s('convs2s_iwslt_de_en', is_test, pretrained_path, model_args)
    return model


def convs2s_wmt_en_ro(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    model_args = dict(decoder_out_embed_dim=512, **kwargs)
    model_args = base_architecture(model_args)
    model = _create_convs2s('convs2s_wmt_en_ro', is_test, pretrained_path, model_args)
    return model


def convs2s_wmt_en_de(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    convs = "[(512, 3)] * 9"  # first 9 layers have 512 units
    convs += " + [(1024, 3)] * 4"  # next 4 layers have 1024 units
    convs += " + [(2048, 1)] * 2"  # final 2 layers use 1x1 convolutions
    model_args = dict(encoder_layers=convs, decoder_layers=convs,
                      encoder_embed_dim=768, decoder_embed_dim=768,
                      decoder_out_embed_dim=512, **kwargs)
    model_args = base_architecture(model_args)
    model = _create_convs2s('convs2s_wmt_en_de', is_test, pretrained_path, model_args)
    return model


def convs2s_wmt_en_fr(is_test=False, pretrained_path=None, **kwargs):
    for cfg in cfgs: assert cfg in kwargs, f'missing argument:{cfg}'
    convs = "[(512, 3)] * 6"  # first 6 layers have 512 units
    convs += " + [(768, 3)] * 4"  # next 4 layers have 768 units
    convs += " + [(1024, 3)] * 3"  # next 3 layers have 1024 units
    convs += " + [(2048, 1)] * 1"  # next 1 layer uses 1x1 convolutions
    convs += " + [(4096, 1)] * 1"  # final 1 layer uses 1x1 convolutions

    model_args = dict(encoder_layers=convs, decoder_layers=convs,
                      encoder_embed_dim=768, decoder_embed_dim=768,
                      decoder_out_embed_dim=512, **kwargs)
    model_args = base_architecture(model_args)
    model = _create_convs2s('convs2s_wmt_en_fr', is_test, pretrained_path, model_args)
    return model



