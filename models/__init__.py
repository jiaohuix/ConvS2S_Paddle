import os
from .base_model import  EncoderDecoderModel
from .base_encoder import PaddleEncoder
from .base_decoder import PaddleDecoder
from .convs2s import ConvS2SEncoder,ConvS2SDecoder,AttentionLayer,ConvS2SModel
from .convs2s import convs2s_iwslt_de_en,convs2s_wmt_en_ro,convs2s_wmt_en_de,convs2s_wmt_en_fr
import models

def build_model(conf,is_test=False):
    model_args,gen_args=conf.model,conf.generate
    model_path=os.path.join(model_args.init_from_params,'convs2s.pdparams')
    model_path=None if not os.path.exists(model_path) else model_path
    model=getattr(models,model_args.model_name)(
                                        is_test=is_test,
                                        pretrained_path=model_path,
                                        src_vocab_size=model_args.src_vocab_size,
                                        tgt_vocab_size=model_args.tgt_vocab_size,
                                        max_src_positions=model_args.max_length,
                                        max_tgt_positions=model_args.max_length,
                                        dropout=model_args.dropout,
                                        beam_size=gen_args.beam_size,
                                        max_out_len=gen_args.max_out_len,
                                        rel_len=gen_args.rel_len,
                                        alpha=gen_args.alpha)
    return model
