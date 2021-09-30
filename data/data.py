import os
import numpy as np
from functools import partial
import paddle
import paddle.distributed as dist
from .sampler import DynamicBatchSampler,DynamicTestSampler,DistributedDynamicBatchSampler

from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import Vocab, Pad
from paddlenlp.datasets import load_dataset
import yaml


# 自定义读取本地数据的方法
def read(src_path, tgt_path, is_predict=False):
    if is_predict:
        with open(src_path, 'r', encoding='utf-8') as src_f:
            for src_line in src_f.readlines():
                src_line = src_line.strip()
                if not src_line:
                    continue
                yield {'src': src_line, 'tgt': ''}
    else:
        with open(src_path, 'r', encoding='utf-8') as src_f, open(tgt_path, 'r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f.readlines(), tgt_f.readlines()):
                src_line, tgt_line = src_line.strip(), tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                yield {'src': src_line, 'tgt': tgt_line}


class SentsLenDataset(paddle.io.Dataset):
    ''' 返回src、tgt句子最大长度的dataset，用于sampler动态采样 '''

    def __init__(self, dataset):
        if len(dataset[0])==2:
            self.sents_max_len = [max(len(data[0]), len(data[1])) for data in dataset]
        elif len(dataset[0])==1:
            self.sents_max_len = [len(data) for data in dataset]

    def __getitem__(self, idx):
        return self.sents_max_len[idx]

    def __len__(self):
        return len(self.sents_max_len)


def prep_dataset(conf, mode='train'):
    data_args=conf.data
    assert mode in ['train', 'dev', 'test']
    if mode == 'train':
        dataset = load_dataset(read, src_path=data_args.training_file.split(',')[0],
                               tgt_path=data_args.training_file.split(',')[1], lazy=False)
    elif mode == 'dev':
        dataset = load_dataset(read, src_path=data_args.validation_file.split(',')[0],
                               tgt_path=data_args.validation_file.split(',')[1], lazy=False)
    else:
        file=data_args.predict_file.split(',')[0]
        dataset = load_dataset(read, src_path=data_args.predict_file.split(',')[0], tgt_path=None, is_predict=True, lazy=False)
    return dataset


def prep_vocab(conf):
    data_args=conf.data
    src_vocab = Vocab.load_vocabulary(
        data_args.src_vocab_fpath,
        bos_token=data_args.special_token[0],
        eos_token=data_args.special_token[1],
        unk_token=data_args.special_token[2],
        pad_token=data_args.special_token[3]
    )
    tgt_vocab = Vocab.load_vocabulary(
        data_args.tgt_vocab_fpath,
        bos_token=data_args.special_token[0],
        eos_token=data_args.special_token[1],
        unk_token=data_args.special_token[2],
        pad_token=data_args.special_token[3]
    )
    # 是否把vocab词数pad到factor倍数，可以加速训练
    if data_args.pad_vocab:
        padding_vocab = (
            lambda x: (x + data_args.pad_factor - 1) // data_args.pad_factor * data_args.pad_factor
        )
        conf['model']['src_vocab_size'] = padding_vocab(len(src_vocab)) # 要修改attrdict必须用中括号
        conf['model']['tgt_vocab_size'] = padding_vocab(len(tgt_vocab))
    else:
        conf['model']['src_vocab_size'] = len(src_vocab)
        conf['model']['tgt_vocab_size'] = len(tgt_vocab)

    return src_vocab, tgt_vocab


def convert_samples(sample, src_vocab, tgt_vocab):
    source = sample['src'].split()
    target = sample['tgt'].split()
    source = src_vocab.to_indices(source)
    target = tgt_vocab.to_indices(target)
    return source, target


# 过滤掉长度 ≤min_len或者≥max_len 的数据
def min_max_filer(data, max_len, min_len=0):
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def batchify_train_dev(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by training into a list.
    # insts是含batch个元素的list，每个batch含src和tgt两个元素
    """
    src_pad = Pad(pad_idx, pad_right=False)
    tgt_pad = Pad(pad_idx, pad_right=True)
    src_word = src_pad([inst[0] + [eos_idx] for inst in insts])  # src+</s>
    tgt_word = tgt_pad([[bos_idx] + inst[1] for inst in insts])  # <s>+tgt
    lbl_word = np.expand_dims(tgt_pad([inst[1] + [eos_idx] for inst in insts]),
                              axis=2)  # lbl+</s> # pad时候加了bos或eos，导致size突变，*bsz倍
    data_inputs = [src_word, tgt_word, lbl_word]

    return data_inputs


def batchify_infer(insts, eos_idx, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx, pad_right=True)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])

    return src_word


def prep_loader(conf,dataset, mode='train', multi_process=False):
    assert mode in ['train', 'dev', 'test']
    data_args,model_args,strategy_args,train_args,gen_args=conf.data,conf.model,conf.learning_strategy,conf.train,conf.generate
    # load vocab
    src_vocab, tgt_vocab = prep_vocab(conf)
    # dataset
    trans_fn=partial(convert_samples,src_vocab=src_vocab,tgt_vocab=tgt_vocab)
    if mode != 'test':
        dataset = dataset.map(trans_fn, lazy=False)\
                         .filter(partial(min_max_filer,max_len=model_args.max_length))

        batchify_fn = partial(batchify_train_dev, bos_idx=model_args.eos_idx, eos_idx=model_args.eos_idx, pad_idx=model_args.pad_idx)
    else:
        dataset = dataset.map(trans_fn, lazy=False)
        batchify_fn = partial(batchify_infer, eos_idx=model_args.eos_idx, pad_idx=model_args.pad_idx)

    # sampler
    sents_maxlen_dset = SentsLenDataset(dataset)  # dataset对应的最大句长
    if multi_process==True:
        batch_sampler=DistributedDynamicBatchSampler(data_source=sents_maxlen_dset,shuffle=(mode=='train'),num_buckets=model_args.max_length,
                                                     min_size=model_args.min_length,max_size=model_args.max_length,max_tokens=train_args.max_tokens,
                                                     bsz_factor=train_args.batch_size_factor) # max_sentences=train_args.max_sentences
    else:
        if mode=='test':
            batch_sampler = DynamicTestSampler(data_source=sents_maxlen_dset, max_tokens=gen_args.infer_max_tokens)
        else:
            batch_sampler = DistributedDynamicBatchSampler(data_source=sents_maxlen_dset,shuffle=(mode == 'train'),num_buckets=model_args.max_length,
                                                           min_size=model_args.min_length,max_size=model_args.max_length,max_tokens=train_args.max_tokens,
                                                           bsz_factor=train_args.batch_size_factor,num_replicas=1,rank=0) #,max_sentences=train_args.max_sentences
    # dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=train_args.num_workers,
        )

    return dataloader if mode!='test' else (dataloader,batch_sampler,tgt_vocab.to_tokens)

