import numpy as np
from functools import partial
from paddle.io import DataLoader
from paddlenlp.data import Vocab, Pad, Stack
from paddlenlp.datasets import load_dataset
from .sampler import DistributedDynamicBatchSampler

def read(src_path, tgt_path, is_test=False, has_target=False):
    if is_test and not has_target:
        with open(src_path, 'r', encoding='utf-8') as src_f:
            for sample_id, src_line in enumerate(src_f.readlines()):
                src_line = src_line.strip()
                if not src_line:
                    continue
                yield {'id': sample_id, 'src': src_line, 'tgt': ''}
    else:
        with open(src_path, 'r', encoding='utf-8') as src_f, open(tgt_path, 'r', encoding='utf-8') as tgt_f:
            for sample_id, (src_line, tgt_line) in enumerate(zip(src_f.readlines(), tgt_f.readlines())):
                src_line, tgt_line = src_line.strip(), tgt_line.strip()
                if not src_line or not tgt_line:
                    continue
                yield {'id': sample_id, 'src': src_line, 'tgt': tgt_line}


def merge_pref_lang(pref, lang):
    return f"{pref.strip()}.{lang.strip()}"


def prep_dataset(conf, mode='train'):
    assert mode in ['train', 'dev', 'test']
    data_args = conf.data
    src_lang = data_args.src_lang
    tgt_lang = data_args.tgt_lang
    if mode == 'train':
        src_path = merge_pref_lang(data_args.train_pref, src_lang)
        tgt_path = merge_pref_lang(data_args.train_pref, tgt_lang)
    elif mode == 'dev':
        src_path = merge_pref_lang(data_args.valid_pref, src_lang)
        tgt_path = merge_pref_lang(data_args.valid_pref, tgt_lang)
    else:
        src_path = merge_pref_lang(data_args.test_pref, src_lang)
        tgt_path = merge_pref_lang(data_args.test_pref, tgt_lang)

    dataset = load_dataset(read, src_path=src_path, tgt_path=tgt_path, is_test=(mode == 'test'),
                           has_target=conf.data.has_target,lazy=False)
    return dataset


def prep_vocab(conf):
    data_args = conf.data
    src_vocab_fpath = merge_pref_lang(data_args.vocab_pref, data_args.src_lang)
    tgt_vocab_fpath = merge_pref_lang(data_args.vocab_pref, data_args.tgt_lang)
    src_vocab = Vocab.load_vocabulary(
        src_vocab_fpath,
        bos_token=data_args.special_token[0],
        pad_token=data_args.special_token[1],
        eos_token=data_args.special_token[2],
        unk_token=data_args.special_token[3]
    )
    tgt_vocab = Vocab.load_vocabulary(
        tgt_vocab_fpath,
        bos_token=data_args.special_token[0],
        pad_token=data_args.special_token[1],
        eos_token=data_args.special_token[2],
        unk_token=data_args.special_token[3]
    )
    # 是否把vocab词数pad到factor倍数，可以加速训练
    conf.defrost()
    if data_args.pad_vocab:
        padding_vocab = (
            lambda x: (x + data_args.pad_factor - 1) // data_args.pad_factor * data_args.pad_factor
        )
        conf.model.src_vocab_size = padding_vocab(len(src_vocab))
        conf.model.tgt_vocab_size  = padding_vocab(len(tgt_vocab))
    else:
        conf.model.src_vocab_size = len(src_vocab)
        conf.model.tgt_vocab_size = len(tgt_vocab)
    conf.freeze()
    return src_vocab, tgt_vocab


def convert_samples(sample, src_vocab, tgt_vocab):
    sample_id = sample['id']
    source = sample['src'].split()
    target = sample['tgt'].split()
    source = src_vocab.to_indices(source)
    target = tgt_vocab.to_indices(target)
    return source, target, sample_id


# 过滤掉长度 ≤min_len或者≥max_len 的数据
def min_max_filer(data, max_len, min_len=0):
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def batchify(insts, bos_idx, eos_idx, pad_idx, is_test=False, has_target=False):
    """
    Put all padded data needed by training into a list.
    # insts是含batch个元素的list，每个batch含src和tgt,和id元素[([],[]),([],[]),...]
    """
    # ★sort by descending source length
    neg_src_len = list(map(lambda inst: -len(inst[0]), insts))
    sorted_src_idx = np.argsort(neg_src_len, kind='mergsort')  # 不能用[::-1]，假设在长度全相等时，会从1-n变成n-1;且默认quicksort不稳定
    insts = np.array(insts)[sorted_src_idx].tolist()

    # pad data to full sentence length
    left_pad = Pad(pad_idx, pad_right=False)
    right_pad = Pad(pad_idx, pad_right=True, dtype='int64')
    src_word = left_pad([inst[0] + [eos_idx] for inst in insts])  # src+</s>
    samples_id = Stack()([inst[2] for inst in insts])
    if not is_test:
        prev_word = right_pad([[bos_idx] + inst[1] for inst in insts])  # <s>+tgt
        tgt_word = np.expand_dims(right_pad([inst[1] + [eos_idx] for inst in insts]),
                                  axis=2)  # lbl+</s> # pad时候加了bos或eos，导致size突变，*bsz倍
        data_inputs = [samples_id, src_word, prev_word, tgt_word]
    else:
        if not has_target:
            data_inputs = [samples_id, src_word]
        else:
            tgt_word = right_pad([inst[1] for inst in insts])
            data_inputs = [samples_id, src_word, tgt_word]

    return data_inputs


def prep_loader(conf, dataset, mode='train', multi_process=False):
    assert mode in ['train', 'dev', 'test']
    data_args, model_args, strategy_args, train_args, gen_args = conf.data, conf.model, conf.learning_strategy, conf.train, conf.generate
    # load vocab
    src_vocab, tgt_vocab = prep_vocab(conf)
    # dataset
    trans_fn = partial(convert_samples, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    dataset = dataset.map(trans_fn, lazy=False)
    if mode != 'test':
        filt_fn = partial(min_max_filer, max_len=model_args.max_length)
        dataset = dataset.filter(filt_fn)
    batchify_fn = partial(batchify, bos_idx=model_args.eos_idx, eos_idx=model_args.eos_idx,
                          pad_idx=model_args.pad_idx, is_test=mode == 'test', has_target=data_args.has_target)

    # samplerv2
    max_tokens = train_args.max_tokens if mode != 'test' else gen_args.max_tokens
    max_sentences = train_args.max_sentences if mode != 'test' else gen_args.max_sentences
    batch_sampler = DistributedDynamicBatchSampler(dataset,
                                                     mode=mode,
                                                     has_target=data_args.has_target,
                                                     max_tokens=max_tokens,
                                                     max_sentences=eval(str(max_sentences)),
                                                     bsz_factor=train_args.batch_size_factor,
                                                     seed=conf.seed,
                                                     num_replicas=None if multi_process == True else 1,
                                                     rank=None if multi_process == True else 0,
                                                     drop_last=False)

    if conf.train.resume and mode == 'train':  # resume应该bool,路径由init来决定
        batch_sampler.set_epoch(conf.train.last_epoch + 1)
        print(f"----- Resume Training: set sampler's epoch to {conf.train.last_epoch + 1} as a random seed")

    # dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=train_args.num_workers,
    )

    return dataloader
