#-*- coding: utf-8 -*-
import os
import sys
import math
import paddle
import utils
from config import get_arguments, get_config
from data import prep_dataset, prep_loader
from utils import get_logger, sort_file
from data import prep_vocab
from models import build_model, SequenceGenerator
from paddlenlp.metrics import BLEU

@paddle.no_grad()
def generate(conf):
    utils.same_seeds(seed=conf.seed)
    if not os.path.exists(conf.SAVE): os.makedirs(conf.SAVE)
    generate_path = os.path.join(conf.SAVE, conf.generate.generate_path)
    sorted_path = os.path.join(conf.SAVE, conf.generate.sorted_path)
    # 当设置generate_path时，保存结果到文件，否则打印
    out_file = open(generate_path, 'w', encoding='utf-8') if conf.generate.generate_path else sys.stdout
    logger = get_logger(loggername=f"ConvS2S", save_path=conf.SAVE)
    test_dset = prep_dataset(conf, mode='test')
    test_loader = prep_loader(conf, test_dset, mode='test')
    src_vocab, tgt_vocab = prep_vocab(conf)

    logger.info('Prep | Loading models...')
    model = build_model(conf, is_test=True)
    model.eval()
    scorer = BLEU()
    generator = SequenceGenerator(model, vocab_size=model.tgt_vocab_size, beam_size=conf.generate.beam_size)
    # 1.for batch
    logger.info('Pred | Predicting...')
    has_target = conf.data.has_target
    for batch_id, batch_data in enumerate(test_loader):
        print(f'batch_id:[{batch_id + 1}/{len(test_loader)}]')
        samples_id, src_tokens, tgt_tokens = None, None, None
        if has_target:
            samples_id, src_tokens, tgt_tokens = batch_data
        else:
            samples_id, src_tokens = batch_data
        bsz = src_tokens.shape[0]
        # samples_id=paddle.arange(bsz)
        samples = {'id': samples_id, 'nsentences': bsz,
                   'net_input': {'src_tokens': paddle.cast(src_tokens, dtype='int64')},  # 需要和后面生成的cand_indices类型一致
                   'target':tgt_tokens}
        hypos = generator.generate(samples)

        # 2.for sample
        for i, sample_id in enumerate(samples["id"].tolist()):
            # 解码src和tgt，并打印
            src_text = utils.post_process(sentence=" ".join(src_vocab.to_tokens(test_dset[sample_id][0])),
                                          symbol='subword_nmt')
            print("S-{}\t{}".format(sample_id, src_text), file=out_file)
            if has_target:
                tgt_text = utils.post_process(sentence=" ".join(tgt_vocab.to_tokens(test_dset[sample_id][1])),
                                              symbol='subword_nmt')
                print("T-{}\t{}".format(sample_id, tgt_text), file=out_file)

            # 3.for prediction
            for j, hypo in enumerate(hypos[i][: conf.generate.n_best]):  # 从第i个sample的beam=5个hypo中，取best=1个
                # 3.1对hypo后处理
                hypo_str = utils.to_string(hypo["tokens"], tgt_vocab, bpe_symbol='subword_nmt',
                                           extra_symbols_to_ignore=[model.bos_id, model.eos_id, model.pad_id])
                # 3.2 打印信息
                score = (hypo["score"] / math.log(2)).item()
                print("H-{}\t{:.4f}\t{}".format(sample_id, score, hypo_str), file=out_file)
                print(
                    "P-{}\t{}".format(sample_id,
                                      " ".join(
                                          map(lambda x: "{:.4f}".format(x),
                                              # convert from base e to base 2
                                              (hypo["positional_scores"] / math.log(2)).tolist(),
                                              )
                                      ),
                                      ),
                    file=out_file
                )
                # 3.3 记录得分（hypo target）token分数，是索引的
                if has_target and j == 0:
                    scorer.add_inst(cand=hypo_str.split(), ref_list=[tgt_text.split()])

    # 打印最终得分
    if has_target:
        logger.info(f"BlEU Score:{scorer.score() * 100:.4f}")
    if conf.generate.generate_path and conf.generate.sorted_path:
        sort_file(gen_path=generate_path, out_path=sorted_path)


if __name__ == '__main__':
    args = get_arguments()
    conf = get_config(args)
    generate(conf)
