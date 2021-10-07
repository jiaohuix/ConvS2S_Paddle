#!/usr/bin/python
# -*- coding: UTF-8 -*-
import codecs
import argparse
#  paste -d "\align_norm" train.en train.de >train.tmp

def merge(src_path,tgt_path,out_path):
    with codecs.open(src_path,'r',encoding='utf-8') as src_f,codecs.open(tgt_path,'r',encoding='utf-8') as tgt_f, codecs.open(out_path,'w',encoding='utf-8') as fw:
        for src_line,tgt_line in zip(src_f.readlines(),tgt_f.readlines()):
            src_line,tgt_line=src_line.strip(),tgt_line.strip()
            if (not src_line) or (not tgt_line):continue
            merge_line=src_line+' ||| ' +tgt_line
            fw.write(merge_line + '\n' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge src and tgt', add_help=True)
    parser.add_argument('-s', '--src', default='./wmt16_enro_bpe/corpus.bpe.en', type=str)
    parser.add_argument('-t', '--tgt', default='./wmt16_enro_bpe/corpus.bpe.en', type=str)
    parser.add_argument('-o', '--out', default='output/train.prallel', type=str)
    args = parser.parse_args()

    merge(src_path=args.src,tgt_path=args.tgt, out_path=args.out)