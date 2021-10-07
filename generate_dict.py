#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import argparse
from tqdm import tqdm


def align_pair(align_dict):
    '''
    :param align_dict:key为src token,val为src token对应的tgt token字典，值是词频。如：align_dict={'a':{'a1':3,'a2':4,'a4':2},'b':{'b1':3,'b2':2,'b3':5},'c':{'c1':5,'c2':42,'c4':4}}
    :return: aligned_dict,按照src和tgt对齐频率，取最高的作为对齐结果。如：{'a': 'a2', 'b': 'b3', 'c': 'c2'}
    '''
    aligned_dict={}
    for k,v in align_dict.items():
        v=sorted(v.items(),key=lambda item:item[1],reverse=True)
        aligned_dict[k]=v[0][0]
    return aligned_dict

def count_freq(align_path,src_path,tgt_path):
    '''
    :return: 对齐的词频表,key是src token，val是src对应的tgt以及词频的字典，如：{src1:{tgt1:2,tgt2:3...},...}
    '''
    align_dict={} #eg：{'I': {'Ich': 1}, 'declare': {'erkläre': 1}, 'resumed': {'erkläre': 1, 'am': 1}, 'the': {'die': 1}, 'session': {'erkläre': 1}}
    with open(align_path, 'r', encoding='utf-8') as align_f ,open(src_path, 'r', encoding='utf-8') as src_f, open(tgt_path, 'r', encoding='utf-8') as tgt_f:
        for align_line,src_line,tgt_line in tqdm(zip(align_f.readlines(),src_f.readlines(),tgt_f.readlines())):
            align_line,src_line,tgt_line=align_line.strip(),src_line.strip(),tgt_line.strip()
            if (not align_line) or (not src_line) or (not tgt_line):
                continue
            align_line,src_line,tgt_line=align_line.split(),src_line.split(),tgt_line.split()
            # 迭代align，统计词频
            for align in align_line:
                src_idx,tgt_idx=align.split('-')
                src_word,tgt_word=src_line[int(src_idx)],tgt_line[int(tgt_idx)]
                # 初始化
                if src_word not in align_dict.keys():
                    align_dict[src_word]={}
                # 统计词频，并更新
                freq=align_dict[src_word].get(tgt_word,0)+1
                align_dict[src_word][tgt_word]=freq
    return align_dict


def generate_aligned_dict(align_path,src_path,tgt_path,out_path):
    print('aligning...')
    align_dict=count_freq(align_path, src_path, tgt_path)
    print('sorting...')
    aligned_dict=align_pair(align_dict)
    with open(out_path, 'w',encoding='utf-8') as f:
        json.dump(aligned_dict, f,ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate aligned dict', add_help=True)
    parser.add_argument('-a', '--align', default='output/final.align', type=str)
    parser.add_argument('-s', '--src', default='wmt14_en_de/train.en', type=str)
    parser.add_argument('-t', '--tgt', default='wmt14_en_de/train.de', type=str)
    parser.add_argument('-o', '--out', default='output/align_dict.json', type=str)
    args = parser.parse_args()

    generate_aligned_dict(align_path=args.align,src_path=args.src,tgt_path=args.tgt,out_path=args.out)