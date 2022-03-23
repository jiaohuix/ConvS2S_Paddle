#!/bin/bash
# 参考 https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer
# bash preprocess.sh in_folder out_folder SRC TGT bpe_operations

# input output dir
in_folder=$1
out_folder=$2

# suffix of source language files
SRC=ru
if [ ! -n $3 ];then
  SRC=$3
fi
# suffix of target language files
TRG=zh
if [ ! -n $4 ];then
  TRG=$4
fi

bpe_operations=40000
if [ ! -n $5 ];then
  bpe_operations=$5
fi
pip install subword-nmt

outdir=${out_folder}/${SRC}${TRG}_bpe
# create out dir
if [ ! -e  ${outdir} ];then
  mkdir -p ${outdir}
fi

echo "source learn-bpe and apply-bpe..."
subword-nmt learn-bpe -s ${bpe_operations} <${in_folder}/train.${SRC}> ${outdir}/bpe_codes.${SRC}
echo "apply source train"
subword-nmt apply-bpe -c ${outdir}/bpe_codes.${SRC} <${in_folder}/train.${SRC}> ${outdir}/train.${SRC}
echo "apply source dev"
subword-nmt apply-bpe -c ${outdir}/bpe_codes.${SRC} <${in_folder}/dev.${SRC}-${TRG}.${SRC}> ${outdir}/dev.${SRC}-${TRG}.${SRC}
if [ -e ${in_folder}/dev.${TRG}-${SRC}.${SRC} ];then
  subword-nmt apply-bpe -c ${outdir}/bpe_codes.${SRC} <${in_folder}/dev.${TRG}-${SRC}.${SRC}> ${outdir}/dev.${TRG}-${SRC}.${SRC}
fi
echo "apply source test"
subword-nmt apply-bpe -c ${outdir}/bpe_codes.${SRC} <${in_folder}/test.${SRC}-${TRG}.${SRC}> ${outdir}/test.${SRC}-${TRG}.${SRC}

echo "target learn-bpe and apply-bpe..."
subword-nmt learn-bpe -s ${bpe_operations} <${in_folder}/train.${TRG}> ${outdir}/bpe_codes.${TRG}
echo "apply target train"
subword-nmt apply-bpe -c ${outdir}/bpe_codes.${TRG} <${in_folder}/train.${TRG}> ${outdir}/train.${TRG}
echo "apply target dev"

subword-nmt apply-bpe -c ${outdir}/bpe_codes.${TRG} <${in_folder}/dev.${SRC}-${TRG}.${TRG}> ${outdir}/dev.${SRC}-${TRG}.${TRG}
if [ -e ${in_folder}/dev.${TRG}-${SRC}.${TRG} ];then
  subword-nmt apply-bpe -c ${outdir}/bpe_codes.${TRG} <${in_folder}/dev.${TRG}-${SRC}.${TRG}> ${outdir}/dev.${TRG}-${SRC}.${TRG}
fi
echo "apply target test"
subword-nmt apply-bpe -c ${outdir}/bpe_codes.${TRG} <${in_folder}/test.${TRG}-${SRC}.${TRG}> ${outdir}/test.${TRG}-${SRC}.${TRG}

echo "source get-vocab. if loading pretrained model, use its vocab."
subword-nmt  get-vocab -i ${outdir}/train.${SRC} -o ${outdir}/temp
echo -e "<s>\n</s>\n<unk>\n<pad>" > ${outdir}/vocab.${SRC}
cat ${outdir}/temp | cut -f1 -d ' ' >> ${outdir}/vocab.${SRC}
rm -f ${outdir}/temp

echo "target get-vocab. if loading pretrained model, use its vocab."
subword-nmt  get-vocab -i ${outdir}/train.${TRG} -o ${outdir}/temp
echo -e "<s>\n</s>\n<unk>\n<pad>" > ${outdir}/vocab.${TRG}
cat ${outdir}/temp | cut -f1 -d ' ' >> ${outdir}/vocab.${TRG}
rm -f ${outdir}/temp
