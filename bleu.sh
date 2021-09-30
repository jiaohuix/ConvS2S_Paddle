#git clone https://github.com/moses-smt/mosesdecoder.git
#tar -zxf mosesdecoder.tar.gz
## 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' output/predict.txt > output/predict.tok.txt
sed -r 's/(@@ )|(@@ ?$)//g' output/reference.txt > output/reference.tok.de
# 计算multi-bleu
! perl mosesdecoder/scripts/generic/multi-bleu.perl output/reference.tok.de < output/predict.tok.txt
