# install fast_align
git clone https://hub.fastgit.org/clab/fast_align.git
tar -zvf fast_align.tar.gz
sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../..

# align src and tgt
src_path="./wmt16_enro_bpe/corpus.bpe.en"
tgt_path="./wmt16_enro_bpe/corpus.bpe.ro"

# merge
#paste -d "\t" $src_path $tgt_path > output/train.tmp
python merge_src_tgt.py --src $src_path --tgt $tgt_path --out output/train.parallel || exit 1

# align
./fast_align/build/fast_align -i output/train.parallel -d -o -v > output/forward.align
./fast_align/build/fast_align -i output/train.parallel -d -o -v -r > output/reverse.align
./fast_align/build/atools -i output/forward.align -j output/reverse.align -c grow-diag-final-and > output/final.align
#rm  output/train.parallel

# generate aligned dict by frequence
python generate_dict.py --align output/final.align --src $src_path --tgt $tgt_path --out output/align_dict.json || exit 1

echo 'aligned dict build over!'
