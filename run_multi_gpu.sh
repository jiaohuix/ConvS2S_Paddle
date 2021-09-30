pip install -r requirements.txt

datasets_prefix="/root/paddlejob/workspace/train_data/datasets/data110574/"
zip_name="wmt14ende_newstest.zip"
data_path=$datasets_prefix$zip_name
ckpt_path='/root/paddlejob/workspace/train_data/datasets/data110631/ckpt.zip'

echo "Unpacking data..."
#tar xvf $path
unzip $data_path

echo "Unpacking data..."
unzip $ckpt_path

# train model
echo "Training model"
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_conv_workspace_size_limit=1024
python main_multi_gpu.py --ngpus 4 --last_epoch 110 --config config/ende_news.yaml --resume ckpt/epoch_110