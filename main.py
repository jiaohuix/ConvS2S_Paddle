import os
import time
import yaml
import argparse
from data import prep_dataset,prep_loader,prep_vocab
from utils import same_seeds
from train import train_model
from predict import predict
from attrdict import AttrDict
from utils import get_logger
from models import build_model

def main(args):
    conf = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    same_seeds(seed=conf.train.random_seed)
    logger = get_logger(loggername=f"ConvS2S", save_path=conf.SAVE)
    if args.mode=='train':
        logger.info('Prep | Preparaing train datasets...')
        t1 = time.time()
        train_dset = prep_dataset(conf, mode='train')
        dev_dset = prep_dataset(conf, mode='dev')
        logger.info(f'datasets load cost:{time.time()-t1}')
        train_loader = prep_loader(conf, train_dset, mode='train')
        dev_loader = prep_loader(conf, dev_dset, mode='dev')
        logger.info(f'Prep | Train num:{len(train_loader.dataset)} | Val num:{len(dev_loader.dataset)} | Cost {time.time() - t1} s ')
        logger.info(f'cfg:{conf}')
        logger.info('Prep | Loading models...')
        model = build_model(conf, is_test=False)
        logger.info('Train | Training...')
        train_model(conf,model,train_loader,dev_loader,logger=logger)

    elif args.mode=='pred':
        test_dset=prep_dataset(conf,mode='test')
        test_loader = prep_loader(conf,test_dset, mode='test')
        logger.info('Prep | Loading models...')
        model = build_model(conf, is_test=True)
        to_tokens=prep_vocab(conf)[1].to_tokens
        logger.info('Pred | Predicting...')
        predict(conf,model,test_loader,to_tokens=to_tokens,logger=logger)

    else:
        logger.info('Mode error!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paddle Model Training', add_help=True)
    parser.add_argument('-c', '--config', default='config/ru2zh_ifly.yaml', type=str, metavar='FILE', help='yaml file path')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'pred'])
    args = parser.parse_args()
    main(args)
    '''
    multi:
    python main_multi_gpu.py --cfg configs/ruzh_ifly.yaml --max-epoch 60 --save-epoch 5 --amp --accum-iter 4 --src-lang zh --tgt-lang ru --max-tokens 3000 --patience 5
    python main_multi_gpu.py --cfg configs/en2ro.yaml --eval --pretrained ckpt_ro/88
    python generate.py --cfg configs/en2ro.yaml --pretrained ckpt_ro/88 
    '''