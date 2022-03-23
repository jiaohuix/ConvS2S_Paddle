# -*- coding:utf-8 -*-
import os
import logging
import math
import random
import paddle
import paddle.nn.functional as F
from paddle.optimizer.lr import ReduceOnPlateau
from paddle import Tensor
import numpy as np

def same_seeds(seed=2021):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def save_model(model,optimizer,save_dir,best_bleu=None):
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    paddle.save(model.state_dict(),os.path.join(save_dir, "convs2s.pdparams"))
    optim_state=optimizer.state_dict()
    if  best_bleu: optim_state['LR_Scheduler'].setdefault('best_bleu',best_bleu) # save best bleu score in optim state
    paddle.save(optim_state,os.path.join(save_dir, "convs2s.pdopt"))

class NMTMetric(paddle.metric.Metric):
    def __init__(self,name='convs2s'):
        self.smooth_loss=0
        self.nll_loss=0
        self.steps=0
        self.gnorm=0
        self._name=name

    @paddle.no_grad()
    def update(self,sum_loss,logits,target,sample_size,pad_id,gnorm):
        '''
        :return: current batch loss,nll_loss,ppl
        '''
        loss=sum_loss/sample_size/math.log(2)
        nll_loss,ppl=calc_ppl(logits,target,sample_size,pad_id)
        self.smooth_loss+=float(loss)
        self.nll_loss+=float(nll_loss)
        self.steps+=1
        self.gnorm+=gnorm
        return loss,nll_loss,ppl

    def accumulate(self):
        '''
        :return:accumulate batches loss,nll_loss,ppl
        '''
        avg_loss=self.smooth_loss/self.steps
        avg_nll_loss=self.nll_loss/self.steps
        ppl=pow(2, min(avg_nll_loss, 100.))
        gnorm=self.gnorm/self.steps
        return avg_loss,avg_nll_loss,ppl,gnorm

    def reset(self):
        self.smooth_loss=0
        self.nll_loss=0
        self.steps=0
        self.gnorm=0

    def name(self):
        """
        Returns metric name
        """
        return self._name

@paddle.no_grad()
def calc_ppl(logits,tgt_tokens,token_num,pad_id,base=2):
    tgt_tokens = tgt_tokens.astype('int64')
    nll = F.cross_entropy(logits, tgt_tokens, reduction='sum',ignore_index=pad_id)  # bsz seq_len 1
    nll_loss = nll / token_num / math.log(2) # hard ce
    nll_loss = min(nll_loss.item(), 100.)
    ppl = pow(base, nll_loss)
    return nll_loss,ppl


class ReduceOnPlateauWithAnnael(ReduceOnPlateau):
    '''
        Reduce learning rate when ``metrics`` has stopped descending. Models often benefit from reducing the learning rate
    by 2 to 10 times once model performance has no longer improvement.
        [When lr is not updated for force_anneal times,then force shrink the lr by factor.]
    '''
    def __init__(self,
                 learning_rate,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 force_anneal=50,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 epsilon=1e-8,
                 verbose=False,
                 ):
        args=dict(locals())
        args.pop("self")
        args.pop("__class__",None)
        self.force_anneal=args.pop('force_anneal')
        super(ReduceOnPlateauWithAnnael,self).__init__(**args)
        self.num_not_updates=0

    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr','num_not_updates'
        ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        # loss must be float, numpy.ndarray or 1-D Tensor with shape [1]
        if isinstance(metrics, (Tensor, np.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                                                                      "should be (1L,), but the current metrics.shape is {}. Maybe that " \
                                                                      "you should call paddle.mean to process it first.".format(metrics.shape)
        elif not isinstance(metrics,
                            (int, float, np.float32, np.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray' or 'paddle.Tensor', but receive {}".
                    format(type(metrics)))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metrics, self.best):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:  # 大于【等于】patience，要更新lr，【并要annel清0】
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                self.num_not_updates=0
                new_lr = max(self.last_lr * self.factor, self.min_lr)
                if self.last_lr - new_lr > self.epsilon:
                    self.last_lr = new_lr
                    if self.verbose:
                        print('Epoch {}: {} set learning rate to {}.'.format(
                            self.last_epoch, self.__class__.__name__,
                            self.last_lr))
            else: # Update here
                self.num_not_updates+=1
                if self.num_not_updates>=self.force_anneal:
                    self.num_not_updates=0
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
                    new_lr = max(self.last_lr * self.factor, self.min_lr)
                    if self.last_lr - new_lr > self.epsilon:
                        self.last_lr = new_lr
                        if self.verbose:
                            print('Epoch {}: {} set learning rate to {} because of force anneal.'.format(
                                self.last_epoch, self.__class__.__name__,
                                self.last_lr))



def force_anneal(scheduler:ReduceOnPlateau,anneal:int):
    setattr(scheduler,'force_anneal',anneal)
    setattr(scheduler,'num_not_updates',0)
    def state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr','num_not_updates'
        ]
    setattr(scheduler,'state_keys',state_keys)

    def step(self, metrics, epoch=None):
        pass
    setattr(scheduler,'step',step)
    return scheduler

def ExpDecayWithWarmup(warmup_steps,lr_start,lr_peak,lr_decay):
    ''' warmup and exponential decay'''
    # exp_sched = paddle.optimizer.lr.ExponentialDecay(learning_rate=lr_peak, gamma=lr_decay)
    exp_sched = ReduceOnPlateauWithAnnael(learning_rate=lr_peak, factor=lr_decay)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=exp_sched, warmup_steps=warmup_steps,
                                                 start_lr=lr_start, end_lr=lr_peak, verbose=True)
    return scheduler


def get_logger(loggername,save_path='.'):
    # 创建一个logger
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    save_path = save_path

    # 创建一个handler，用于写入日志文件
    log_path = os.path.join(save_path,"logs")  # 指定文件输出路径，注意logs是个文件夹，一定要加上/，不然会导致输出路径错误，把logs变成文件名的一部分了
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logname = os.path.join(log_path,f'{loggername}.log') # 指定输出的日志文件名
    fh = logging.FileHandler(logname, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
    fh.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s | %(name)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    待修改：添加unk的处理！
    """
    # sum_eos=sum([t for t in seq if t==eos_idx]) # 预测不出eos 0个或最后一个
    # print(f'sum  eos is :{sum_eos}')
    eos_pos = len(seq) - 1 # 初始化eos索引
    for i, idx in enumerate(seq): # 找eos位置
        if idx == eos_idx: # 第一个eos的位置即可
            # if i==0: # 如果遇到输出bos（本例为eos），则取后一个eos
            #     print('it is bos ')
            #     continue
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1] #取bos和eos中间内容
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)# 控制是否输出bos和eos
    ]
    return seq

def strip_pad(tensor, pad_id):
    return tensor[tensor!=pad_id]

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re
        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(' +', ' ', sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence

def to_string(
        tokens,
        vocab,
        bpe_symbol=None,
        extra_symbols_to_ignore=None,
        separator=" "):
    extra_symbols_to_ignore=set(extra_symbols_to_ignore or [])
    tokens=[int(token) for token in tokens if int(token) not in extra_symbols_to_ignore] # 去掉extra tokens
    sent=separator.join(
        vocab.to_tokens(tokens)
    )
    return post_process(sent,bpe_symbol)


def sort_file(gen_path='generate.txt',out_path='result.txt'):
    result = []
    with open(gen_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.startswith('H-'):
                result.append(line.strip())
    result = sorted(result, key=lambda line: int(line.split('\t')[0].split('-')[1]))
    result = [line.split('\t')[2].strip() for line in result]
    with open(out_path, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(result))
    print(f'write to file {out_path} success.')


def get_grad_norm(grads):
    norms=paddle.stack([paddle.norm(g,p=2) for g in grads])
    gnorm=paddle.norm(norms,p=2)
    return float(gnorm)
