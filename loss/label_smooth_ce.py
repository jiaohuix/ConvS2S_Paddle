import paddle
import paddle.nn as nn
import paddle.nn.functional as F
'''
参考fair的实现，速度太慢，遂废弃
'''

def get_perplexity(loss, ndigits=2, base=2):
    if loss is None:
        return 0.0
    ppl= base ** min(loss,100)
    return round(ppl,ndigits)

class LabelSmoothedCrossEntropyCriterion(nn.Layer):
    def __init__(self,epsilon, pad_idx=0, ignore_prefix_size=0,sentence_avg=False):
        """
        Computes the cross entropy loss for given input with label smoothing.

        Args:
            label_smooth_eps (float, optional):
                The weight used to mix up the original ground-truth distribution
                and the fixed distribution. Defaults to None. If given, label smoothing
                will be applied on `label`.
            pad_idx (int, optional):
                The token id used to pad variant sequence. Defaults to 0.
            ignore_prefix_size:
                Igrnore fist several tokens.
            sentence_avg: bool, true:return num samples,false:return num tokens，用在除以梯度上？
        """
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.epsilon = epsilon
        self.pad_idx = pad_idx
        self.ignore_prefix_size = ignore_prefix_size
        self.sentence_avg=sentence_avg

    def forward(self, logits, target, reduce=True):
        """
               Computes cross entropy loss with or without label smoothing.

               Args:
                   predict (Tensor):
                       The predict results of `TransformerModel` with shape
                       `[batch_size, sequence_length, vocab_size]` whose data type can
                       be float32 or float64.
                   target (Tensor):
                       The label for correspoding results with shape
                       `[batch_size, sequence_length, 1]`.

              Returns a tuple with three elements:
              1) the loss
              2) the sample size, which is used as the denominator for the gradient  ###？？？？
              3) logging outputs to display while training
 """
        # 句数、词数
        nsentences=target.shape[0]
        ntokens=paddle.cast(target != self.pad_idx, dtype=paddle.get_default_dtype()).sum()
        ntokens.stop_gradient = True
        # 1.计算logit的logrpob
        lprobs, target = self.get_logprobs_target(logits, target)
        # 2. 计算label smooth
        sum_loss,nll_loss=self.label_smoothed_nll_loss(lprobs,target)
        # 4.记录信息
        ppl=get_perplexity(nll_loss.item())
        sample_size=nsentences if self.sentence_avg else ntokens # 句数或token数
        avg_loss,nll_loss=sum_loss/sample_size,nll_loss/sample_size # 求平均loss
        logging_output = {
            "loss": avg_loss.item(),  # 训练
            "nll_loss": nll_loss.item(),  # 计算ppl
            "ppl":ppl,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return sum_loss,avg_loss,sample_size,logging_output

    def get_logprobs_target(self, logits, target):
        '''
        :param logits: model output
        :param target: real label
        :return:
                lprobs:log softmax of logits,and ignore prefix,shape is [bsz*seq_len,vocab_size]
                target:reshape target to 2dim and ignore prefix,shape is [bsz*seq_len,1]
        '''
        # 1.calc logprob
        lprobs = F.log_softmax(logits, axis=-1)
        # 2.ignore prefix
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[self.ignore_prefix_size:, :, :]
            target = target[self.ignore_prefix_size:, :, :]
        # 3.reshape to 2dim
        lprobs = logits.reshape((-1, lprobs.shape[-1]))
        target = target.reshape((-1, 1))
        return lprobs, target

    def label_smoothed_nll_loss(self, lprobs, target, reduce=True):
        '''
        :return:
            loss: cross entropy of log probs and soft label         = soft label crossentropy
            nll_loss: sum of all postive predict log probs,equal    = hard label crossentropy
        '''
        nll_loss = -lprobs.index_sample(target)
        smooth_loss = -lprobs.sum(axis=-1, keepdim=True)
        if self.pad_idx is not None:
            pad_mask = target == self.pad_idx
            nll_loss = paddle.where(pad_mask, paddle.zeros_like(target), target)
            smooth_loss = paddle.where(pad_mask, paddle.zeros_like(target), target)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        K = lprobs.shape[-1]  # vocab_size
        alpha = self.epsilon / (K - 1)  # false pred prob
        # loss=(1-e)*nll+α*(smooth-nll) # (1-eps)*Ztrue+eps*(Ztrue+Zfalse)
        loss = (1.0 - self.epsilon - alpha) * nll_loss + alpha * smooth_loss
        return loss,nll_loss


if __name__ == '__main__':
    from tqdm import tqdm
    v=43676
    bsz,seql=100,30
    logits=paddle.randn((bsz,seql,v))
    target=paddle.randint(0,v,(bsz,seql,1))
    criterion=LabelSmoothedCrossEntropyCriterion(epsilon=0.1,pad_idx=1)
    import time
    for i in tqdm(range(500)):
        if i == 100:  # 启动开销不算
            start = time.time()
        if i == 499:
            end = time.time()
        sum_loss,avg_loss,sample_size,logging_output=criterion(logits,target)
    print('loss avg time: ' + str((end - start)/400)) # 9s

    '''
    Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [17433.90039062])
{'loss': 17433.900390625, 'nll_loss': 19371, 'ppl': 1267650600228229401496703205376, 'ntokens': Tensor(shape=[1], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
       [True]), 'nsentences': 4, 'sample_size': Tensor(shape=[1], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
       [True])}
       nll loss 比smooth loss大
    '''