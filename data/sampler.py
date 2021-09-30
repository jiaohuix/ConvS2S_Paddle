import math
import numpy as np
from paddle.io import BatchSampler
from itertools import chain

# 注：与torch不同，paddle的dataloader里面穿的batchsampler必须继承batchsampler
class DynamicBatchSampler(BatchSampler):
    def __init__(self,data_source, sampler=None,shuffle=False, num_buckets=100, min_size=0, max_size=1000,
                 max_tokens=None, max_sentences=None, bsz_factor=1,drop_last=False):
        """
        参考：https://zhuanlan.zhihu.com/p/100135945
        修改：
        1.支持shuffle
        2.采样时不需要计算idx长度，在data_source里已经存了，直接索引
        3.修改full_batch，使结果严格小于max_tokens
        4.支持max_sentence和factor
        ---------------------------------------------------------------------------------------------------
        :param data_source: src和tgt句子最大长度的dataset #sampler的dataset只能是paddle dataset，或者iterdataset
        :param sampler: 或Rand、Sequence sampler，或None # 其dataset是src和tgt最大句子长度
        :param shuffle: sampler为None时，按shuffle决定是随机还是顺序取
        :param num_buckets: 利用桶原理将相似长度的样本放在一个batchsize中，桶的数量 ?
        :param min_size: 最小长度的样本，小于这个值的样本会被过滤掉。依据这个值来创建样桶
        :param max_size: 最大长度的样本
        :param max_sentences: 若非空，则取min(batchsize,max_sentences)
        :param bsz_factor: 若max_sentences为None时，bsz_factor（提前预判）
        -------------------------------------------------------------
        """
        super(DynamicBatchSampler, self).__init__(data_source,shuffle=shuffle)
        self.data_source=data_source
        if sampler is not None:
            self.sampler = sampler
        else:  # 如果sampler非空则取父类按shuffle初始化的sampler
            self.sampler = self.sampler

        self.num_tokens_fn = lambda idx:self.data_source[idx]+1 # 长度dset,一定要加eos或sos！！
        # self.num_tokens_fn = lambda idx:max(len(self.data_source[idx][0]),len(self.data_source[idx][1])) # raw dset
        self.num_buckets = num_buckets
        self.min_size = min_size
        self.max_size = max_size
        self.bsz_factor=bsz_factor
        self.drop_last=drop_last

        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.drop_last = drop_last

    def is_full_batch(self,num_tokens,batch_idxs,factor_tokens,next_max_len=None):
        ''' 判断是否返回一个batch
          重要！在能被factor整除的if中，elif可以提前预判下一个factor会不会长度突变，从而超过max_tokens,该情况下直接返回。
          例1：bsz=120,len=31,if中预测bsz=128,num_tokens=3968才可返回;在bsz=126,len=【31】,num_tokens=3906，未越界;
          而bsz=127,len=【32】,num_tokens=4064,突然超过max_token后上一个if就失效了，所以要在120时判断128的token数是否越界。
          例2：返回的索引是128*31，但pad时多了sos或eos，变长了128*32>4000，所以计算索引句长度要显式加1
        '''
        batch_len=len(batch_idxs)
        if self.max_sentences is not None:
            # avg_tokens = factor_tokens / self.bsz_factor  # 平均每句话词数
            # max_bsz = self.max_tokens // avg_tokens
            # max_bsz=max_bsz-max_bsz%self.bsz_factor # 去余数
            # if batch_len == min(max_bsz,self.max_sentences):
            if batch_len == self.max_sentences:
                return True
        elif batch_len % self.bsz_factor==0: # 返回factor倍数的bsz
            assert self.max_tokens is not None,'Must specify max_tokens'
            # 在等长度条件下，预判是否还够一个factor
            if num_tokens<=self.max_tokens and (num_tokens+factor_tokens)>self.max_tokens:
                return True
            # 在长度不等时(渐增长)，预判下一个factor是否突变越界
            elif next_max_len is not None and (batch_len+self.bsz_factor)*next_max_len>self.max_tokens:
                return True
        else:
            return False

    def __iter__(self):
        buckets = [[] for _ in range(self.num_buckets)] # 为什么要这么多桶子？是不是放不同长度
        buckets_len = [0] * self.num_buckets # 每个桶中句子的最大长度

        # process equal len batch
        for idx in self.sampler: # 随机的idx，然后取一个bucket句子
            idx_len = self.num_tokens_fn(idx)
            len_ratio=(idx_len - self.min_size) / (self.max_size - self.min_size + 1) #长度百分比,+1确保索引取不到max tokens
            bucket_id = math.floor(len_ratio * self.num_buckets) # 放第几个桶
            buckets_len[bucket_id] = max(buckets_len[bucket_id], idx_len) # 当前桶最大token数
            num_tokens = len(buckets[bucket_id]) * buckets_len[bucket_id] # 句子数*最大长度=pad后batcht token个数
            factor_tokens=self.bsz_factor * buckets_len[bucket_id] # 1factor的batch中含token数
            if self.is_full_batch(num_tokens,buckets[bucket_id],factor_tokens): # 判断当前桶是否满，若满了则返回
                # yield this batch
                yield buckets[bucket_id]
                buckets[bucket_id] = [] # 清空该bucket
                buckets_len[bucket_id] = 0 # 清空该bucket长度

            buckets[bucket_id].append(idx) # 若不满，则往该桶中加入句子

        # process unequal len batch
        residual_batch = []
        bucket_len = 0
        residual_all = [idx for bucket in buckets for idx in bucket] # 把剩下的索引按最大句长从小到大排列
        for i, idx in enumerate(residual_all):  # residual_all不是迭代器，可以被索引
            idx_len = self.num_tokens_fn(idx)
            bucket_len=max(bucket_len,idx_len) # bucket内最大元素的值 【模拟最长句子长度】，这里应该修改，因为大小只会越来越大
            num_tokens = len(residual_batch)  * bucket_len # 当前batch总token数
            factor_tokens=self.bsz_factor * bucket_len # 一个factor句子含token数
            # 求下一个factor的最大句子长度，防止突变超过max_tokens
            if i+self.bsz_factor<len(residual_all):
                next_max_len =self.num_tokens_fn(residual_all[i+self.bsz_factor])
            else:
                next_max_len=None
            if self.is_full_batch(num_tokens, residual_batch, factor_tokens, next_max_len):
                yield residual_batch
                residual_batch = []
                bucket_len = 0

            residual_batch.append(idx)

        # process last batch
        if len(residual_batch)>0 and not self.drop_last:
            yield residual_batch

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass

# 注：与torch不同，paddle的dataloader里面穿的batchsampler必须继承batchsampler
class DynamicTestSampler(BatchSampler):
    def __init__(self,data_source, max_tokens=None, max_sentences=None,drop_last=False):
        '''
        用于测试集的采样，将索引按照句长递增排列，然后取max_tokens的索引。
        '''
        super(DynamicTestSampler, self).__init__(data_source)
        self.data_source=data_source
        self.num_tokens_fn = lambda idx:self.data_source[idx]+1 # 长度dset,一定要加eos或sos！！
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.drop_last=drop_last

    def is_full_batch(self, num_tokens, batch_indices):
        if num_tokens>self.max_tokens:
            return True
        if (self.max_sentences is not None) and (len(batch_indices)==self.max_sentences):
            return True
        return False

    def __iter__(self):
        sentences_len=np.array([sent_len+1 for sent_len in self.data_source]) #长度dset,要加eos或sos
        sort_indices=np.argsort(sentences_len).tolist() # 句长递增索引
        _sample_iter = iter(sort_indices)
        batch_indices = [] #一个batch索引
        max_len=0 #一个batch最大长度
        for idx in _sample_iter:
            max_len=max(max_len,self.num_tokens_fn(idx))
            num_tokens=max_len*len(batch_indices)
            batch_indices.append(idx)

            if self.is_full_batch(num_tokens,batch_indices):
                yield batch_indices
                batch_indices = []
                max_len=0

        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass



class DistributedDynamicBatchSampler(BatchSampler):
    ''' 支持多卡训练的动态bsz采样器
        目前有两个bug:
            1.在迭代结束后，程序会无故逗留1.5min
            2.bsz_factor设置8以上会报错：SystemError: (Fatal) Blocking queue is killed because the data reader raises an exception. 暂时设置4即可
        另外：wmt14en_de平均句长33tokens，2000tokens≈64bsz [paddle要12g]，4000≈128bsz [paddle 24g,fair12g,原因是conv1d算子快但是占用显存大],也就是说32g大概6000tokens无恙
    '''
    def __init__(self,
                 data_source,
                 num_buckets=1024,
                 min_size=0,
                 max_size=1024,
                 max_tokens=4000,
                 max_sentences=None,
                 bsz_factor=1,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        self.data_source=data_source

        self.num_tokens_fn = lambda idx:self.data_source[idx]+1 # 长度dset,一定要加eos或sos！！
        self.num_buckets = num_buckets
        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        self.min_size = min_size
        self.max_size = max_size
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        assert isinstance(bsz_factor, int) and bsz_factor > 0, \
                "bsz_factor should be a positive integer"
        self.bsz_factor = bsz_factor
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"
        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last # 如果多余了就不变，bool不变，然后删除最后一个；如果没多余
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.data_source) * 1.0 / self.nranks)) # 向上取整，强行凑满nrank张卡的倍数，ceil(105/4)=27
        # self.total_size = self.num_samples * self.nranks # 比实际samples大，因为上一步向上取整了 ,108

    def __iter__(self):
        # get indices
        num_samples=len(self.data_source)
        indices=np.arange(num_samples).tolist()
        # shuffle
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch+=1

        # subsample
        batches_indices=self._get_batches_by_max_tokens(indices)

        # yield batch indices
        _batch_iter=iter(batches_indices)
        for i,batch_indices in enumerate(_batch_iter):
            yield batch_indices

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass

    def set_epoch(self,epoch):
        '''
         Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        '''
        self.epoch = epoch

    def is_full_batch(self,num_tokens,batch_idxs,factor_tokens,next_max_len=None):
        ''' 判断是否返回一个batch
          重要！在能被factor整除的if中，elif可以提前预判下一个factor会不会长度突变，从而超过max_tokens,该情况下直接返回。
          例1：bsz=120,len=31,if中预测bsz=128,num_tokens=3968才可返回;在bsz=126,len=【31】,num_tokens=3906，未越界;
          而bsz=127,len=【32】,num_tokens=4064,突然超过max_token后上一个if就失效了，所以要在120时判断128的token数是否越界。
          例2：返回的索引是128*31，但pad时多了sos或eos，变长了128*32>4000，所以计算索引句长度要显式加1
        '''
        batch_len=len(batch_idxs)
        if self.max_sentences is not None: #这里出了问题， 句子数达到最大，bsz怎么算
            # avg_tokens=factor_tokens/self.bsz_factor # 平均每句话词数
            # max_bsz=self.max_tokens//avg_tokens
            # if next_max_len is not None:
            #     max_bsz=self.max_tokens//next_max_len
            # max_bsz=max_bsz-max_bsz%self.bsz_factor # 去余数
            # if batch_len == min(max_bsz,self.max_sentences):
            #     return True
            if batch_len == self.max_sentences:
                return True
        elif batch_len % self.bsz_factor==0: # 返回factor倍数的bsz
            # assert self.max_tokens is not None,'Must specify max_tokens'
            # 在等长度条件下，预判是否还够一个factor
            if num_tokens<=self.max_tokens and (num_tokens+factor_tokens)>self.max_tokens and next_max_len is None:
                return True
            # 在长度不等时(渐增长)，预判下一个factor是否突变越界
            elif next_max_len is not None and (batch_len+self.bsz_factor)*next_max_len>self.max_tokens:
                return True
        else:
            return False
        return False

    # def is_full_batch(self,num_tokens,batch_idxs,factor_tokens,next_max_len=None): # factor_tokens和next_max_len本质一样，不应该按照当前的长度算，而是下一个factor的长度
    #     batch_len=len(batch_idxs)
    #     if next_max_len is None: # equal len batch
    #         if batch_len % self.bsz_factor==0 and num_tokens<=self.max_tokens and (num_tokens+factor_tokens)>self.max_tokens: # 返回factor倍数的bsz  # 其实这里也可能增加，
    #             return True
    #         else:
    #             return False
    #     else:
    #         # 在长度不等时(渐增长)，预判下一个factor是否突变越界
    #         if batch_len % self.bsz_factor==0 and num_tokens<=self.max_tokens and (batch_len+self.bsz_factor)*next_max_len>self.max_tokens:
    #             return True
    #         else:
    #             return False

    # def is_full_batch(self,num_tokens,batch_ids,factor_tokens,next_max_len=None):
    #     batch_len=len(batch_ids)
    #     if next_max_len is None:
    #         if (batch_len%self.bsz_factor==0) and (num_tokens<=self.max_tokens) and (num_tokens+factor_tokens)>self.max_tokens:
    #         # if (batch_len%self.bsz_factor==0) and (num_tokens>=self.max_tokens) :
    #             return True
    #                 # (batch_len+self.bsz_factor)*next_max_len>self.max_tokens:
    #     else:
    #         if (batch_len%self.bsz_factor==0) and (num_tokens<=self.max_tokens) and (batch_len+self.bsz_factor)*next_max_len>self.max_tokens:
    #             return True
    #     return False


    # def is_full_batch(self,num_tokens,batch_idxs,factor_tokens,is_residual=False):
    #     if len(batch_idxs) == 0:
    #         return False
    #     if len(batch_idxs) == self.max_sentences:
    #         return True
    #     if num_tokens > self.max_tokens:
    #         return True
    #     return False

    def _get_batches_by_max_tokens(self,indices):
        batches_indices=[]
        buckets = [[] for _ in range(self.num_buckets)]
        buckets_len = [0] * self.num_buckets  # 每个桶中句子的最大长度

        # process equal len batch （其实这里也不均衡，不是每个桶size都一样，否则为什么要用max计算呢？）
        _sample_iter = iter(indices)
        for i,idx in enumerate(_sample_iter):  # 随机的idx，然后取一个bucket句子
            idx_len = self.num_tokens_fn(idx)
            len_ratio = (idx_len - self.min_size) / (self.max_size - self.min_size + 1)  # 长度百分比,+1确保索引取不到max tokens
            bucket_id = math.floor(len_ratio * self.num_buckets)  # 放第几个桶
            buckets_len[bucket_id] = max(buckets_len[bucket_id], idx_len)  # 当前桶最大token数
            num_tokens = len(buckets[bucket_id]) * buckets_len[bucket_id]  # 句子数*最大长度=pad后batcht token个数
            # if i+self.bsz_factor < len(indices):
            #     next_max_len=max(buckets_len[bucket_id],self.num_tokens_fn(indices[i+self.bsz_factor])) # 不能确保后面八个是自己人啊！！！
            # else:
            #     next_max_len=max(buckets_len[bucket_id],self.num_tokens_fn(indices[-1])) # 不足factor取最后一个
            factor_tokens = self.bsz_factor * buckets_len[bucket_id]  # 1factor的batch中含token数
            if self.is_full_batch(num_tokens, buckets[bucket_id],factor_tokens, next_max_len=None):  # 只考虑当前桶满了没
                # yield this batch
                batches_indices.append(buckets[bucket_id])
                buckets[bucket_id] = []  # 清空该bucket
                buckets_len[bucket_id] = 0  # 清空该bucket长度   # 是不是被误清空了？？？？？？？？？？？

                # now_total=len(list(chain(*batches_indices))) # 实际取
                # res_should=len(self.data_source)-len(list(chain(*batches_indices))) # 理应剩余
                # buc=len(list(chain(*buckets))) # 桶内剩余
                # res_real=buc+len(indices)-(i+1)
                # if res_should==39374:
                #     res_should=res_should
                # print(f'in id {bucket_id} now:{now_total} | res_should:{res_should} | res_real:{res_real}| bucket :{buc}')
                continue

            buckets[bucket_id].append(idx)  # 若不满，则往该桶中加入句子
            # now_total = len(list(chain(*batches_indices)))  # 实际取
            # res_should = len(self.data_source) - len(list(chain(*batches_indices)))  # 理应剩余
            # buc = len(list(chain(*buckets)))  # 桶内剩余
            # res_real = buc + len(indices) - (i + 1)
            # print(f'out id {bucket_id} now:{now_total} | res_should:{res_should} | res_real:{res_real}| bucket num:{buc}')


        # print(f'1轮已有:{len(list(chain(*batches_indices)))} | 应当剩下:{len(self.data_source)-len(list(chain(*batches_indices)))} 实际还有：{len(list(chain(*buckets)))}') # sample len:36396 | raw:39414   res=3018
        # now_total = len(list(chain(*batches_indices)))  # 实际取
        # res = len(self.data_source) - len(list(chain(*batches_indices)))  # 理应剩余
        # buc = len(list(chain(*buckets)))  # 桶内剩余
        # print(f'out !!!!!! now:{now_total} | res:{res} | bucket :{buc}', res == buc)
        # process unequal len batch
        residual_batch = []
        bucket_len = 0
        # residual_all = [idx for bucket in buckets for idx in bucket]  # 把剩下的索引按最大句长从小到大排列
        residual_all = list(chain(*buckets))  # 把剩下的索引按最大句长从小到大排列
        _sample_iter = iter(residual_all)
        for i, idx in enumerate(_sample_iter):  # residual_all不是迭代器，可以被索引
            idx_len = self.num_tokens_fn(idx)
            bucket_len = max(bucket_len, idx_len)  # bucket内最大元素的值 【模拟最长句子长度】，这里应该修改，因为大小只会越来越大
            num_tokens = len(residual_batch) * bucket_len  # 当前batch总token数

            # if i+self.bsz_factor < len(residual_all):
            #     next_max_len=self.num_tokens_fn(residual_all[i+self.bsz_factor])
            # else:
            #     next_max_len=self.num_tokens_fn(residual_all[-1]) # 不足factor取最后一个


            factor_tokens = self.bsz_factor * bucket_len  # 一个factor句子含token数
            # 求下一个factor的最大句子长度，防止突变超过max_tokens，如果没有下一个factor怎么办
            if i + self.bsz_factor < len(residual_all): # 这也不合理，应该是+factor的余数。。
                next_max_len = self.num_tokens_fn(residual_all[i + self.bsz_factor])
            else:
                next_max_len = None
            if self.is_full_batch(num_tokens, residual_batch, factor_tokens,next_max_len):
                batches_indices.append(residual_batch)
                residual_batch = []
                bucket_len = 0
            else:
                residual_batch.append(idx)
        # print(f'2轮已有:{len(list(chain(*batches_indices)))} | 应当剩下:{len(self.data_source)-len(list(chain(*batches_indices)))} 实际还有：{len(residual_batch)}') # sample len:36396 | raw:39414   res=3018
        # process last batch
        if not self.drop_last and len(residual_batch)>0:
            batches_indices.append(residual_batch)
        all_len=len(list(chain(*batches_indices)))
        # print(f'last sample len:{all_len} | raw:{len(self.data_source)}')
        if self.nranks>1:
            local_batches_indices=[]
            last_batches=len(batches_indices)%self.nranks # 多余的batch
            # 补全batches
            if last_batches>0:
                batches_indices.extend(batches_indices[:(self.nranks-last_batches)])
            assert len(batches_indices) % self.nranks == 0  # 确保batch数是nrank的倍数

            # sabsample for each process
            for i in range(0,len(batches_indices),self.nranks):
                local_batches_indices.append(batches_indices[i])
            return local_batches_indices
        # single process
        return batches_indices

'''
明日：
1.处理last
2.indices不用补全
3.sentence
'''