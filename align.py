import torch
import paddle
from models import convs2s_wmt_en_de
path='ckpt/checkpoint_last.pt'
# out=np.load('fconv_dec.npy',allow_pickle=True)[0]
# 1.1获取torch权重和键值
torch_weights = torch.load(path)['model']
torch_keys=[k for k in torch_weights.keys()]

# 1.2获取paddle权重和键值
model=convs2s_wmt_en_de(is_test=False,
                     src_vocab_size=42243,
                     tgt_vocab_size=43676,
                     max_src_positions=1024,
                     max_tgt_positions=1024,
                     bos_id=2,
                     eos_id=2,
                    beam_size=5,
                    max_out_len=50,
                    rel_len=True,
                        alpha=0.6)
paddle_keys=[k for k in model.state_dict().keys()]
paddle_weights=model.state_dict()

print(torch_weights['encoder.fc2.weight_g'].shape)
print(torch_weights['encoder.fc2.weight_v'].shape)

# # 打印
print('torch keys')
# 打印torch参数名和shape
for k, v in torch_weights.items():  # 可以轻易找到对应key
    print(k, list(v.shape))

print('paddle keys')
# 打印参数名字和对应的shape
for k, v in paddle_weights.items():
    print(k, v.shape)


# torch key转换规则
torch_to_paddle_keys={}
skip_weights = ["decoder.version"]

# 关注维度是2的
donot_transpose=[
    "encoder.embed_tokens.weight",'encoder.embed_positions.weight','decoder.embed_tokens.weight','decoder.embed_positions.weight']

'''
Namespace(arch='fconv_wmt_en_de', clip_norm=0.1, data='/private/home/edunov/wmt14_en_de', decoder_attention='True', 
decoder_embed_dim=768, decoder_layers='[(512, 3)] * 9 + [(1024, 3)] * 4 + [(2048, 1)] * 2', decoder_out_embed_dim=512, dropout=0.1, 
encoder_embed_dim=768, encoder_layers='[(512, 3)] * 9 + [(1024, 3)] * 4 + [(2048, 1)] * 2', force_anneal=26, label_smoothing=0.0, 
log_interval=500, lr=1.25, lrshrink=0.1, max_epoch=0, max_positions=1024, max_tokens=4000, min_lr=1e-05, model='fconv', momentum=0.99, 
no_epoch_checkpoints=False, no_progress_bar=True, no_save=False, restore_file='checkpoint_last.pt', sample_without_replacement=0, 
save_dir='checkpoints', save_interval=-1, seed=1, source_lang='en', target_lang='de', 
test_subset='test', train_subset='train', valid_subset='valid', weight_decay=0.0, workers=4)
'''


for i,(tk, tw) in enumerate(torch_weights.items()):
    transpose = False  # 仅做打印
    # 跳过不需要的w
    if tk in skip_weights:
        continue
    # 转置linear的weight
    if tk.find('.weight') != -1: # 单独weight的只有embed
        if tk not in donot_transpose:
            tail=tk.split('.')[-1]
            if tail=='weight_g':
                tw=tw.squeeze()
            elif tail=='weight_v':
                if tw.ndim == 2:
                    tw = tw.transpose(0, 1)
                    transpose = True
                elif tw.ndim==3:
                    tw = tw.permute(2,1,0)
                    transpose = True
            else:
                print('w err')

    # 转换key名
    pk = tk
    for k, v in torch_to_paddle_keys.items():
        pk = pk.replace(k, v)
    print(f"[{i}/{len(torch_weights)}]Converting: {tk} => {pk} | is_transpose {transpose}")
    paddle_weights[pk] = tw.cpu().detach().numpy()
paddle.save(paddle_weights, 'convs2s.pdparams')
print('save over')


# 打印不一样的
def check_different(ls1,ls2): #torch和paddle的keys
    # len
    print(f'len: ls1: {len(ls1)} | ls2: {len(ls2)}')
    ls_inter=[] # 相交部分
    ls_paddle=[] # paddle中多的
    ls_torch=[] # torch中多的
    # 过滤：
    # for i,pkey in enumerate(ls2):
    #     ls2[i]=pkey_filter(pkey)
    # print('filter over')
    for k1 in ls1:
        if k1 in ls2:
            ls_inter.append(k1)
        else:
            ls_torch.append(k1)
    for k2 in ls2:
        if k2 not in ls1:
            ls_paddle.append(k2)
    print(f'Intersection num: {len(ls_inter)} | Torch keys not aligned: {len(ls_torch)} | Paddle keys not aligned: {len(ls_paddle)}')
    return ls_inter,ls_torch,ls_paddle
ls_inter,ls_torch,ls_paddle=check_different(torch_keys,paddle_keys)
print(f'torch 多了:{ls_torch} | paddle多了:{ls_paddle}')

def pair_info(torch_keys,torch_weights,paddle_keys,paddle_weights):
    for tkeys in torch_keys:
        if tkeys in paddle_keys:
            print(f'torch key: {tkeys}  | paddle key: {tkeys}')
            print(f'torch weight: {list(torch_weights[tkeys].shape)}  | paddle weight: {paddle_weights[tkeys].shape}')
        else:
            print(f'torch key: {tkeys}  | torch weight: {list(torch_weights[tkeys].shape)}')
        print('**' * 50)

# pair_info(torch_keys,torch_weights,paddle_keys,paddle_weights)

'''
torch 多了:['encoder.fc2.weight_g', 'encoder.fc2.weight_v'] | paddle多了:['encoder.fc2.weight']
'''

'''
对齐规则:
1.子网络:
    -encoder
        - embed
            - tokens
            - positions
        - projections
        - convolutions
        - fc
    -decoder
        - embed...
        - projections
        - convolutions
        - attention
            - in proj
            - out proj
        - fc
2.权重:
    weight (embed)
    bias
    weight_g (squeeze即可)
        - conv  (3dim)
        - fc,attn,proj (2dim)
    weight_v 
        - conv  transpose(2,1,0)
        - fc,attn,proj transpose(1,0)
3.规则:
    a.所有的weight squeeze #embed和weight_g,bias搞定，先不squeeze，否则conv中k为1就没了 (squeeze weight_g)
    b.如果key含weight_v:
        如果含conv且维度三维: transpose(2,1,0) #conv搞定
        如果维度两维: transpose(1,0) #fc,attn,proj搞定
'''

'''
embed：直接赋值 （检查pos产生的对不对）
除了embed全用了weight_norm，都要g和v,且都有bias（其实就fc、conv两类）
fc：
proj
conv
attn（也是proj）
总结：
1.名含fc、proj的是全连接，v需要转置(1,0)，g需要squeeze，bias不变
2.名含conv的是卷积，v需要转置(2,1,0)，g需要squeeze，bias不变
3.含embed的直接赋值
'''
def align2paddle(torch_weights, paddle_weights):
    paddle_keys = [k for k in paddle_weights.keys()]
    torch_keys = [k for k in torch_weights.keys()]
    for pk in paddle_keys:
        if pk in torch_keys:
            torch_w=torch_weights[pk].detach().cpu()
            last_name =pk.split('.')[-1]
            # 对齐嵌入
            if 'embed' in pk or last_name=='bias':
                paddle_weights[pk] = torch_w.numpy()
            # 对齐卷积
            elif 'convolutions' in pk:
                if last_name=='weight_g':
                    paddle_weights[pk] = torch_w.squeeze().numpy()
                elif last_name=='weight_v':
                    paddle_weights[pk] = torch_w.numpy().transpose(2, 1, 0) #[k in out]->[out in k]
            # 对齐全连接（含attention）
            elif 'fc' in pk or 'proj' in pk:
                if last_name=='weight_g':
                    paddle_weights[pk] = torch_w.squeeze().numpy()
                elif last_name=='weight_v':
                    paddle_weights[pk] = torch_w.numpy().transpose(1, 0) #[out in]->[in out]
        else:
            print(f'key not alligned:{pk}')
    return paddle_weights

# pad_weights=align2paddle(torch_weights,paddle_weights)
# print(pad_weights)
# paddle.save(pad_weights, 'ckpt/last/new_align.pdparams')


def allign2torch(torch_weights,paddle_weights):
    torch_keys = [k for k in torch_weights.keys()]
    paddle_keys = [k for k in paddle_weights.keys()]

    for tk in torch_keys:
        if tk in paddle_keys:
            paddle_w = paddle_weights[tk].numpy()
            # 对weight_g添加维度
            if tk.find('weight_g') != -1: # torch 3dim前两位是1，2dim后一位是1
                if len(torch_weights[tk].shape)==3:
                    torch_weights[tk] = torch.from_numpy(paddle_w).unsqueeze(0).unsqueeze(0)
                elif len(torch_weights[tk].shape)==2:
                    torch_weights[tk] = torch.from_numpy(paddle_w).unsqueeze(1)
            # 其他权重（维度相等，或者直接赋值、或者转置）
            if list(torch_weights[tk].shape) == paddle_w.shape:
                torch_weights[tk] = torch.from_numpy(paddle_w)
            elif tk.find('weight_v')!=-1 and len(paddle_w.shape)==3:
                torch_weights[tk] = torch.from_numpy(paddle_w.transpose(2, 1, 0))
            elif tk.find('weight_v')!=-1 and len(paddle_w.shape)==2:
                torch_weights[tk] = torch.from_numpy(paddle_w.transpose(1, 0))
        else:
            print(f'key not alligned:{tk}')
    print('aligned over!')
    return torch_weights

# 转换成torch权重
t_path='ckpt_ro/checkpoint_last.pt'
p_path='ckpt_ro/epoch_100/convs2s.pdparams'
torch_weights = torch.load(t_path)
paddle_weights=paddle.load(p_path)

tmodel_weights=allign2torch(torch_weights['model'],paddle_weights)
torch_weights['model']=tmodel_weights
torch.save(torch_weights,'ckpt_ro/checkpoint_enro100.pt')


