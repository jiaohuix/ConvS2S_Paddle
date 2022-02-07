import os
import paddle
import numpy as np
from utils import post_process_seq,get_logger
from tqdm import tqdm
def predict(conf,model,test_loader,to_tokens,logger):
    if conf.train.use_gpu:
        place = "gpu"
    else:
        place = "cpu"
    paddle.set_device(place)
    test_sampler=test_loader.batch_sampler

    # Define data loader
    logger.info(f'Prep | Test num:{len(test_loader.dataset)}  ')

    # Load the trained model
    assert conf.model.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(conf.model.init_from_params, "convs2s.pdparams"))
    logger.info(f'Loaded weights from {conf.model.init_from_params}')
    model.load_dict(model_dict)

    # Set evaluate mode
    model.eval()
    f = open(conf.data.output_file.split(',')[0], "w",encoding='utf-8')
    pred_indices=[]
    with paddle.no_grad():
        for (src_indices,src_tokens) in tqdm(test_loader):
            finished_seq = model(src_tokens=src_tokens)
            finished_seq = finished_seq.numpy().transpose([0, 2, 1]) #[bsz len beam]->[bsz beam len]
            for idx,ins in enumerate(finished_seq): #ins [beam len]
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= conf.generate.n_best: # 取前n个句子
                        break
                    id_list = post_process_seq(beam, conf.model.bos_idx, conf.model.eos_idx) # bos eos要注意！！！
                    word_list = to_tokens(id_list)
                    word_list=[w for w in word_list]
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence) # 若nbest>1,岂不是同一句要写好几行？

                pred_indices.append(src_indices[idx])
            # break # 生成1batch
    f.close()

    # 读取tgt文本
    test_lines=[]
    with open(conf.data.predict_file.split(',')[1], "r",encoding='utf-8') as fp:
        for line in fp.readlines():
            test_lines.append(line.strip())
    # 按索引写入reference.txt
    ref_lines=[test_lines[idx] for idx in pred_indices]
    with open(conf.data.output_file.split(',')[1], "w",encoding='utf-8') as fr:
        fr.write('\n'.join(ref_lines))
