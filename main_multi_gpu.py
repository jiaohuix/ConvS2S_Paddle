import os
import time
import paddle
import argparse
import logging
import sys
from data import prep_dataset, prep_loader
from tqdm import tqdm
from eval import eval_model
from utils import same_seeds, get_config, calc_ppl, save_model, ReduceOnPlateauWithAnnael
from models import build_model
import paddle.distributed as dist
from paddlenlp.transformers import CrossEntropyCriterion

# python main_multi_gpu_orig.py --config config/en2ro.yaml --last_epoch 60 --resume ckpt_ro/epoch_60

parser = argparse.ArgumentParser(description='ConvS2S', add_help=True)
parser.add_argument('-c', '--config', default='config/base.yaml', type=str, metavar='FILE', help='yaml file path')
parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'pred'])
parser.add_argument('--ngpus', type=int, default=-1)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--last_epoch', type=int, default=None)
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()
conf = get_config(args)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m%d %I:%M:%S %p")
# set logging format
logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(conf.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)


@paddle.no_grad()
def validation(dataloader, model, criterion):
    # Validation
    model.eval()
    total_smooth_loss = 0
    total_nll_loss = 0
    total_tokens = 0
    with paddle.no_grad():
        for input_data in dataloader:
            (src_tokens, tgt_tokens, lbl_tokens) = input_data
            logits = model(src_tokens=src_tokens, prev_output_tokens=tgt_tokens)[0]
            sum_cost, avg_cost, token_num = criterion(logits, lbl_tokens)
            avg_nll_loss, avg_ppl = calc_ppl(logits, lbl_tokens, token_num, conf.model.pad_idx)

            nll_loss = avg_nll_loss * token_num
            dist.all_reduce(sum_cost)
            dist.all_reduce(nll_loss)
            dist.all_reduce(token_num)

            total_smooth_loss += sum_cost
            total_nll_loss += nll_loss
            total_tokens += token_num

        avg_smooth_loss = float(total_smooth_loss / total_tokens)
        avg_nll_loss = min(float(total_nll_loss / total_tokens), 100.)
        avg_ppl = pow(2, avg_nll_loss)

        logger.info(
            f"Eval rank:[{dist.get_rank()}] | Avg loss: {avg_smooth_loss:.3f} | nll_loss:{avg_nll_loss:.3f} | ppl: {avg_ppl:.3f} | ")

    model.train()

    return avg_smooth_loss, avg_nll_loss, avg_ppl


def train_one_epoch(dataloader,
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    epoch,
                    step,
                    debug_steps=100,
                    accum_iter=1):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: ConvS2S model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
        accum_iter: int, num of iters for accumulating gradients
    Returns:
        train_loss_meter.avg
        train_acc_meter.avg
        train_time
    """
    model.train()
    # Train loop
    sentences = 0
    tic_train = time.time()
    for batch_id, input_data in enumerate(dataloader):
        # forward
        (src_tokens, tgt_tokens, lbl_tokens) = input_data
        sentences += src_tokens.shape[0]
        # 创建AMP上下文环境，开启自动混合精度训练
        with paddle.amp.auto_cast(enable=conf.train.auto_cast):
            logits = model(src_tokens=src_tokens, prev_output_tokens=tgt_tokens)[0]
            sum_cost, avg_cost, token_num = criterion(logits, lbl_tokens)

        # 使用 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        scaled = scaler.scale(avg_cost)
        scaled.backward()  # 只反传梯度，不更新参数

        # accumulate grad
        if (batch_id + 1) % accum_iter == 0:
            # 训练模型
            scaler.minimize(optimizer, scaled)
            optimizer.clear_grad()
        # log
        if (batch_id + 1) % debug_steps == 0:
            avg_bsz = sentences / (batch_id + 1)
            avg_total_steps = len(dataloader.dataset) // avg_bsz // dist.get_world_size()
            nll_loss, ppl = calc_ppl(logits, lbl_tokens, token_num, conf.model.pad_idx)

            logger.info(
                f"Train rank:[{dist.get_rank()}] | Epoch: [{epoch}/{conf.train.max_epoch}] | Step: [{batch_id+1}/{avg_total_steps}] | Avg bsz:{avg_bsz:.1f} "
                f"Avg loss: {float(avg_cost):.3f} | nll_loss:{float(nll_loss):.3f} | ppl: {float(ppl):.3f} | "
                f"Speed:{debug_steps / (time.time() - tic_train):.2f} step/s ")
            tic_train = time.time()
        step += 1

    return step


def main_worker(*args):
    # 0.Preparation
    dist.init_parallel_env()
    last_epoch = conf.train.last_epoch
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    seed = conf.train.random_seed + local_rank
    same_seeds(seed)
    # 1. Create train and val dataloader
    dataset_train, dataset_val = args[0], args[1]
    train_loader = prep_loader(conf, dataset_train, 'train', True)
    dev_loader = prep_loader(conf, dataset_val, 'dev', True)
    logger.info(f'Prep | Train num:{len(train_loader.dataset)} | Val num:{len(dev_loader.dataset)}')
    # 2. Create model
    model = build_model(conf, is_test=False)
    model = paddle.DataParallel(model)
    # 3. Define criterion
    criterion = CrossEntropyCriterion(conf.learning_strategy.label_smooth_eps, pad_idx=conf.model.pad_idx)
    # 4. Define optimizer and lr_scheduler
    scheduler = ReduceOnPlateauWithAnnael(learning_rate=conf.learning_strategy.learning_rate,
                                          patience=conf.learning_strategy.patience,
                                          force_anneal=conf.learning_strategy.force_anneal,
                                          factor=conf.learning_strategy.lr_shrink,
                                          min_lr=conf.learning_strategy.min_lr)  # reduce the learning rate until it falls below 10−4
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=conf.learning_strategy.clip_norm)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        momentum=conf.learning_strategy.momentum,
        use_nesterov=conf.learning_strategy.use_nesterov,
        grad_clip=clip,
        parameters=model.parameters())

    # 5. Load  resume  optimizer states
    if conf.model.resume:
        model_path = os.path.join(conf.model.resume, 'convs2s.pdparams')
        optim_path = os.path.join(conf.model.resume, 'convs2s.pdopt')
        assert os.path.isfile(model_path) is True
        assert os.path.isfile(optim_path) is True
        model_state = paddle.load(model_path)
        model.set_dict(model_state)
        opt_state = paddle.load(optim_path)
        optimizer.set_state_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {conf.model.resume}")

    # 6. Validation
    if conf.eval:
        logger.info('----- Start Validating')
        val_loss, val_nll_loss, val_ppl = eval_model(model, dev_loader, criterion)
        return

    # 6. Start training and validation
    # 定义 GradScaler
    scale_init = conf.train.fp16_init_scale
    growth_interval = conf.train.growth_interval if conf.train.amp_scale_window else 2000
    scaler = paddle.amp.GradScaler(init_loss_scaling=scale_init, incr_every_n_steps=growth_interval)
    step = 0
    lowest_val_loss = 0
    num_runs = 0
    for epoch in range(last_epoch + 1, conf.train.max_epoch + 1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        step = train_one_epoch(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=step,
            debug_steps=conf.train.log_step,
            accum_iter=conf.train.accumulate_batchs)

        # evaluate model on valid data after one epoch
        val_loss, val_nll_loss, val_ppl = validation(dev_loader, model, criterion)

        # adjust learning rate when val ppl stops improving.
        scheduler.step(val_ppl)
        cur_lr = round(optimizer.get_lr(), 5)
        min_lr = round(conf.learning_strategy.min_lr, 5)
        if local_rank == 0:
            if cur_lr == min_lr:
                save_model(conf.model, model, optimizer, dir_name=f"min_lr")
                break

        # early stop
        if conf.train.stop_patience > 1:
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                num_runs = 0
            else:
                num_runs += 1
                if num_runs >= conf.train.stop_patience:
                    logger.info(
                        f"early stop since valid performance hasn't improved for last {conf.train.early_stop_num} runs")
                    break

        # save model after several epochs
        if local_rank == 0:
            if epoch % conf.train.save_epoch == 0:
                save_model(conf.model, model, optimizer, dir_name=f"epoch_{epoch}")

    # save last model
    if local_rank == 0:
        if conf.model.save_model:
            save_model(conf.model, model, optimizer, dir_name="epoch_final")


def main():
    dataset_train = prep_dataset(conf, mode='train')  # 由于是单机，所以数据只用加载一次
    dataset_dev = prep_dataset(conf, mode='dev')
    conf.ngpus = len(paddle.static.cuda_places()) if conf.ngpus == -1 else conf.ngpus
    dist.spawn(main_worker, args=(dataset_train, dataset_dev,), nprocs=conf.ngpus)  # 启动多个进程


if __name__ == "__main__":
    main()
