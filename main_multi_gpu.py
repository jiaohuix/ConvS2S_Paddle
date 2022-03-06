import os
import time
import paddle
from valid import validation
from data import prep_dataset, prep_loader
from utils import same_seeds, save_model, \
    ReduceOnPlateauWithAnnael, ConvS2SMetric, get_logger
from models import build_model
import paddle.distributed as dist
from paddlenlp.transformers import CrossEntropyCriterion, LinearDecayWithWarmup
from config import get_config, get_arguments
from visualdl import LogWriter
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

def train_one_epoch(dataloader,
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    epoch,
                    step_id,
                    metric,
                    logger,
                    logwriter,
                    max_epoch,
                    pad_idx=1,
                    amp=False,
                    log_steps=100,
                    accum_iter=1,
                    scheduler=None):  # for warmup
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: ConvS2S model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        log_steps: int, num of iters to log info
        accum_iter: int, num of iters for accumulating gradients
    Returns:
        train_loss_meter.avg
        train_acc_meter.avg
        train_time
    """
    world_size = paddle.distributed.get_world_size()
    model.train()
    # Train loop
    sentences = 0
    tic_train = time.time()
    for batch_id, input_data in enumerate(dataloader):
        (samples_id, src_tokens, prev_tokens, tgt_tokens) = input_data
        # for multi card training
        if world_size>1:
            if amp is True: # mixed precision training
                # step 1 : skip gradient synchronization by 'no_sync'
                with model.no_sync():
                    with paddle.amp.auto_cast():
                        logits = model(src_tokens=src_tokens, prev_output_tokens=prev_tokens)[0]
                        sum_cost, avg_cost, token_num = criterion(logits, tgt_tokens)
                    scaled = scaler.scale(avg_cost)
                    scaled.backward()
                if ((batch_id + 1) % accum_iter == 0) or (batch_id+1==len(dataloader)):
                    fused_allreduce_gradients(list(model.parameters()), None)
                    scaler.minimize(optimizer, scaled)
                    optimizer.clear_grad()
            else: # full precision training
                with model.no_sync():
                    logits = model(src_tokens=src_tokens, prev_output_tokens=prev_tokens)[0]
                    sum_cost, avg_cost, token_num = criterion(logits, tgt_tokens)
                    avg_cost.backward()
                if ((batch_id + 1) % accum_iter == 0) or (batch_id+1==len(dataloader)):
                    fused_allreduce_gradients(list(model.parameters()), None)
                    optimizer.step()
                    optimizer.clear_grad()
        # for single card training
        else:
            if amp is True:  # mixed precision training
                with paddle.amp.auto_cast():
                    logits = model(src_tokens=src_tokens, prev_output_tokens=prev_tokens)[0]
                    sum_cost, avg_cost, token_num = criterion(logits, tgt_tokens)
                scaled = scaler.scale(avg_cost)
                scaled.backward()
                if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                    scaler.minimize(optimizer, scaled)
                    optimizer.clear_grad()
            else:  # full precision training
                logits = model(src_tokens=src_tokens, prev_output_tokens=prev_tokens)[0]
                sum_cost, avg_cost, token_num = criterion(logits, tgt_tokens)
                avg_cost.backward()
                if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.clear_grad()

        # aggregate metric
        loss, nll_loss, ppl = metric.update(sum_cost, logits, target=tgt_tokens, sample_size=token_num, pad_id=pad_idx)
        sentences += src_tokens.shape[0]

        # log
        if (batch_id + 1) % log_steps == 0:
            avg_bsz = sentences / (batch_id + 1)
            avg_total_steps = len(dataloader.dataset) // avg_bsz // dist.get_world_size()
            loss, nll_loss, ppl = metric.accumulate()

            logger.info(
                f"Train | Epoch: [{epoch}/{max_epoch}] | Step: [{batch_id + 1}/{avg_total_steps}] | Avg bsz:{avg_bsz:.1f} "
                f"Avg loss: {float(loss):.3f} | nll_loss:{float(nll_loss):.3f} | ppl: {float(ppl):.3f} | "
                f"Speed:{log_steps / (time.time() - tic_train):.2f} step/s ")
            tic_train = time.time()

        # if scheduler:scheduler.step()
        if dist.get_rank() == 0:
            logwriter.add_scalar(tag='train/loss', step=step_id, value=loss)
            logwriter.add_scalar(tag='train/ppl', step=step_id, value=ppl)

        step_id += 1

    return step_id


def main_worker(*args):
    # 0.Preparation
    conf = args[0]
    dist.init_parallel_env()
    last_epoch = conf.train.last_epoch
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    logger = get_logger(loggername=f"ConvS2S_{local_rank}", save_path=conf.SAVE)
    logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    seed = conf.seed + local_rank
    same_seeds(seed)

    # 1. Create train and val dataloader
    dataset_train, dataset_val = args[1], args[2]
    # Create training dataloader
    train_loader = None
    if not conf.eval:
        train_loader = prep_loader(conf, dataset_train, 'train', multi_process=True)
        logger.info(
            f'----- Total of train set:{len(train_loader.dataset)} ,train batch: {len(train_loader)} [single gpu]')
    dev_loader = prep_loader(conf, dataset_val, 'dev', multi_process=True)
    logger.info(f'----- Total of valid set:{len(dev_loader.dataset)} ,valid batch: {len(dev_loader)} [single gpu]')
    if local_rank == 0:
        logger.info(f'configs:\n{conf}')

    # 2. Create model
    model = build_model(conf, is_test=False)
    model = paddle.DataParallel(model)
    # 3. Define criterion
    criterion = CrossEntropyCriterion(conf.learning_strategy.label_smooth_eps, pad_idx=conf.model.pad_idx)
    metric = ConvS2SMetric()
    logwriter = None
    best_bleu = 0
    if local_rank == 0:
        logwriter = LogWriter(
            logdir=os.path.join(conf.SAVE, f'vislogs/convs2s_{conf.data.src_lang}{conf.data.tgt_lang}'))

    # 4. Define optimizer and lr_scheduler
    scheduler = ReduceOnPlateauWithAnnael(learning_rate=conf.learning_strategy.learning_rate,
                                          patience=conf.learning_strategy.patience,
                                          force_anneal=conf.learning_strategy.force_anneal,
                                          factor=conf.learning_strategy.lr_shrink,
                                          min_lr=conf.learning_strategy.min_lr)  # reduce the learning rate until it falls below 10−4
    # scheduler=LinearDecayWithWarmup(learning_rate=conf.learning_strategy.learning_rate,
    #                                 warmup=conf.learning_strategy.warmup,
    #                                 last_epoch=conf.train.last_epoch,
    #                                 total_steps=conf.train.max_epoch * conf.train.avg_steps)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=conf.learning_strategy.clip_norm)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        momentum=conf.learning_strategy.momentum,
        weight_decay=float(conf.learning_strategy.weight_decay),  # int object not callable error
        use_nesterov=conf.learning_strategy.use_nesterov,
        grad_clip=clip,
        parameters=model.parameters())

    # 5. Load  resume  optimizer states
    if conf.train.resume:
        model_path = os.path.join(conf.train.resume, 'convs2s.pdparams')
        optim_path = os.path.join(conf.train.resume, 'convs2s.pdopt')
        assert os.path.isfile(model_path) is True, f"File {model_path} does not exist."
        assert os.path.isfile(optim_path) is True, f"File {optim_path} does not exist."
        model_state = paddle.load(model_path)
        opt_state = paddle.load(optim_path)
        if conf.learning_strategy.reset_lr:  # weather to reset lr
            opt_state['LR_Scheduler']['last_lr'] = conf.learning_strategy.learning_rate
        # resume best bleu
        best_bleu = opt_state['LR_Scheduler'].get('best_bleu', 0)
        model.set_dict(model_state)
        optimizer.set_state_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {conf.train.resume},LR={optimizer.get_lr():.5f}----- ")

    # 6. Validation
    if conf.eval:
        logger.info('----- Start Validating')
        val_loss, val_nll_loss, val_ppl, dev_bleu = validation(conf, dev_loader, model, criterion, logger)
        return

    # 6. Start training and validation
    # 定义 GradScaler
    scale_init = conf.train.fp16_init_scale
    growth_interval = conf.train.growth_interval if conf.train.amp_scale_window else 2000
    scaler = paddle.amp.GradScaler(init_loss_scaling=scale_init, incr_every_n_steps=growth_interval)
    global_step_id = conf.train.last_epoch * len(train_loader) + 1
    lowest_val_loss = 0
    num_runs = 0
    for epoch in range(last_epoch + 1, conf.train.max_epoch + 1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.5f}")
        global_step_id = train_one_epoch(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step_id=global_step_id,
            metric=metric,
            logger=logger,
            logwriter=logwriter,
            max_epoch=conf.train.max_epoch,
            pad_idx=conf.model.pad_idx,
            amp=conf.train.amp,
            log_steps=conf.train.log_steps,
            accum_iter=conf.train.accum_iter,
            # scheduler=scheduler
        )
        metric.reset()
        # evaluate model on valid data after one epoch
        val_loss, val_nll_loss, val_ppl, dev_bleu = validation(conf, dev_loader, model, criterion, logger)
        # save best model state
        if (best_bleu < dev_bleu) and (local_rank == 0):
            best_bleu = dev_bleu
            save_dir = os.path.join(conf.SAVE, conf.model.save_model, "model_best")
            save_model(model, optimizer, save_dir=save_dir, best_bleu=best_bleu)
            logger.info(f"Epoch:[{epoch}] | Best Valid Bleu: {best_bleu:.3f} saved to {save_dir}!")

        # visualize valid metrics
        if local_rank == 0:
            logwriter.add_scalar(tag='valid/loss', step=epoch, value=val_loss)
            logwriter.add_scalar(tag='valid/ppl', step=epoch, value=val_ppl)
            logwriter.add_scalar(tag='valid/bleu', step=epoch, value=dev_bleu)

        # adjust learning rate when val ppl stops improving.
        scheduler.step(val_ppl)
        # stop training when lr too small
        cur_lr = round(optimizer.get_lr(), 5)
        min_lr = round(conf.learning_strategy.min_lr, 5)
        if (cur_lr <= min_lr) and (local_rank == 0):
            save_model(model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, "min_lr"))
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
        if (epoch % conf.train.save_epoch == 0) and (local_rank == 0):
            save_model(model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, f"epoch_{epoch}"))

    # save last model
    if (conf.model.save_model) and (local_rank == 0):
        save_model(model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, "epoch_final"))

    if local_rank==0:
        logwriter.close()


def main():
    args = get_arguments()
    conf = get_config(args)
    if not conf.eval:
        dataset_train = prep_dataset(conf, mode='train')
    else:
        dataset_train = None
    dataset_dev = prep_dataset(conf, mode='dev')

    dist.spawn(main_worker, args=(conf, dataset_train, dataset_dev,), nprocs=conf.ngpus)


if __name__ == "__main__":
    main()
