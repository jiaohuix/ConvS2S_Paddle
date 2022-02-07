import os
import time
import paddle
from valid import validation
from utils import save_model, ReduceOnPlateauWithAnnael, ConvS2SMetric,same_seeds
from paddlenlp.transformers import CrossEntropyCriterion
from reprod_log import ReprodLogger, ReprodDiffHelper
reprod = ReprodLogger()
diff_helper = ReprodDiffHelper()


def train_model(conf, model, train_loader, dev_loader,logger):
    model_args, strategy_args, train_args = conf.model, conf.learning_strategy, conf.train
    if train_args.use_gpu:
        place = "gpu"
    else:
        place = "cpu"
    paddle.set_device(place)
    # Set seed for CE
    same_seeds(conf.seed)

    # Define loss
    # from loss import LabelSmoothedCrossEntropyCriterion
    # criterion = LabelSmoothedCrossEntropyCriterion(strategy_args.label_smooth_eps, pad_idx=model_args.pad_idx)
    criterion = CrossEntropyCriterion(strategy_args.label_smooth_eps, pad_idx=model_args.pad_idx)
    metric = ConvS2SMetric()

    # Define optimizer
    scheduler = ReduceOnPlateauWithAnnael(learning_rate=strategy_args.learning_rate,
                                          patience=strategy_args.patience,
                                          force_anneal=strategy_args.force_anneal,
                                          factor=strategy_args.lr_shrink,
                                          min_lr=strategy_args.min_lr)  # reduce the learning rate until it falls below 10−4
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=strategy_args.clip_norm)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        momentum=strategy_args.momentum,
        use_nesterov=strategy_args.use_nesterov,
        grad_clip=clip,
        parameters=model.parameters())

    #  Load  resume  optimizer states
    if conf.train.resume:
        model_path = os.path.join(conf.train.resume, 'convs2s.pdparams')
        optim_path = os.path.join(conf.train.resume, 'convs2s.pdopt')
        assert os.path.isfile(model_path) is True
        assert os.path.isfile(optim_path) is True
        model_state = paddle.load(model_path)
        opt_state = paddle.load(optim_path)
        if conf.learning_strategy.reset_lr: # 是否重置lr
            opt_state['LR_Scheduler']['last_lr']=conf.learning_strategy.learning_rate
        model.set_dict(model_state)
        optimizer.set_state_dict(opt_state)
        logger.info(f"----- Resume Training: Load model and optmizer states from {conf.train.resume},LR={optimizer.get_lr():.5f}----- ")

    # 定义 GradScaler
    scale_init = conf.train.fp16_init_scale
    growth_interval = conf.train.growth_interval if conf.train.amp_scale_window else 2000
    scaler = paddle.amp.GradScaler(init_loss_scaling=scale_init, incr_every_n_steps=growth_interval)
    step_idx = 0
    # early stop
    num_runs = 0
    lowest_val_loss = 1e9

    # Train loop
    tic_train = time.time()
    for epoch in range(conf.train.last_epoch+1, conf.train.max_epoch + 1):
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        sentences, token_num = 0, 0
        for batch_id, input_data in enumerate(train_loader, start=1):
            # forward
            (samples_id, src_tokens, tgt_tokens, lbl_tokens) = input_data
            # print(src_tokens)
            # if batch_id==save_step:
            #     reprod.add('samples_id',samples_id.numpy())
            #     reprod.add('src_tokens',src_tokens.numpy())
            #     reprod.add('prev_tokens',tgt_tokens.numpy())
            #     reprod.add('tgt_tokens',lbl_tokens.squeeze().numpy())
            sentences += src_tokens.shape[0]
            # auto mixing precision training
            with paddle.amp.auto_cast(enable=conf.train.auto_cast):
                logits = model(src_tokens=src_tokens, prev_output_tokens=tgt_tokens)[0]
                # if batch_id == save_step:
                #     reprod.add('logits', logits.numpy()) # 是不是因为计算了loss后logits改变了？
                sum_cost, avg_cost, token_num_i = criterion(logits, lbl_tokens)
                token_num += token_num_i
                # print(f'sum cost:{float(sum_cost)} | avg:{float(avg_cost)} | num :{int(token_num_i)}')
                # if batch_id == save_step:
                #     reprod.add('loss', np.array(float(sum_cost)))
                #     reprod.save('paddle_forward.npy')

            # 使用 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
            scaled = scaler.scale(avg_cost)
            scaled.backward()  # 只反传梯度，不更新参数

            # accumulate grad
            if batch_id % train_args.accum_iter == 0:
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the fly, so inspecting model.parameters()
                # or optimizer.params may still show the original, unscaled gradients.)
                # numer = ParallelEnv().nranks
                # c = numer / (token_num or 1.0)
                # optimizer._rescale_grad = c # 什么时候multiply梯度，有用？

                # update params
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()

                # reset accumulated token num
                token_num=0

            # aggregate metric
            loss, nll_loss, ppl = metric.update(sum_cost, logits, target=lbl_tokens, sample_size=token_num_i,
                                                pad_id=model_args.pad_idx)

            # log
            if batch_id % train_args.log_steps == 0:
                avg_bsz = sentences / (batch_id + 1)
                # avg_total_steps = len(train_loader.dataset) // avg_bsz
                loss, nll_loss, ppl = metric.accumulate()  # 返回累积batch的平均指标

                logger.info(
                    f"Train | Epoch: [{epoch}/{train_args.max_epoch}] | Step: [{batch_id}/{len(train_loader)}] | Avg bsz:{avg_bsz:.1f} "
                    f"Avg loss: {float(loss):.3f} | nll_loss:{float(nll_loss):.3f} | ppl: {float(ppl):.3f} | "
                    f"Speed:{train_args.log_steps / (time.time() - tic_train):.2f} step/s ")
                tic_train = time.time()

            step_idx += 1

        metric.reset()  # 重置指标

        # evaluate model on valid data after one epoch
        val_loss, val_nll_loss, val_ppl,dev_bleu = validation(conf, dev_loader, model, criterion, logger)
        # adjust learning rate when val ppl stops improving.
        scheduler.step(val_ppl)
        cur_lr = round(optimizer.get_lr(), 5)
        min_lr = round(strategy_args.min_lr, 5)
        logger.info(f'Epoch:[{epoch}/{train_args.max_epoch}]: Current learning rate is {cur_lr}')
        if cur_lr == min_lr:
            save_model(model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, "min_lr"))
            break

        # early stop
        if train_args.stop_patience > 1:
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                num_runs = 0
            else:
                num_runs += 1
                if num_runs >= train_args.stop_patience:
                    logger.info(
                        f"early stop since valid performance hasn't improved for last {train_args.early_stop_num} runs")
                    break

        # save model after several epochs
        if epoch % train_args.save_epoch == 0:
            save_model(model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, f"epoch_{epoch}"))

    # save last model
    if model_args.save_model:
        save_model(conf.model, model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, "epoch_final"))
