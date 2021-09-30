import paddle
import numpy as np
from utils import logger, calc_ppl
from tqdm import tqdm


@paddle.no_grad()
def eval_model(model, dev_loader, criterion):
    # Validation
    model.eval()
    total_smooth_loss = 0
    total_nll_loss = 0
    total_token_num = 0
    with paddle.no_grad():
        for input_data in tqdm(dev_loader):
            (src_tokens, tgt_tokens, lbl_tokens) = input_data
            logits = model(src_tokens=src_tokens, prev_output_tokens=tgt_tokens)[0]
            sum_cost, avg_cost, token_num = criterion(logits, lbl_tokens)
            nll_loss, ppl = calc_ppl(logits, lbl_tokens, token_num, model.encoder.pad_id)
            total_smooth_loss += sum_cost
            total_nll_loss += nll_loss*token_num
            total_token_num += token_num

        avg_smooth_loss = float(total_smooth_loss / total_token_num)
        avg_nll_loss = min(float(total_nll_loss / total_token_num), 100.)
        avg_ppl = pow(2, avg_nll_loss)
        logger.info(f"Eval | Avg loss: {avg_smooth_loss:.3f} | nll_loss:{avg_nll_loss:.3f} | ppl: {avg_ppl:.3f} | ")
    model.train()

    return avg_smooth_loss, avg_nll_loss, avg_ppl
from paddle import DataParallel