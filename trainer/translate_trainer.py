import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .trainer import Trainer

# --------- Mask 工具（与 vocab 配合使用）---------
def make_src_mask(src, src_vocab, device):
    pad_id = src_vocab.word2id['<pad>']
    mask = (src != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
    return mask.to(device)

def make_trg_mask(trg_in, trg_vocab, device):
    pad_id = trg_vocab.word2id['<pad>']
    B, T = trg_in.size(0), trg_in.size(1)
    trg_pad_mask = (trg_in != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
    subsequent = torch.triu(torch.ones((1,1,T,T), device=trg_in.device), diagonal=1).bool()  # [1,1,T,T]
    # 允许的位置为 True：既不是未来位，也不是 pad
    mask = trg_pad_mask & (~subsequent)
    return mask.to(device)

# --------- 一个简洁的 BLEU（1-4gram + BP）---------
def _simple_bleu(pred_tokens, ref_tokens, max_n=4):
    if len(pred_tokens) == 0:
        return 0.0
    weights = [1.0 / max_n] * max_n
    precisions = []
    from collections import Counter

    def ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)] if len(seq) >= n else []

    for n in range(1, max_n + 1):
        p_ngrams = Counter(ngrams(pred_tokens, n))
        r_ngrams = Counter(ngrams(ref_tokens, n))
        overlap = sum((p_ngrams & r_ngrams).values())
        total = max(1, sum(p_ngrams.values()))
        precisions.append(overlap / total)

    if min(precisions) == 0:
        geo = 0.0
    else:
        geo = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))

    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(1, pred_len))
    return bp * geo


class TranslateTrainer(Trainer):
    def _train_epoch(self, epoch):
        self.model.train()
        device = next(self.model.parameters()).device

        total_loss = 0.0

        progress = tqdm(
            enumerate(self.data_loader),
            total=len(self.data_loader),
            ncols=100,
            desc=f"Train  Epoch {epoch}/{self.epochs}",
            leave=False
        )

        for idx, (src, trg) in progress:
            src = src.to(device)
            trg = trg.to(device)

            src_mask = make_src_mask(src, self.data_loader.src_vocab, device)
            trg_in = trg[:, :-1]
            trg_out = trg[:, 1:]
            trg_mask = make_trg_mask(trg_in, self.data_loader.trg_vocab, device)

            self.optimizer.zero_grad()
            output = self.model(src, trg_in, src_mask, trg_mask)  # [B, T-1, V]
            output_dim = output.size(-1)
            loss = self.criterion(
                output.contiguous().view(-1, output_dim),
                trg_out.contiguous().view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            mean_loss = total_loss / (idx + 1)
            progress.set_postfix(loss=f"{mean_loss:.4f}")

        progress.close()
        # 注意：这里 total_loss 是“各 batch 的平均 loss 再求均值”，按 batch 计平均即可
        mean_train_loss = total_loss / len(self.data_loader)

        val_loss, bleu, acc, ppl = (None, None, None, None)
        if self.do_validation:
            self.logger.debug("start validation")
            val_loss, bleu, acc, ppl = self._valid_epoch()

        self.logger.info(
            f"Train Epoch: {epoch} | train_loss: {mean_train_loss:.6f} | "
            f"val_loss: {float(val_loss):.6f} | BLEU: {float(bleu):.4f} | "
            f"Acc: {float(acc):.4f} | PPL: {float(ppl):.4f}"
        )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 返回给基类用于 history 记录 & 选择 best
        return float(val_loss), {
            'train_loss': float(mean_train_loss),
            'val_loss': float(val_loss),
            'bleu': float(bleu),
            'acc': float(acc),
            'ppl': float(ppl)
        }

    def _valid_epoch(self):
        self.model.eval()
        device = next(self.model.parameters()).device

        total_loss = 0.0          # 累加“每样本”的 loss（见下面 * batch_size）
        total_examples = 0        # 验证集样本总数（考虑最后一个不满 batch 的情况）
        total_acc = 0
        total_tokens = 0
        total_bleu = 0.0
        sent_count = 0

        pad_id = self.valid_data_loader.trg_vocab.word2id['<pad>']
        eos_id = self.valid_data_loader.trg_vocab.word2id['<eos>']

        progress = tqdm(
            enumerate(self.valid_data_loader),
            total=len(self.valid_data_loader),
            ncols=100,
            desc="Valid ",
            leave=False
        )

        with torch.no_grad():
            for idx, (src, trg) in progress:
                src = src.to(device)
                trg = trg.to(device)

                src_mask = make_src_mask(src, self.valid_data_loader.src_vocab, device)
                trg_in = trg[:, :-1]
                trg_out = trg[:, 1:]
                trg_mask = make_trg_mask(trg_in, self.data_loader.trg_vocab, device)

                output = self.model(src, trg_in, src_mask, trg_mask)  # [B, T-1, V]
                output = F.log_softmax(output, dim=-1)
                output_dim = output.size(-1)

                loss = self.criterion(
                    output.contiguous().view(-1, output_dim),
                    trg_out.contiguous().view(-1)
                )
                batch_size = src.size(0)

                # 注意：criterion 默认返回“平均到样本”的 loss；乘以 batch_size 后，变为“该 batch 的总 loss”
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                # token 级准确率（忽略 PAD）
                pred_ids = output.argmax(dim=-1)  # [B, T-1]
                mask = (trg_out != pad_id)
                total_acc += ((pred_ids == trg_out) & mask).sum().item()
                total_tokens += mask.sum().item()

                # 句级 BLEU（遇到 <eos> 截断，忽略 PAD）
                for i in range(batch_size):
                    pred_seq = []
                    for tid in pred_ids[i].tolist():
                        if tid == eos_id:
                            break
                        if tid != pad_id:
                            pred_seq.append(self.valid_data_loader.trg_vocab.id2word[tid])
                    ref_seq = []
                    for tid in trg_out[i].tolist():
                        if tid == eos_id:
                            break
                        if tid != pad_id:
                            ref_seq.append(self.valid_data_loader.trg_vocab.id2word[tid])
                    total_bleu += _simple_bleu(pred_seq, ref_seq)
                    sent_count += 1

                # 进度条显示当前以“样本”为单位的 val_loss 均值
                avg_loss_running = total_loss / max(1, total_examples)
                progress.set_postfix(val_loss=f"{avg_loss_running:.4f}")

        progress.close()

        # ✅ 按“验证集样本总数”取平均，避免被 batch_size 放大
        avg_loss = total_loss / max(1, total_examples)
        avg_acc = (total_acc / max(1, total_tokens)) if total_tokens > 0 else 0.0
        avg_bleu = total_bleu / max(1, sent_count)
        ppl = float(np.exp(avg_loss)) if avg_loss < 20 else float('inf')
        return float(avg_loss), float(avg_bleu), float(avg_acc), float(ppl)

    def _plot_and_dump(self):
        os.makedirs(self.save_dir, exist_ok=True)
        hist = self.history

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # 1) Loss (train / val)
        ax = axes[0, 0]
        ax.plot(hist['train_loss'], label='train_loss')
        ax.plot(hist['val_loss'], label='val_loss')
        ax.set_title('Loss'); ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.legend()

        # 2) BLEU
        ax = axes[0, 1]
        ax.plot(hist['bleu'])
        ax.set_title('BLEU'); ax.set_xlabel('epoch'); ax.set_ylabel('BLEU')

        # 3) Accuracy
        ax = axes[1, 0]
        ax.plot(hist['acc'])
        ax.set_title('Accuracy'); ax.set_xlabel('epoch'); ax.set_ylabel('acc')

        # 4) Perplexity
        ax = axes[1, 1]
        ax.plot(hist['ppl'])
        ax.set_title('Perplexity'); ax.set_xlabel('epoch'); ax.set_ylabel('ppl')

        fig.tight_layout()
        out_png = os.path.join(self.save_dir, 'training_metrics.png')
        fig.savefig(out_png)

        out_pkl = os.path.join(self.save_dir, 'metrics_data.pkl')
        with open(out_pkl, 'wb') as f:
            pickle.dump(hist, f)
