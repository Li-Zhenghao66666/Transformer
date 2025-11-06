#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch

from config import Config, Logger
from model import Transformer
from utils.vocab import Vocab
from utils.tokenizer import Tokenizer
from utils.util import make_src_mask, make_trg_mask

@torch.no_grad()
def translate(sentence, model, device, src_vocab, trg_vocab, src_tokenizer, max_len=100):
    """贪心解码的 EN->DE 翻译函数"""
    model.eval()

    # 1) 分词并加特殊符号
    toks = src_tokenizer.tokenizer(sentence)
    toks = ['<sos>'] + toks + ['<eos>']
    src_ids = [src_vocab.word2id.get(t, src_vocab.word2id['<unk>']) for t in toks]

    # 2) 编码
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]
    src_mask = make_src_mask(src, src_vocab, device)
    enc_out = model.encoder(src, src_mask)

    # 3) 解码（贪心）
    trg_ids = [trg_vocab.word2id['<sos>']]
    for _ in range(max_len):
        trg = torch.tensor(trg_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        trg_mask = make_trg_mask(trg, trg_vocab, device)
        dec_out = model.decoder(trg, enc_out, src_mask, trg_mask)  # [1, T, H]
        logits = model.fc(dec_out)                                  # [1, T, V]
        next_id = int(logits.argmax(dim=-1)[0, -1])
        trg_ids.append(next_id)
        if next_id == trg_vocab.word2id['<eos>']:
            break

    # 4) 去掉 <sos>/<eos> 并还原词
    toks_out = [trg_vocab.id2word[i] for i in trg_ids]
    text_out = " ".join(toks_out[1:-1]) if len(toks_out) >= 2 else ""
    return toks_out, text_out


def main():
    parser = argparse.ArgumentParser(description="Test English→German translation")
    parser.add_argument("--sent", required=True, help="输入英文句子")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--model", default=None, help="模型权重路径（可覆盖配置中的 resume_path）")
    parser.add_argument("--device", default=None, help="可选：cpu 或 cuda:0 等")
    parser.add_argument("--max_len", type=int, default=100, help="最大生成长度")
    args = parser.parse_args()

    # === 读取配置 ===
    logger = Logger()
    cfg = Config(logger, args)
    cfg.load_config(args.config)
    if args.model is not None:
        cfg.config["resume_path"] = args.model

    # === 词表（配置里：src=en，trg=de）===
    src_vocab = Vocab(); src_vocab.load(cfg.config["src_vocab"])  # EN
    trg_vocab = Vocab(); trg_vocab.load(cfg.config["trg_vocab"])  # DE

    # === 设备 ===
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 加载 ckpt ===
    ckpt = torch.load(cfg.config["resume_path"], map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    # === 从 ckpt 推断 encoder/decoder 需要的词表大小 ===
    if "encoder.word_embeddings.weight" not in state:
        raise KeyError("Checkpoint缺少键：encoder.word_embeddings.weight")
    enc_vs_ckpt = state["encoder.word_embeddings.weight"].shape[0]
    if "decoder.word_embeddings.weight" in state:
        dec_vs_ckpt = state["decoder.word_embeddings.weight"].shape[0]
    else:
        # 有些实现没有独立的 decoder embedding 名称，就以 fc.weight 的输出维度为目标词表大小
        dec_vs_ckpt = state["fc.weight"].shape[0]

    # === 当前词表大小（来自 en_vocab.pkl / de_vocab.pkl）===
    src_vs = src_vocab.vocab_size   # EN
    trg_vs = trg_vocab.vocab_size   # DE

    print(f"[Info] src_vocab={src_vs}, trg_vocab={trg_vs}, "
          f"ckpt(enc)={enc_vs_ckpt}, ckpt(dec)={dec_vs_ckpt}")

    # === 如果发现正好反了，就交换 src/trg 词表（仅在内存中交换，不改文件名）===
    if src_vs == dec_vs_ckpt and trg_vs == enc_vs_ckpt:
        print("[Fix] Detected swapped vocab sizes vs checkpoint. Swapping src/trg vocabs for model build.")
        src_vocab, trg_vocab = trg_vocab, src_vocab
        src_vs, trg_vs = trg_vs, src_vs

    # === 构建模型（使用与词表一致的大小）===
    model = Transformer(
        src_vocab_size=src_vs,
        target_vocab_size=trg_vs,
        device=device, **cfg.config
    ).to(device)

    # === 严格加载权重，确保维度完全匹配 ===
    model.load_state_dict(state, strict=True)

    # === 源语言分词器：英文 ===
    en_tok = Tokenizer("en_core_web_sm")

    # === 翻译输出 ===
    print("Input:", args.sent)
    toks, text = translate(args.sent, model, device, src_vocab, trg_vocab, en_tok, max_len=args.max_len)
    print("Tokens:", toks)
    print("Translation (EN→DE):", text)


if __name__ == "__main__":
    main()
