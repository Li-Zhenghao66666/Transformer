#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from config import Config, Logger
from model import Transformer
from utils.vocab import Vocab
from utils.tokenizer import Tokenizer
from utils.util import make_src_mask, make_trg_mask


def translate_sentence(sentence, model, device, src_vocab, trg_vocab, src_tokenizer, max_len=100):
    model.eval()
    tokens = src_tokenizer.tokenizer(sentence)
    tokens = ['<sos>'] + tokens + ['<eos>']
    print("Tokenized input:", tokens)

    tokens = [src_vocab.word2id.get(tok, src_vocab.word2id['<unk>']) for tok in tokens]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = make_src_mask(src_tensor, src_vocab, device)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg = [trg_vocab.word2id['<sos>']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg).unsqueeze(0).to(device)
        trg_mask = make_trg_mask(trg_tensor, trg_vocab, device)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
            output = model.fc(output)
        pred_token = output.argmax(2)[:, -1].item()
        trg.append(pred_token)
        if pred_token == trg_vocab.word2id['<eos>']:
            break

    trg_tokens = [trg_vocab.id2word[idx] for idx in trg]
    return trg_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer 英德翻译推理脚本")

    parser.add_argument('--sent', type=str, required=True, help="输入要翻译的英文句子")
    parser.add_argument('--config', type=str, default='config.json', help="配置文件路径")
    parser.add_argument('--model', type=str, default=None, help="模型权重文件路径（优先级高于配置文件中的resume_path）")
    parser.add_argument('--device', type=str, default=None, help="可选：指定 cpu 或 cuda:0")
    parser.add_argument('--max_len', type=int, default=100, help="翻译输出的最大长度")

    args = parser.parse_args()

    logger = Logger()
    cfg = Config(logger, args)
    cfg.load_config(args.config)

    # 手动覆盖模型路径（如果指定了 --model）
    if args.model is not None:
        cfg.config['resume_path'] = args.model

    src_vocab = Vocab()
    trg_vocab = Vocab()
    src_vocab.load(cfg.config['src_vocab'])
    trg_vocab.load(cfg.config['trg_vocab'])

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Transformer(
        src_vocab_size=src_vocab.vocab_size,
        target_vocab_size=trg_vocab.vocab_size,
        device=device, **cfg.config
    )
    checkpoint = torch.load(cfg.config['resume_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    en_tokenizer = Tokenizer('en_core_web_sm')
    sentence = args.sent
    print("Input sentence:", sentence)

    res = translate_sentence(sentence, model, device, src_vocab, trg_vocab, en_tokenizer, max_len=args.max_len)
    print("Output tokens:", res)
    print("Translation:", " ".join(res[1:-1]))  # 去掉 <sos>/<eos>
