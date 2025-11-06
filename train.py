import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

# 保障能从当前目录作为包导入
sys.path.append(os.path.dirname(__file__))

from config import Config, Logger
from utils.util import get_optimizer, initialize_weights
from utils.vocab import Vocab
from model import Transformer
from data import Zh2EnDataLoader

# 关键：你的 translate_trainer 在 trainer/ 包里
try:
    from trainer.translate_trainer import TranslateTrainer
except ModuleNotFoundError:
    # 兜底：有的版本类在 trainer/trainer.py 里
    from trainer.trainer import TranslateTrainer


parser = argparse.ArgumentParser()

# dataset parameter
parser.add_argument('--src_train_data', type=str, default='dataset/train.de.token')
parser.add_argument('--trg_train_data', type=str, default='dataset/train.en.token')
parser.add_argument('--src_valid_data', type=str, default='dataset/val.de.token')
parser.add_argument('--trg_valid_data', type=str, default='dataset/val.en.token')
parser.add_argument('--src_vocab', type=str, default='dataset/de_vocab.pkl')
parser.add_argument('--trg_vocab', type=str, default='dataset/en_vocab.pkl')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32)

# model parameter
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--enc_n_layers', type=int, default=3)
parser.add_argument('--dec_n_layers', type=int, default=3)
parser.add_argument('--enc_n_heads', type=int, default=8)
parser.add_argument('--dec_n_heads', type=int, default=8)
parser.add_argument('--enc_dropout', type=float, default=0.1)
parser.add_argument('--dec_dropout', type=float, default=0.1)
parser.add_argument('--enc_pf_dim', type=int, default=512)
parser.add_argument('--dec_pf_dim', type=int, default=512)

# Loss & Optimizer
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--l2', type=float, default=0.0)

# train parameter
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--save_epochs', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_path', type=str, default='./saved_models/model_best.pt')
parser.add_argument('--log_step', type=int, default=20)

# other
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--config_file', type=str, default='./config.json')
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
logger = Logger()

cfg = Config(logger=logger, args=args)
cfg.print_config()
cfg.save_config(cfg.config['config_file'])

# reproducibility
torch.manual_seed(cfg.config['seed'])
if cfg.config['cuda']:
    torch.cuda.manual_seed(cfg.config['seed'])
torch.backends.cudnn.enabled = False
np.random.seed(cfg.config['seed'])

# ensure save_dir
os.makedirs(cfg.config['save_dir'], exist_ok=True)

# vocab
src_vocab = Vocab(); src_vocab.load(cfg.config['src_vocab'])
trg_vocab = Vocab(); trg_vocab.load(cfg.config['trg_vocab'])

# data_loader
train_data_loader = Zh2EnDataLoader(
    cfg.config['src_train_data'], cfg.config['trg_train_data'],
    src_vocab, trg_vocab, cfg.config['batch_size'], cfg.config['shuffle'], logger
)
valid_data_loader = Zh2EnDataLoader(
    cfg.config['src_valid_data'], cfg.config['trg_valid_data'],
    src_vocab, trg_vocab, cfg.config['batch_size'], cfg.config['shuffle'], logger
)

# model
device = 'cuda:0' if cfg.config['cuda'] else 'cpu'
model = Transformer(
    src_vocab_size=src_vocab.vocab_size,
    target_vocab_size=trg_vocab.vocab_size,
    device=device, **cfg.config
).to(device)
logger.info(model)

# optimizer & loss
params = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], params, lr=cfg.config['lr'], l2=cfg.config['l2'])
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.word2id['<pad>'])

# trainer
model.apply(initialize_weights)
trainer = TranslateTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    cfg=cfg.config,
    logger=logger,
    data_loader=train_data_loader,
    valid_data_loader=valid_data_loader,
    lr_scheduler=None
)
trainer.train()
