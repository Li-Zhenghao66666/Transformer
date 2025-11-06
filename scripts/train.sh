#!/bin/bash
# ===============================================
# 基础模型训练脚本（Base Transformer）
# ===============================================

# 创建输出目录（若不存在）
mkdir -p ./checkpoint/base

# 运行训练
python train.py \
  --batch_size 32 \
  --h_dim 192 \
  --lr 0.0005 \
  --epochs 10 \
  --enc_dropout 0.2 \
  --dec_dropout 0.2 \
  --l2 0.0001 \
  --enc_n_layers 2 \
  --dec_n_layers 2 \
  --enc_n_heads 6 \
  --dec_n_heads 6 \
  --save_dir ./checkpoint/base \
  --config_file ./checkpoint/base/config.json

echo "✅ 基础模型训练完成，结果已保存到 ./checkpoint/base"
