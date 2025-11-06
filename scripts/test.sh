#!/bin/bash
# ===============================================
# 德译英模型推理脚本 (Test: German → English)
# ===============================================

# 指定输入句子（可改为任意德语句）
INPUT_SENTENCE="Danke für deine Hilfe!"

# 指定配置与模型路径（需与训练阶段一致）
CONFIG_PATH="checkpoint/base50epoch/config.json"
MODEL_PATH="checkpoint/base50epoch/model_best.pt"

# 选择设备
DEVICE="cuda:0"  # 如无GPU可改为 cpu

# 执行推理
python test_de_en.py \
  --sent "$INPUT_SENTENCE" \
  --config "$CONFIG_PATH" \
  --model "$MODEL_PATH" \
  --device "$DEVICE"

echo "✅ 翻译测试完成！"
