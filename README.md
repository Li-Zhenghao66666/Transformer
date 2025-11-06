# 大模型基础与应用期中作业

手动实现Transformer: seq2seq，机器翻译，英文<-->德文

Github链接：https://github.com/Li-Zhenghao66666/MidtermAssignment

### 安装

在开始之前使用下面的设置命令配置环境：

```python
#创建并激活虚拟环境
conda create -n translate python=3.8.20 -y
conda activate translate

#安装Pytorch（与CUDA适配）
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

#克隆项目并安装项目库
git clone https://github.com/Li-Zhenghao66666/MidtermAssignment.git
cd xxxxxxxxxxx
pip install -r requirements .txt
```

### 文件结构

```
.
├──────  checkpoint/              # 模型保存目录（自动生成）
├──────   config/                  # 配置文件目录
├──────   data/                    # 数据处理模块
├──────   dataset/                 # 训练与验证数据集
├──────   logs/                    # 日志输出目录
├──────   model/                   # 模型定义
├──────   scripts/                 # 训练与测试脚本
├──────   trainer/                 # 模型训练器与逻辑控制
├──────   utils/                   # 工具函数（优化器、词表、mask等）         
├── config.json              # 模型配置文件
├── README.md                # 项目说明文档
├── requirements.txt         # 依赖库列表
├── test_de_en.py            # 德译英翻译测试文件
├── test_en_de.py            # 英译德翻译测试文件
├── test.py                  # 通用测试脚本
├── command.txt              # 命令参数记录
└── train.py                 # 模型训练入口
```



### 数据预处理

首先，本实验使用的是IWSLT2017 (EN↔DE)数据集。 [下载数据集](https://github.com/puttisandev/iwslt2017)。

然后，对数据集进行划分，`segment.py` 可自动按比例拆分数据：

- 默认划分比例：训练集（80%）、验证集（10%）、测试集（10%）。

最后，使用Vocab类运行，生成字典和以'\t'为分隔符的分词数据集。

输出文件说明：`train.de.token`训练集德语分词结果，`train.en.token`训练集英语分词结果；`val.de.token`验证集德语分词结果，`val.en.token`验证集英语分词结果；`test.de.token`测试集德语分词结果，`test.en.token`测试集英语分词结果。

注：在该项目中已经分割好数据并已构建词表，可直接使用（放在dataset中）。

```python
from utils import Vocab

de = 'de_core_web_md'
en = 'en_core_web_md'
de_train = 'dataset/train.de'
en_train = 'dataset/train.en'

de_valid = 'dataset/valid.de'
en_valid = 'dataset/valid.en'

de_vocab = Vocab()
en_vocab = Vocab()

de_vocab.create(zh_train, de)
de_vocab.create(zh_valid, de)

en_vocab.create(en_train, en)
en_vocab.create(en_valid, en)

de_vocab.save('dataset/de_vocab.pkl')
en_vocab.save('dataset/en_vocab.pkl')
```



### 脚本执行

```
#脚本执行训练(Linux)：
bash scripts/train.sh
#脚本执行测试(Linux)：
bash scripts/test.sh

#脚本执行训练(win)：
scripts\train_win.bat
#脚本执行测试(win)：
scripts\test_win.bat
```



### 训练模型（baseline）

```python 
#Baseline
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 6 --dec_n_heads 6 --save_dir ./checkpoint/baseline

```

### 消融实验（seed默认都为1234）

```python
#dropout消融（0.05/0.1/0.2/0.3）     dropout=0.2即baseline
#drop0.05:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.05 --dec_dropout 0.05 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 6 --dec_n_heads 6 --save_dir ./checkpoint/drop0.05

#drop0.1:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.1 --dec_dropout 0.1 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 6 --dec_n_heads 6 --save_dir ./checkpoint/drop0.1

#drop0.3:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.3 --dec_dropout 0.3 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 6 --dec_n_heads 6 --save_dir ./checkpoint/drop0.3


#h_dim隐藏层维度消融（120/192/240）          h_dim=192即baseline
#h_dim120:
python train.py --batch_size 32 --h_dim 120 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 6 --dec_n_heads 6 --save_dir ./checkpoint/h_dim120

#h_dim240:
python train.py --batch_size 32 --h_dim 240 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 6 --dec_n_heads 6 --save_dir ./checkpoint/h_dim240


#head注意力头数消融（0/2/4/6/8）        head=6即baseline
#head1:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 1 --dec_n_heads 1 --save_dir ./checkpoint/head1

#head2:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 2 --dec_n_heads 2 --save_dir ./checkpoint/head2

#head4:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads 4 --dec_n_heads 4 --save_dir ./checkpoint/head4

#head8:
python train.py --batch_size 32 --h_dim 192 --lr 0.0005 --epochs 10 --enc_dropout 0.2 --dec_dropout 0.2 --l2 0.0001 --enc_n_layers 2 --dec_n_layers 2 --enc_n_heads  --dec_n_heads 8 --save_dir ./checkpoint/head8
```



### 实验结果对比命令

```python
#命令示例
python compare.py head2/head2.pkl head4/head4.pkl head6/head6.pkl --outdir plot --title "Head comparison"
```



### 测试Demo

**德译英命令示例：**

```python
#德语意思：很高兴见到你
python test_de_en.py --sent "Freut mich, dich zu sehen!" --config checkpoint/base50epoch/config.json --model checkpoint/base50epoch/model_best.pt
```

![](C:\Users\86158\Desktop\test1.png)

```python
#德语意思：谢谢你的帮助
python test_de_en.py --sent "Danke für deine Hilfe!" --config checkpoint/base50epoch/config.json --model checkpoint/base50epoch/model_best.pt
```

![](C:\Users\86158\Desktop\test2.png)

##### 英译德命令示例：

```
python test_en_de.py --sent "This is a simple example." --model checkpoint/en2de/model_best.pt
```

![image-20251106193937324](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20251106193937324.png)

![image-20251106194510969](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20251106194510969.png)

```
python test_en_de.py --sent "nice to meet you" --model checkpoint/en2de/model_best.pt
```

![image-20251106194322495](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20251106194322495.png)

![image-20251106194427711](C:\Users\86158\AppData\Roaming\Typora\typora-user-images\image-20251106194427711.png)
