# 小型DeepSeek-V3预训练

### 项目简介
本项目是针对小型DeepSeek-V3模型的预训练实现，包含完整的数据处理、模型训练流程。适合在有限资源下进行大语言模型的预训练实验。

### 主要特性
- **目标**：从数据集拉取开始简要跑通整个预训练的流程
- **技术实现**：DeepSpeed Zero-1数据并行训练、DeepSeekMOE、DeepSeekMTP、MLA、YARN（改进的旋转位置编码）、张量并行(列并行和行并行线性层)等
- **适用场景**：在有限资源下进行大语言模型的预训练实验
- **完整流程**：涵盖数据下载、预处理、模型训练
## 1.目录结构

**请仔细阅读这个目录，有一些子的空文件夹需要你自行创建**

```
mini-deepseekv3/
├── requirements_pretraining.txt              # 预训练环境、数据预处理环境依赖
├── requirements_data_step2.txt               # 数据清洗Step2中的Datajucier环境依赖
├── pretrain.py                               # 预训练主代码
├── test_model.py                             # 测试模型能否跑通的脚本
├── logger_utils.py                           # 日志工具
│
├── model/                                    # 模型文件目录
│   ├── config.py                            # 不同参数大小的模型配置参考
│   ├── convert.py                           # DeepSeek官方模型转换代码
│   ├── deepseekv3_mtp_model.py              # DeepSeek-V3模型（含MTP）
│   ├── fp8_cast_bf16.py                     # DeepSeek官方FP8转BF16代码
│   └── kernel.py                            # DeepSeek官方自定义算子代码
│
└── data/                                    # 数据处理目录
    ├── raw_data/                            # 原始数据存储目录 [注意这些子文件夹(空文件夹)需要你自己创建！！！]
    │   ├── ape/                             # 存储APE210K原始数据的文件夹
    │   ├── openr1math/                      # 存储OpenR1原始数据的文件夹
    │   ├── skypile/                         # 存储Skypile原始数据的文件夹
    │   ├── slimpajama/                      # 存储SlimPajama原始数据的文件夹
    │   ├── starcoder/                       # 存储StarCoder原始数据的文件夹
    │   └── hfd_revised.sh                   # 数据下载脚本
    ├── step1/                               # 数据处理第一步：JSONL化
    │   ├── data_step1/                      # 数据处理第一步输出的文件夹
    │   └── step1_pretrain_basic_dataprocess.py  # 转化成JSONL的脚本
    ├── step2/                               # 数据处理第二步：数据清洗
    │   ├── data_step1_split/                # JSONL分割成多个小的JSONL后的数据存储文件夹
    │   ├── data_step2/                      # 数据处理第二步输出的文件夹 [注意这些子文件夹(空文件夹)需要你自己创建！！！]
    │   │    ├──ape/                         # 存储APE210K清洗后数据的文件夹
    │   │    ├──openr1math/                  # 存储OpenR1清洗后数据的文件夹
    │   │    ├──skypile/                     # 存储Skypile清洗后数据的文件夹
    │   │    ├──slimpajama/                  # 存储SlimPajama清洗后数据的文件夹
    │   │    └──starcoder/                   # 存储StarCoder清洗后数据的文件夹
    │   ├── split_jsonl.py                   # JSONL分割成多个小的JSONL的脚本
    │   ├── minidsv3_code_starcoder.yaml     # StarCoder配置
    │   ├── minidsv3_text_ape_.yaml          # APE配置
    │   ├── minidsv3_text_openr1.yaml        # OpenR1配置
    │   ├── minidsv3_text_skypile.yaml       # Skypile配置
    │   ├── minidsv3_text_slimpajama.yaml    # SlimPajama配置
    │   ├── run_step2_ape.sh                 # APE清洗脚本
    │   ├── run_step2_openr1.sh              # OpenR1清洗脚本
    │   ├── run_step2_skypile.sh             # Skypile清洗脚本
    │   ├── run_step2_slimpajama.sh          # SlimPajama清洗脚本
    │   └── run_step2_starcoder.sh           # StarCoder清洗脚本
    ├── step3/                               # 数据处理第三步：Tokenizer
    │   ├── step3_prepare_data_for_pretrain.py # 数据转化成token并打包bin的脚本
    │   ├── data_step3/                      # 数据处理第二步输出的文件夹 [注意这个空文件夹需要你自己创建！！！]
    │   ├── tokenizer.json                   # DeepSeek官方Tokenizer模型
    │   ├── tokenizer_config.json            # DeepSeek官方Tokenizer配置
    │   └── lit_gpt/                         # LitGPT库(用于打包数据成bin文件)
    └── step4/                               # 数据处理第四步：Dataset封装
        └── step4_dataset.py
```
## 2.环境要求
### 2.1 会创建两个环境，如下：
| 序号  | 环境名称 | 主要用途 | 依赖文件 |
| --- | --- | --- | --- |
| 1   | `pretraining` | 预训练的主环境 | `requirements_pretraining.txt` |
| 2   | `data_clean_step2` | 在数据预处理中、Step2中datajucier所运行的环境| `requirements_step2.txt` |

- 数据处理此处分为四步，具体如下：
```python
		  【hfd数据拉取】	
          📂 raw data (PDF/HTML/TXT)
            ↓  (数据转换)-----------------------------------------------【Step 1】
		  📂 JSONL {"text": "..."}
            ↓  (Datajucier数据清洗)-------------------------------------【Step 2】
          📂 cleaned JSONL
            ↓  (tokenizer预处理)----------------------------------------|
          📂 Tokenized JSONL {"tokens": [...]}                         |【Step 3】
            ↓  (token拼接，变成一个一个句子)----------------------------- |
          📂 Pretrain Data (bin/lmdb)
		  	↓  (Dataset封装)---------------------------------------------【Step4：这一步具体由预训练代码导入】
		  📂 Dataset (Pytorch)
```

## 3.环境搭建
### 3.1 **环境的详细搭建流程**：

- **推荐使用autodl云端配置**

- **此环境是在autodl中租赁下创建的，使用的配置为：**

```
镜像 PyTorch  2.1.2 Python  3.10(ubuntu22.04) CUDA  11.8  # 请勿轻易更换
GPU RTX 4090(24GB) * 2 升降配置
CPU 32 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
内存 240GB
硬盘 系统盘:30 GB
     数据盘:免费:50GB SSD  付费:100GB
附加磁盘 无
端口映射 无
自定义服务端口协议 6006端口：http 6008端口：http 
网络 同一地区实例共享带宽
```
#### 3.1.1 下载此仓库
```bash
cd /root/autodl-tmp

# 克隆到临时文件夹（避免目录冲突）
git clone https://github.com/sea-abc/Mini-DeepseekV3.git temp_repo

# 将临时文件夹中的内容移动到当前目录
mv temp_repo/* .

# 删除临时文件夹（可选）
rm -rf temp_repo
```

#### 3.1.2 环境创建
- 预训练**环境 pretraining**
```bash
cd /root/autodl-tmp/mini-deepseekv3

# 建立虚拟环境
python3 -m venv pretraining
source pretraining/bin/activate

# 安装依赖
pip install --upgrade pip setuptools wheel
pip install -r requirements_pretraining.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 检验模型能否顺利运行，并输出当前模型参数量和存储空间大小
python test_model.py

```
- 数据预处理 **拉取数据** 时所需要的包安装
```bash
cd /root/autodl-tmp/mini-deepseekv3
source pretraining/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
#更新软件包索引
apt update
#安装aria2c
apt install aria2
#安装git-lfs
apt install git-lfs 
#安装jq辅助下载
apt install jq -y
```
- 数据预处理 **Step1中JOSNL化** 的包安装
```bash
cd /root/autodl-tmp/mini-deepseekv3
source pretraining/bin/activate

# 环境补完
pip install zstandard
pip install pyarrow
```
- 数据预处理 **Step3中Tokenizer和打包bin** 的包安装
```bash
# 配置环境
pip install lightning
pip install jsonargparse
```
- 正式 **分布式预训练** 时的包安装
```bash
# 安装指定版本（0.16.1）的 DeepSpeed 库
pip install deepspeed==0.16.1
```
---
---

- 数据预处理 **Step2** 中使用Datajuicer 所需要**环境 data_clean_step2**
```bash
cd /root/autodl-tmp/mini-deepseekv3

# 安装 Python 3.10 版本的核心解释器、虚拟环境工具和开发依赖包
apt update
apt install -y python3.10 python3.10-venv python3.10-dev

# 建立虚拟环境
python3.10 -m venv data_clean_step2
source data_clean_step2/bin/activate

# 设置huggingface镜像
export HF_ENDPOINT=https://hf-mirror.com

# 安装依赖
pip install --upgrade pip
pip install --upgrade pip setuptools wheel
apt-get install -y build-essential cmake
pip install -r requirements_data_step2.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置data-jucier
source data_clean_step2/bin/activate
pip install --upgrade starlette
pip install --upgrade fastapi

# 安装的版本是1.4.4的
cd /root/autodl-tmp/mini-deepseekv3/data/step2
git clone https://github.com/modelscope/data-juicer.git

cd /root/autodl-tmp/mini-deepseekv3/data/step2/data-juicer

# 安装最小依赖
pip install -v -e .

# 升级 jsonargparse（确保兼容）
pip install --upgrade jsonargparse

# 验证安装
python -c "import data_juicer as dj; print(dj.__version__)"
```

## 4.文件夹说明
## 5.使用流程
### 5.1 数据拉取与预处理

#### 5.1.1 数据集说明
| 数据集编号 | 数据集名称 | 数据属性 | 数据量级 | 存储格式 | 是否经过数据清洗 |
| --- | --- | --- | --- | --- | --- |
| 1   | Skypile-150B | 中文文本 | 620GB | JSONL | 是   |
| 2   | SlimPajama | 以英文为主的混合语言文本 | 1TB | JSONL（压缩） | 是   |
| 3   | Starcoder | 代码数据 | 768GB | Parquet | 是   |
| 4   | OpenR1Math | 数学CoT数据 | 8G+ | JSONL | 是   |
| 5   | **APE210K(用来跑通代码)** | **中文数学CoT数据** | 49MB | JSONL | 是   |
#### 5.1.2 <span style="color:red;">**数据（raw data）拉取**</span>
- 请自行申请[Huggingface账号](https://huggingface.co/)与[申请AccessToken](https://huggingface.co/settings/tokens)，注意代码数据集Starcoder还需要在[数据集页面](https://huggingface.co/datasets/bigcode/starcoderdata)进行申请
- 首先需要将数据拉取到data/raw_data的各个子数据集目录下 **(这里都只拉取一点点作为示例，实际预训练时请尽量拉取所有数据)**
- 数据拉取准备
```bash
export HF_ENDPOINT=https://hf-mirror.com

cd /root/autodl-tmp/mini-deepseekv3/data/raw_data
source /root/autodl-tmp/mini-deepseekv3/pretraining/bin/activate

# 赋予hfd_revised.sh调用权限
chmod a+x hfd_revised.sh
```

- 数据正式拉取：

**拉取的速度一般都比较快，如果过慢(3h+)，证明aria2c可能没有正常使用，建议重新安装一次** `数据预处理step1所需要的包安装`

[**skypile**](https://huggingface.co/datasets/Skywork/SkyPile-150B/)：
```bash
#【确保你的系统可以执行4线程、5文件并行拉取，再大会报错】
./hfd_revised.sh Skywork/SkyPile-150B --dataset \
  --include "data/2020-45_zh_head_0010.jsonl" \
  --include "data/2021-04_zh_head_0011.jsonl" \
  --include "data/2021-17_zh_middle_0012.jsonl" \
  --include "data/2021-17_zh_head_0009.jsonl" \
  --hf_username 这里输入你自己的用户名 \
  --hf_token 这里输入你自己申请的文本密钥 \
  --tool aria2c \
  -x 4 -j 5\
  --local-dir /root/autodl-tmp/mini-deepseekv3/data/raw_data/skypile
```
  [**Slim Pajama**](https://huggingface.co/datasets/cerebras/SlimPajama-627B/):
```bash
./hfd_revised.sh cerebras/SlimPajama-627B --dataset \
  --include "train/chunk1/example_train_0.jsonl.zst" \
  --include "train/chunk2/example_train_10.jsonl.zst" \
  --include "train/chunk3/example_train_1001.jsonl.zst" \
  --hf_username 这里输入你自己的用户名 \
  --hf_token 这里输入你自己申请的文本密钥 \
  --tool aria2c \
  -x 1 -j 1 \
  --local-dir /root/autodl-tmp/mini-deepseekv3/data/raw_data/slimpajama
```
  [**Starcoder**](https://huggingface.co/datasets/bigcode/starcoderdata):
```bash
#【确保你的系统可以执行4线程、5文件并行拉取】
./hfd_revised.sh bigcode/starcoderdata --dataset \
  --include "python/train-00000-of-00059.parquet" \
  --include "sql/train-00000-of-00011.parquet" \
  --include "matlab/*" \
  --hf_username 这里输入你自己的用户名 \
  --hf_token 这里输入你自己申请的代码的密钥 \
  --tool aria2c \
  -x 4 -j 5 \
  --local-dir /root/autodl-tmp/mini-deepseekv3/data/raw_data/starcoder
```
[**OpenR1-Math-220K**](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k):
```bash
./hfd_revised.sh open-r1/OpenR1-Math-220k --dataset \
  --include "all/default-00000-of-00010.parquet" \
  --include "all/default-00001-of-00010.parquet" \
  --include "all/default-00002-of-00010.parquet" \
  --hf_username 这里输入你自己的用户名 \
  --hf_token 这里输入你自己申请的文本密钥 \
  --tool aria2c \
  -x 4 -j 5 \
  --local-dir /root/autodl-tmp/mini-deepseekv3/data/raw_data/openr1math
```
[**APE210K**](https://huggingface.co/datasets/MU-NLPC/Calc-ape210k):
```bash
./hfd_revised.sh MU-NLPC/Calc-ape210k --dataset \
  --include "data/*" \
  --hf_username 这里输入你自己的用户名 \
  --hf_token 这里输入你自己申请的文本密钥 \
  --tool aria2c \
  -x 4 -j 5 \
  --local-dir /root/autodl-tmp/mini-deepseekv3/data/raw_data/ape
```
#### 5.1.3 Step1：JSONL化
```bash
cd /root/autodl-tmp/mini-deepseekv3/data/step1
source /root/autodl-tmp/mini-deepseekv3/pretraining/bin/activate

# 数据会存储在data_step1中
python step1_pretrain_basic_dataprocess.py main
```
#### 5.1.4 Step2：Datajucier数据清洗
```bash
cd /root/autodl-tmp/mini-deepseekv3/data/step2
deactivate
source /root/autodl-tmp/mini-deepseekv3/data_clean_step2/bin/activate

export HF_ENDPOINT=https://hf-mirror.com

# 运行下列代码、将jsonl分割成100~500MB的小文件
# 有助于提升data_jucier的并行化速度
python split_jsonl.py

# 每个数据分开运行，一起容易报错
# 检验data-jucier能否顺利运行
# 【TIME WARNING：8进程并行】
chmod +x run_step2_ape.sh
bash run_step2_ape.sh

# 【TIME WARNING：8进程并行】
chmod +x run_step2_openr1.sh
bash run_step2_openr1.sh

# 【TIME WARNING：32进程并行】
# 中间还需额外下载一些筛选模型、务必设置hf镜像站的环境变量
chmod +x run_step2_skypile.sh
bash run_step2_skypile.sh

# 【TIME WARNING：8进程并行】
# 中间还需额外下载一些筛选模型、务必设置hf镜像站的环境变量
chmod +x run_step2_slimpajama.sh
bash run_step2_slimpajama.sh

# 【TIME WARNING：32进程并行】
chmod +x run_step2_starcoder.sh
bash run_step2_starcoder.sh
```

- 如果中途报错，请清理所有缓存后重新来过：  
    彻底释放显存 ↓ 否则显存会持续被占用。

```bash
source ~/.bashrc

pkill -9 python

rm -rf /root/autodl-tmp/mini-deepseekv3/data/step2/data_jucier_cache
rm -rf /root/autodl-tmp/mini-deepseekv3/data/step2/data_jucier_cache/temp
```
#### 5.1.5 Step3：Tokenizer与合并
```bash
# 先删一些多余的带stats的json文件
# 1. 先查看
find /root/autodl-tmp/mini-deepseekv3/data/step2/data_step2 -name "*_stats.jsonl" -type f
# 2. 确认后删除
find /root/autodl-tmp/mini-deepseekv3/data/step2/data_step2 -name "*_stats.jsonl" -type f -delete


# 使用训练的虚拟环境
deactivate
cd /root/autodl-tmp/mini-deepseekv3
source /root/autodl-tmp/mini-deepseekv3/pretraining/bin/activate

# 进入代码执行目录
cd /root/autodl-tmp/mini-deepseekv3/data/step3

# 执行step3代码
#【TIME WARNING：64进程并行】
python step3_prepare_data_for_pretrain.py
```
### 5.2 分布式预训练启动
```shell
# 退出当前环境
deactivate

cd /root/autodl-tmp/mini-deepseekv3
source /root/autodl-tmp/mini-deepseekv3/pretraining/bin/activate

# 改变deepspeed版本以支持zero-1数据并行
# 创建 Triton Inference Server（深度学习推理优化工具）自动调优功能所需的配置目录，确保 Triton 在运行自动调优时能正常读写相关文件，避免因目录缺失导致的功能异常。
mkdir -p ~/.triton/autotune

# 使用 DeepSpeed 分布式训练，2张GPU并行、单CPU上64线程并行、执行 10个 epoch
# master_port:官方接口参数
deepspeed --master_port 29500 --num_gpus=2 pretrain.py --epochs 10
```
## 6.注意事项
- 仅供个人学习使用，请勿商用
- 本项目仅供学习和研究使用。相关项目的许可证请参考各自的开源协议。
- 代码基于DeepSeek官方实现进行了适配和修改
  
## 7.常见问题（FAQ）

**Q1: 数据拉取方面的注意事项**

- 请自行申请Huggingface账号与AccessToken
- Starcoder数据集需要在数据集页面进行额外申请
- 使用镜像站加速下载：`export HF_ENDPOINT=https://hf-mirror.com`
- 如果下载速度过慢（超过3小时），请检查aria2c是否正常安装

**Q2: 训练时出现CUDA out of memory错误怎么办？**

A: 可以尝试以下方法：
   - 减小batch size
   - 使用gradient checkpointing
   - 减少模型层数或隐藏层维度
   - 增加GPU数量

**Q3: DataJuicer运行失败怎么办？**

A: 检查以下几点：
   - 确保使用的是`data_clean_step2`环境
   - 检查HF_ENDPOINT环境变量是否设置
   - 清理缓存后重新运行
   - 检查是否有足够的磁盘空间
   - 遇到`pyarrow.lib.ArrowCapacityError`时，减少并行度或移除某些算子
   - 遇到`OSError: [Errno 28] No space left on device`时，清理磁盘空间
   - 中途报错时需要清理缓存：`rm -rf /root/autodl-tmp/mini-deepseekv3/data/step2/data_jucier_cache`

**Q4: 如何调整模型参数？**

A: 将`model/config.py`中的参数配置直接复制到deepseekv3_mtp_model.py文件中，然后重新运行`test_model.py`验证。
## 参考链接

- [DeepSeek官方GitHub推理链接](https://github.com/deepseek-ai/DeepSeek-V3)
- [DataJuicer官方GitHub链接](https://github.com/datajuicer/data-juicer)
- [LitGPT](https://github.com/Lightning-AI/lit-gpt)
