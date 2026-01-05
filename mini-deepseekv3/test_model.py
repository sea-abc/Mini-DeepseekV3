"""
检查模型是否能正确前向传播,输出模型参数量，模型存储空间大小（MB）
"""

import torch
import os
from model.deepseekv3_mtp_model import Transformer, ModelArgs

# 设置默认数据类型和设备
torch.set_default_dtype(torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# 初始化模型参数
args = ModelArgs()

# 创建 Transformer 模型
model = Transformer(args).to(device)

# 生成测试输入数据（token IDs）
batch_size = 1
seq_length = args.max_seq_len
#seq_length = 2048  # 假设最大序列长度
dummy_input = torch.randint(0, args.vocab_size, (batch_size, seq_length), device=device)

# 检查模型是否能正确前向传播
try:
    logits, mtp_logits = model(dummy_input)  # 前向传播
    print("模型前向传播成功！logits输出尺寸:", logits.shape)
    print("模型前向传播成功！MTP输出长度:", len(mtp_logits))
    print("模型前向传播成功！MTP输出尺寸:", mtp_logits[0].shape)
    print(logits)
    print(mtp_logits)
except Exception as e:
    print("模型运行错误:", e)

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总量: {total_params:,}")

# 计算模型存储空间大小（MB）
param_size = sum(p.element_size() * p.numel() for p in model.parameters())
buffer_size = sum(p.element_size() * p.numel() for p in model.buffers())
total_size_mb = (param_size + buffer_size) / (1024 ** 2)
print(f"模型存储空间大小: {total_size_mb:.2f} MB")
