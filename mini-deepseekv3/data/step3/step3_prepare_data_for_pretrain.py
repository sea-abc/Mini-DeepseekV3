"""
数据预处理脚本，
用于将多个JSONL格式的文本文件（如数学题、代码等数据集）通过Hugging Face的tokenizer转换为token序列，
并打包成二进制格式（.bin文件），以便高效地用于训练大型语言模型；
利用multiprocessing模块实现多进程并行处理，将文件分块分配给不同进程，
每个进程独立处理文本编码、累积tokens至指定大小后写入文件，
从而加速数据处理并确保数据完整性（如处理超长文本和填充不足部分）。
"""
import glob
import json
import os
import sys
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
from tqdm import tqdm
import random
import time
from pathlib import Path
from typing import List

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from transformers import AutoTokenizer

random.seed(666)

def process_line_text(text, tokenizer, max_length=131072):
    """
    负责对文本进行 Tokenizer 处理，并把超长 Token 切分成多个小块进行存储。
    其中max_length是tokenizer的config设置中规定的数字。
    """
    text_ids = tokenizer.encode(text)

    # ✅ 防御性检查：过滤掉异常 token
    if any(t >= tokenizer.vocab_size for t in text_ids):
        print(f"🚨 非法 token 出现（超过 vocab_size）: {max(text_ids)}")
        return []

    text_ids = np.array(text_ids, dtype=np.int32)

    if len(text_ids) > max_length:
        print(f"⚠️ 超长文本 ({len(text_ids)} tokens) 被拆分！")
        return [text_ids[i:i+max_length] for i in range(0, len(text_ids), max_length)]
    else:
        return [text_ids]

def process_jsonl_files(set_name, file_dir_list, builder, tokenizer, chunk_size, max_length=131072):
    """
    处理多个 jsonl 文件，确保累计足够的 tokens 才存入 bin
    """
    cache_tokens = []
    total_tokens = 0  # 记录已经积累的 token 数量

    for file_dir in file_dir_list:
        print(f"📖 处理 JSONL 文件: {file_dir}")
        with open(file_dir, encoding="utf-8") as f:
            counter = 0
            for line in tqdm(f, desc=f"📖 Reading {file_dir}"):
                try:
                    text = json.loads(line)["text"]
                    #print(text[:100])
                except Exception as e:
                    print(f"⚠️ 文件 {file_dir} 读取错误: {e}")
                    continue

                text += "<｜end▁of▁sentence｜>"  # ✅ 追加结束符，结束符要与tokenizer相匹配

                # ✅ 处理 `text`，确保它不会被错误拆分，并进行分块
                text_id_chunks = process_line_text(text, tokenizer, max_length)

                # ✅ **累积 tokens，直到满足 `chunk_size`**
                for chunk in text_id_chunks:
                    cache_tokens.extend(chunk)
                    total_tokens += len(chunk)

                    # ✅ **如果 tokens 数量达到 `chunk_size`，才写入 `.bin`**
                    if total_tokens >= chunk_size:
                        print(total_tokens)
                        print(f"🧐 Adding to builder: {len(cache_tokens[:chunk_size])} tokens")
                        print(f"🧐 First 20 tokens: {cache_tokens[:20]}")
                        builder.add_array(np.array(cache_tokens[:chunk_size], dtype=np.uint32))
                        print(f"🚀 写入 bin: {len(cache_tokens[:chunk_size])} tokens")
                        cache_tokens = cache_tokens[chunk_size:]  # ✅ 只保留未存的部分
                        total_tokens = len(cache_tokens)

    # ✅ **处理最后的 tokens（如果不足 `chunk_size`，则填补 `pad_token`）**
    if total_tokens > 0:
        padding_needed = chunk_size - total_tokens
        cache_tokens.extend([tokenizer.pad_token_id] * padding_needed)  # ✅ 填补 `pad_token`
        builder.add_array(np.array(cache_tokens, dtype=np.uint32))
        print(f"🚀 最后存入 bin: {total_tokens} tokens (填补 {padding_needed} 个 pad tokens)")

def multiprocess_data(set_name, file_dir_list, destination_path, chunk_size, checkpoint_dir, process_idx=0):
    """
    负责多进程调用 process_jsonl_files 处理多个 JSONL 文件
    """
    try:
        t0 = time.time()

        # ✅ 每个子进程单独加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # ✅ **每个进程独立创建 `PackedDatasetBuilder`**
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=f"{set_name}_process{process_idx}", # ✅ 避免多个进程写入相同文件
            chunk_size=chunk_size,
            sep_token=tokenizer.pad_token_id,
            dtype=np.int32,
            vocab_size=len(tokenizer),
        )

        # ✅ **跨多个 JSONL 文件，累积 tokens**
        process_jsonl_files(set_name, file_dir_list, builder, tokenizer, chunk_size)
        # ✅ **每个进程独立调用 `write_reminder()`，确保数据写入**
        builder.write_reminder()

        print(f"✅ Process {process_idx} 处理 {len(file_dir_list)} 个文件，总耗时 {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"⚠️ multiprocess_data 处理 {set_name} 发生错误: {str(e)}")

def prepare_full(
    source_path: Path,
    checkpoint_dir: Path,
    destination_path: Path,
    chunk_size: int,
    match: str = "",
    max_files: int = 1_000_000_000,
    process_num: int = 64
):
    """
    遍历 JSONL 文件，使用 multiprocessing 并行处理成 .bin 格式。

    参数说明：
    - source_path: JSONL 文件的根目录
    - checkpoint_dir: tokenizer 的目录
    - destination_path: 输出目录
    - chunk_size: 每个 chunk 的 token 数量
    - match: 可选，用于筛选某个 set_name
    - max_files: 最多处理多少个文件
    - process_num: 并行进程数
    """
    # ✅ 确保输出目录存在
    destination_path.mkdir(parents=True, exist_ok=True)

    # ✅ 遍历你配置的每一类数据（例如 train / val / test）
    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue

        print(f"\n📂 准备处理数据集: {set_name}")
        t0 = time.time()

        # ✅ 使用 pathlib 匹配文件
        filenames = sorted([Path(p) for p in glob.glob(pattern, recursive=True)])
        if not filenames:
            print(f"⚠️ 没有找到符合 pattern `{pattern}` 的文件，跳过 {set_name}")
            continue

        random.shuffle(filenames)
        filenames = filenames[:max_files]
        print(f"📄 文件总数: {len(filenames)}")

        # ✅ 限制进程数量
        actual_process_num = min(process_num, len(filenames))
        file_chunks = np.array_split(filenames, actual_process_num)

        # ✅ 启动多进程
        process_list = []
        for process_idx in range(actual_process_num):
            sub_file_list = [str(p) for p in file_chunks[process_idx]]
            print(f"🚀 启动进程 {process_idx}，文件数: {len(sub_file_list)}")
            process = mp.Process(
                target=multiprocess_data,
                args=(set_name, sub_file_list, destination_path, chunk_size, checkpoint_dir, process_idx)
            )
            process.start()
            process_list.append(process)

        # ✅ 等待所有进程完成
        for process in process_list:
            process.join()

        print(f"✅ {set_name} 数据处理完成，耗时 {time.time() - t0:.2f} 秒")

filename_sets = {
    "djed_openr1": "/root/autodl-tmp/mini-deepseekv3/data/step2/data_step2/openr1math/*.jsonl",
    "djed_ape210K": "/root/autodl-tmp/mini-deepseekv3/data/step2/data_step2/ape/*.jsonl",
    "djed_starcoder": "/root/autodl-tmp/mini-deepseekv3/data/step2/data_step2/starcoder/*.jsonl",
    "djed_skypile": "/root/autodl-tmp/mini-deepseekv3/data/step2/data_step2/skypile/*.jsonl",
    "djed_slimpajama": "/root/autodl-tmp/mini-deepseekv3/data/step2/data_step2/slimpajama/*.jsonl"
}

def prepare(
    source_path: Path = Path("/"),
    checkpoint_dir: Path = Path("/root/autodl-tmp/mini-deepseekv3/data/step3"),
    destination_path: Path = Path("/root/autodl-tmp/mini-deepseekv3/data/step3/data_step3"),
    sample: bool = False,
    match: str = "",
    max_files=10000000000,     
    block_size= 2048,            
    blocks_in_a_chunck= 1024 * 20,  
    process_num=64      
) -> None:
    """程序的主入口，负责调用 prepare_full 并提供 tokenizer 路径等"""
    prepare_fn = prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(block_size + 1) * blocks_in_a_chunck,  # ✅ 直接按 chunk_size 控制
        match=match,
        max_files=max_files,
        process_num=process_num  # ✅ 删除 cache_lines_num
    )

if __name__ == "__main__":
    """允许命令行调用 prepare 进行数据预处理"""
    import jsonargparse
    from jsonargparse import CLI
    CLI(prepare)
