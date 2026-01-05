import json
import random
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset): 
    def __init__(self, data_path_lst, max_length=512, memmap=False, shuffle=True, max_mmap_files=512):
        """
        Args:
            data_path_lst (list): 所有 .bin 文件路径列表
            max_length (int): 每个样本的最大长度
            memmap (bool): 是否使用 memory-mapped 方式读取数据
            shuffle (bool): 是否对数据进行 shuffle
            max_mmap_files (int): 最多同时打开多少个 mmap 文件
        """
        super().__init__()
        self.vocab_size = 128000
        self.data_path_lst = data_path_lst
        self.max_length = max_length
        self.memmap = memmap
        self.shuffle = shuffle
        self.max_mmap_files = max_mmap_files
        self.dtype = np.dtype('uint32')

        # 计算数据集大小
        self.file_offsets = []
        self.total_samples = 0

        for file_path in self.data_path_lst:
            file_size = os.path.getsize(file_path)
            num_samples = (file_size // self.dtype.itemsize) // max_length
            self.file_offsets.append((file_path, self.total_samples, num_samples))
            self.total_samples += num_samples

        # ✅ 懒加载 mmap 缓存池
        if self.memmap:
            self.mmaps = OrderedDict()  # ✅ 使用有序字典控制缓存数量，同时不再提前打开所有文件
        else:
            self.mmaps = None

        # 生成索引列表
        self.indices = list(range(self.total_samples))
        if self.shuffle:
            random.shuffle(self.indices)

        print(f"Loaded {len(self.file_offsets)} files, total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        index = self.indices[index]

        # 找到 index 所属的文件
        for file_idx, (file_path, start_idx, num_samples) in enumerate(self.file_offsets):
            if index < start_idx + num_samples:
                local_index = index - start_idx
                break
        else:
            raise IndexError("Index out of range")

        if self.memmap:
            # ✅ 懒加载 mmap 文件
            if file_path not in self.mmaps:
                # 如果超出限制，移除最久未访问的 mmap 文件
                if len(self.mmaps) >= self.max_mmap_files:
                    removed_path, _ = self.mmaps.popitem(last=False)
                    # print(f"🧹 Removed mmap cache: {removed_path}")

                self.mmaps[file_path] = np.memmap(file_path, dtype=self.dtype, mode='r')
            else:
                # 更新访问顺序
                self.mmaps.move_to_end(file_path)

            mmap_obj = self.mmaps[file_path]
            sample = mmap_obj[local_index * self.max_length : (local_index + 1) * self.max_length]
        else:
            # 非 memmap 模式
            with open(file_path, 'rb') as f:
                f.seek(local_index * self.max_length * self.dtype.itemsize)
                sample = np.frombuffer(f.read(self.max_length * self.dtype.itemsize), dtype=self.dtype)

        X = np.array(sample[:-1], dtype=np.int64)
        Y = np.array(sample[1:], dtype=np.int64)

        retry = 0
        if X.max() >= self.vocab_size or Y.max() >= self.vocab_size:
            retry +=1
            print(f"⚠️ 非法 token_id 出现，跳过该样本 (max X={X.max()}, max Y={Y.max()})")
            if retry > 10:
                raise ValueError("⚠️ 连续多次非法 token，可能数据整体有问题")
            return self.__getitem__((index + 1) % self.__len__())  # 简单跳过当前样本
    
        return torch.from_numpy(X), torch.from_numpy(Y)

class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.padding = 0  # self.tokenizer.special_tokens['<pad>']
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def find_sublist_index(self, main_list, sub_list) -> int:
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])
        q = str(sample['q'])
        a = str(sample['a'])

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        # 0表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor


if __name__ == "__main__":
    pass