# coding=utf-8
# region 导包
import os
import io
import json
import time
import sys
import re
import unicodedata
import pyarrow.parquet as pq
import traceback
import zstandard as zstd
import pyarrow.parquet as pq
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
# endregion

"""
格式处理类，用于递归获取指定路径下的目标文件（.jsonl/.parquet），
并对文件内容进行逐行处理、去重、错误统计，最终将有效数据写入输出文件
"""
class FormatHandler():
    """
    类的构造方法，用于初始化格式处理所需的核心参数
    输入参数：
        input_path（str）: 待处理文件的根目录路径，用于递归查找目标文件
        output_path（str）: 处理后有效数据的输出文件路径
        dataset_name（str）: 数据集名称，用于标识当前处理的数据集（当前代码暂未直接使用，预留扩展）
    """
    def __init__(self, input_path, output_path, dataset_name):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.seen_texts = set() #初始化去重的set

    def get_file_list(self) -> list:
        """
        递归获取输入路径下所有后缀为.jsonl或.parquet的文件，返回文件路径列表
        输入参数：
            无（依赖实例初始化时的input_path属性）
        返回值：
            list: 所有符合条件的文件的完整路径列表，每个元素为str类型的文件路径
        """
        file_list = []
        for root, _, files in os.walk(self.input_path):  # 递归遍历所有子目录
            for file in files:
                if file.endswith(".jsonl") or file.endswith(".parquet"):
                    file_list.append(os.path.join(root, file))
        return file_list
    
    def process_one_line(self, line, fout) -> bool:
        """
        抽象方法：处理单行数据的核心逻辑，需由子类实现具体业务逻辑
        输入参数：
            line（str）: 待处理的单行原始数据（通常来自.jsonl文件的一行文本）
            fout（file object）: 输出文件对象，用于子类直接写入处理后的数据（当前父类未使用，预留子类扩展）
        返回值：
            bool: 子类需返回处理后的文本（str）或False（表示该行数据需过滤）
        异常说明：
            因未实现具体逻辑，直接调用会抛出NotImplementedError，强制子类必须重写该方法
        """
        raise NotImplementedError
    
    def process_one_file(self, file_path, max_lines=None) -> (int, int):
        """
        处理单个文件的完整流程：读取文件内容、逐行处理、去重、错误统计、写入有效数据
        输入参数：
            file_path（str）: 待处理单个文件的完整路径
            max_lines（int, 可选）: 最大处理行数，若指定则仅处理前max_lines行数据，默认None（处理所有行）
        返回值：
            tuple: 包含两个int元素的元组，第一个元素为成功处理的有效行数，第二个元素为跳过/错误的行数
        """
        line_count = 0
        jump_count = 0
        seen_texts = set()  # ✅ 这里用 set() 记录已处理文本
    
        file_path_full = file_path
        with open(self.output_path, "a", encoding="utf-8") as fout:
            with open(file_path_full, "r", encoding="utf-8") as fin:
                for line in fin:
                    if max_lines and line_count >= max_lines:
                        break  # ✅ 只读取 max_lines 行
                    try:
                        processed_text = self.process_one_line(line, fout)  # ✅ 先处理文本
                        if not processed_text:
                            jump_count += 1  # ✅ 统计被过滤的行数
                            continue  
    
                        text = processed_text.strip()
                        if text in seen_texts:
                            jump_count += 1  # ✅ 跳过重复项
                            continue  
    
                        seen_texts.add(text)  # ✅ 记录新的文本
                        if fout is not None:
                            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")  # ✅ 保证写入去重文本
                        line_count += 1
                    except Exception as e:
                        print(f"[Error] 处理行失败: {e}")
                        jump_count += 1  # 记录错误行
        return line_count, jump_count
    
    def process_all(self, max_lines=None):
        """
        处理全部文件的完整流程：递归获取所有.jsonl或.parquet文件，逐个调用process_one_file处理，统计总有效行数、跳过/错误行数
        输入参数：
            max_lines（int, 可选）: 最大处理行数，若指定则仅处理每个文件的前max_lines行数据，默认None（处理所有行）
        返回值：
            无（直接打印处理结果，不返回值）
        """
        st = time.time()
        line_count_all = 0
        jump_count_all = 0
        file_list = self.get_file_list()
        print("[log][{}] number of files is {:d}".format(self.dataset_name, len(file_list)))
        for file in file_list:
            line_count, jump_count = 0, 0
            try:
                line_count, jump_count = self.process_one_file(file, max_lines=max_lines)
            except Exception as e:
                print("[exception][{}] process file {} failed: {}".format(self.dataset_name, file, e))
                print(traceback.format_exc())
            line_count_all += line_count
            jump_count_all += jump_count
        print("[log][{}] timecost is {:.2f} s!".format(self.dataset_name, time.time() - st))
        print("[log][{}] line_count is {:d}, jump_count is {:d}".format(self.dataset_name, line_count_all, jump_count_all))
        print(f"[log] {self.dataset_name} 数据处理完毕！")  

    def quality_assurance(self, line) -> bool:
        """
        质量检查逻辑：检查文本是否符合基本要求，如长度、换行符数量等
        输入参数：
            line（str）: 待检查的原始文本行
        返回值：
            bool: True表示文本符合质量要求，False表示需过滤
        """
        if len(line) < 10:  # 允许短文本，但限制过短的内容
            print("文本太短")
            return False
        if line.count("\n") > 200:  # 提高换行阈值
            print("换行符太多，一共{}".format(line.count("\n")))
            return False
        return True

    def quality_assurance_math(self, line) -> bool:
        """
        质量检查逻辑：检查数学表达式是否符合基本要求，如长度、换行符数量等
        输入参数：
            line（str）: 待检查的原始文本行
        返回值：
            bool: True表示文本符合质量要求，False表示需过滤
        """
        if len(line) < 10:  # 允许短文本，但限制过短的内容
            print("文本太短")
            return False
        return True

    def zh_process(self, line) -> str:
        """
        中文文本处理：对输入文本进行标准化处理，包括 unicode 统一、换行符替换、控制字符移除等
        输入参数：
            line（str）: 待处理的原始文本行
        返回值：
            str: 处理后的文本行，符合质量要求
        """
        # 0. None 处理成空字符串
        if line is None:
            return ""
        # 1. unicode 统一
        line = unicodedata.normalize("NFKC", line)
        # 2. 替换\n
        # line = line.replace("\n\n", "\n")
        # 3. 移除 Unicode 控制字符（如 \u200b, \ufeff）
        line = "".join(ch for ch in line if unicodedata.category(ch)[0] != "C")
        # 4. 将\r替换成\t
        line = line.replace("\r", "").replace("\t", " ")  # 规范换行和制表符
        line = line.strip()  # 去除前后空格
        return line


#################
#Skyiple
#################

class SkypileFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        """
        :param input_path: Skypile 数据文件所在目录
        :param output_path: Skypile 数据输出目录
        :param dataset_name: 数据集名称
        :param sample_rate: 多少行进行一次抽样检查
        """
        super().__init__(input_path, output_path, dataset_name)
        self.seen_texts = set()  # ✅ 添加去重集合

    def get_file_list(self) -> list:
        """递归获取所有子目录中的 .jsonl 文件"""
        file_list = []
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if file.endswith(".jsonl"):
                    file_list.append(os.path.join(root, file))
        return file_list

    def process_one_line(self, line, fout) -> bool:
        """检查一行数据是否符合 {'text': 'xxxx'} 格式，并去重 & 进行写入"""
        try:
            line = line.strip()  # 去掉空格和换行
            data = json.loads(line)  # 确保 JSON 解析成功
            text = data.get("text", "").strip()
    
            # 检查 text 是否为空或是重复
            if not text:
                print(f"[DEBUG] 跳过空白")  # 打印空白文本的行
                return False
            if text in self.seen_texts:
                print(f"[DEBUG] 跳过重复")  # 打印重复文本的内容
                return False
    
            # 如果没有跳过，则添加到 seen_texts 并写入
            self.seen_texts.add(text)  # 记录新的文本
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")  # 直接写入
            return True  # 处理成功
    
        except Exception as e:
            print(f"[ERROR] 处理行失败: {e}")
            return False

    def process_one_file(self, file_path, max_lines=None) -> (int, int):
        """对单个 Skypile 文件进行格式检查并写入 JSONL"""
        line_count = 0
        invalid_count = 0
        total_checked = 0

        with open(file_path, "r", encoding="utf-8") as fin, open(self.output_path, "a", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if max_lines and line_count >= max_lines:  # 限制读取行数
                    break
                
                line_count += 1
                total_checked += 1
                if not self.process_one_line(line, fout):  # 现在 `process_one_line` 负责写入
                    invalid_count += 1
                    print(f"[WARNING] {self.dataset_name}: {file_path} - 格式错误或重复")

        print(f"[INFO] {self.dataset_name}: {file_path} - 总计 {line_count} 行，发现 {invalid_count} 行格式错误。")
        return line_count, invalid_count
    
    def process_all(self, max_lines=None):
        """遍历所有 Skypile 数据文件，进行格式检查 & 生成 JSONL"""
        st = time.time()
        line_count_all = 0
        jump_count_all = 0
        file_list = self.get_file_list()
        print(f"[INFO] {self.dataset_name}: 发现 {len(file_list)} 个文件。")
    
        for file in file_list:
            try:
                line_count, jump_count = self.process_one_file(file, max_lines=max_lines)
            except Exception as e:
                print(f"[EXCEPTION] {self.dataset_name}: 处理文件 {file} 失败: {e}")
                print(traceback.format_exc())
    
            line_count_all += line_count
            jump_count_all += jump_count
    
        print(f"[INFO] {self.dataset_name}: 处理完成！耗时 {time.time() - st:.2f} 秒，总行数 {line_count_all}，跳过 {jump_count_all} 行。")
        print(f"[log] {self.dataset_name} 数据处理完毕！")

#################
#Slimpajama
#################

class SlimpajamaFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(SlimpajamaFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def get_file_list(self) -> list:
        """递归获取所有 .jsonl.zst 文件"""
        files = []
        for root, _, filenames in os.walk(self.input_path):
            for filename in filenames:
                if filename.endswith(".jsonl.zst"):
                    files.append(os.path.join(root, filename))
        return files
    
    def process_one_file(self, file_path, max_lines=None):
        """解压 .jsonl.zst 文件并逐行处理"""
        line_count = 0
        jump_count = 0

        try:
            with open(self.output_path, "a", encoding="utf-8") as fout:
                with open(file_path, "rb") as compressed_file:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(compressed_file) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                        for i,line in enumerate(text_stream):
                            # ✅ 加上 max_lines 限制
                            if max_lines and i >= max_lines:  
                                break
                            line = line.strip()
                            if not line:
                                continue
                            line_count += 1
                            self.process_one_line(line, fout) 
        except Exception as e:
            print(f"[ERROR] 处理文件 {file_path} 失败: {e}")
            print(traceback.format_exc())

        return line_count, jump_count

    def process_one_line(self, line, fout) -> bool:
        """ 处理 Slimpajama 数据，把 {text: xxxxx} 写入 JSONL """
        try:
            data = json.loads(line)
            text = self.zh_process(data.get("text", ""))  
            if not text.strip():
                return False
            if not self.quality_assurance(text):  
                return False
            if fout is not None:
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"[Error] Failed to process line: {e}")
            return False

#################
#StarCoder
#################

class StarcoderFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name, language : list):
        super(StarcoderFormatHandler, self).__init__(input_path, output_path, dataset_name)
        self.language = language
    
    def get_file_list(self) -> list:
        """递归获取所有子目录中的 .parquet 文件"""
        file_list = []
        
        if self.language:  # ✅ 先检查是否有语言子目录
            for lan in self.language:
                path_prefix = os.path.join(self.input_path, lan)
                if os.path.exists(path_prefix):  # ✅ 先检查目录是否存在
                    for root, _, files in os.walk(path_prefix):
                        for file in files:
                            if file.endswith(".parquet"):
                                file_list.append(os.path.join(root, file))
    
        # ✅ 如果上面没有找到文件，直接遍历整个 input_path
        if not file_list:
            for root, _, files in os.walk(self.input_path):
                for file in files:
                    if file.endswith(".parquet"):
                        file_list.append(os.path.join(root, file))

        if not file_list:
            print(f"[WARNING] {self.dataset_name}: 没有找到任何 .parquet 文件！")
    
        return file_list

    def process_one_file(self, file_path, max_lines=None):
        """ 读取 Starcoder Parquet 文件 """
        
        line_count = 0
        jump_count = 0
    
        try:
            table = pq.read_table(file_path)  # ✅ 用 PyArrow 读取 Parquet
            df = table.to_pandas()  # ✅ 转换成 Pandas DataFrame
    
            # ✅ 检查关键列是否存在
            required_columns = ["content", "max_stars_count"]
            for col in required_columns:
                if col not in df.columns:
                    print(f"[WARNING] {self.dataset_name}: {file_path} 缺少关键列 {col}，跳过此文件！")
                    return line_count, jump_count
    
            # 开始逐行进行读取
            with open(self.output_path, "a", encoding="utf-8") as fout:
                for i, row in df.iterrows():
                    if max_lines and line_count >= max_lines:
                        break  # ✅ **限制读取的行数**
    
                    try:
                        stars = int(float(row["max_stars_count"])) if pd.notna(row["max_stars_count"]) else 0
                    except Exception:
                        stars = 0
    
                    # ✅ 调试：打印星级
                    #print(f"[DEBUG] 处理文件: {file_path}, 代码段 {i}, Stars: {stars}")
    
                    # ✅ 调试：先不进行星级过滤
                    # if stars < 5:
                    #     jump_count += 1
                    #     continue
    
                    text = row["content"]
                    if not isinstance(text, str):
                        print(f"[WARNING] {self.dataset_name}: 非字符串 content，跳过！")
                        jump_count += 1
                        continue
    
                    text = text.lstrip("\ufeff").strip()  # 去掉 BOM 以及前后空格
                    if not text:
                        print(f"[WARNING] 发现空白代码，跳过。")
                        jump_count += 1
                        continue
    
                    # ✅ 写入 JSONL
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    line_count += 1
    
        except Exception as e:
            print(f"[ERROR] 处理 {file_path} 失败: {e}")
            print(traceback.format_exc())
    
        return line_count, jump_count

#################
#OpenR1
#################

class OpenR1FormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(OpenR1FormatHandler, self).__init__(input_path, output_path, dataset_name)

    def get_file_list(self) -> list:
        file_list = []
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if file.endswith(".parquet"):
                    file_list.append(os.path.join(root, file))
        print(f"[DEBUG] Found {len(file_list)} files.")  # 打印文件数量和文件路径
        return file_list
    
    def process_one_file(self, file_path, max_lines=None) -> (int, int):
        """ 读取 OpenR1 的 Parquet 文件、合并problem type、problem、solution、answer、generation等内容"""
        line_count = 0
        jump_count = 0
        try:
            # 获取目录下所有的parquet文件
            
            # ✅ **用 PyArrow 读取 Parquet**
            table = pq.read_table(file_path)  
            df = table.to_pandas()  # ✅ **转换成 Pandas DataFrame**
            
            # ✅ **检查关键列是否存在**
            required_columns = ["problem_type", "question_type", "problem", "solution", "answer"]
            for col in required_columns:
                if col not in df.columns:
                    print(f"[WARNING] {self.dataset_name}: {file_path} 缺少关键列 {col}，跳过此文件！")
                    return line_count, jump_count

            with open(self.output_path, "a", encoding="utf-8") as fout:
                for i, row in df.iterrows():
                    if max_lines and line_count >= max_lines:
                        break  # ✅ **只读取 max_lines 行**                
                        
                    # 过滤无效的 problem（全是标点符号）
                    if not re.search(r'[a-zA-Z0-9\u4e00-\u9fa5]', row['problem']):  
                        print(f"[WARNING] 过滤掉无效 problem: {row['problem']}")
                        jump_count += 1
                        continue
                    
                    # 去除 problem 开头和结尾的无效字符
                    row['problem'] = row['problem'].strip('. \n\t')
                    
                    # 替换多个连续的 `.` 为一个 `.`
                    row['problem'] = re.sub(r'\.{2,}', '.', row['problem'])
                    
                    # 删除所有 `$$$$$$$$`（多个 $$）
                    row['problem'] = re.sub(r'(\${2,})', '', row['problem'])
                    
                    # 删除所有 `\qquad\qquad`
                    row['problem'] = re.sub(r'(\\qquad)+', '', row['problem'])
                    
                    text = f"[Problem Type]: {row['problem_type']}\n[Question Type]: {row['question_type']}\n\n"
                    text += f"[Question Type]:\n{row['question_type']}\n\n"
                    text += f"[Problem]:\n{row['problem']}\n\n"
                    text += f"[Solution]:\n{row['solution']}\n\n"
                    text += f"[Answer]:\n{row['answer']}\n\n"
                    text += f"[Correctness_Math_Verify]:\n{row['correctness_math_verify']}\n\n"

                    # 处理 generations（可能是多个答案）
                    generations = row.get("generations", [])
                    if isinstance(generations, list) and generations:  # 确保是列表且非空
                        text += "[Generated Solutions]:\n"
                        for gen in generations:
                            text += f"{gen}\n\n"
                    else:
                        text += f"[Generations]:\n{generations}\n\n"

                    # ✅ **质量检查**
                    if not self.quality_assurance_math(text):
                        jump_count += 1
                        print(f"[WARNING] {self.dataset_name}: {file_path} 质量检查不通过！")
                        print(f"[DETAILS] Data: {text}")  # 打印被跳过的内容，帮助定位问题
                        continue  

                    # ✅ **写入 JSONL**
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    line_count += 1

        except Exception as e:
            print(f"[ERROR] 处理 {file_path} 失败: {e}")
            print(traceback.format_exc())

        return line_count, jump_count

#################
#APE210K
#################

class APE210KFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(APE210KFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_file(self, file_path, max_lines=None):
        """ 处理 APE210K Parquet 文件 """
        line_count = 0
        jump_count = 0
    
        try:
            table = pq.read_table(file_path)  # ✅ 用 PyArrow 读取 Parquet
            data = table.to_pandas()  # ✅ 转换成 pandas DataFrame
            if "question_chinese" not in data.columns:
                print(f"[WARNING] {self.dataset_name}: {file_path} 缺少关键列，跳过处理")
                return line_count, jump_count
    
            with open(self.output_path, "a", encoding="utf-8") as fout:
                for i, row in data.iterrows():
                    if max_lines and line_count >= max_lines:
                        break  # ✅ 只读取 max_lines 行
                    
                    success = self.process_one_line(row, fout)  # ✅ 传入 DataFrame 的 row
                    if success:
                        line_count += 1
                    else:
                        jump_count += 1  # 统计被过滤的行
        
        except Exception as e:
            print(f"[ERROR] 处理 {file_path} 失败: {e}")
            print(traceback.format_exc())
    
        return line_count, jump_count
            
    def process_one_line(self, row, fout) -> bool:
        """ 处理 APE210K 数据，将多个字段合并成一个文本，只提取中文的部分 """
        try:
            # 提取关键字段
            question_chinese = row.get("question_chinese", "").strip()
            chain = row.get("chain", "").strip()
            equation = row.get("equation", "").strip()
            result = row.get("result", "").strip()
    
            # 确保 `question_chinese` 存在，避免存入无效数据
            if not question_chinese:
                return False  # 跳过无效数据
            
            # 生成格式化文本
            text = f"[Question]: {question_chinese}\n\n"
            text += f"[Solution Chain]: {chain}\n\n"
            text += f"[Equation]: {equation}\n\n"
            text += f"[Final Answer]: {result}\n"
    
            # 质量检查
            if not self.quality_assurance(text):
                return False  # 跳过低质量数据
    
            # 写入 JSONL
            if fout is not None:
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            
            return True
        except Exception as e:
            print(f"[Error] Failed to process row: {e}")
            return False

def test_run():
    """简单测试"""
    input_path_root = os.path.dirname(os.path.realpath(__file__))
    output_path_root = input_path_root + "/basic_clean"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        "skypile": (os.path.join(input_path_root, "skypile/data"), SkypileFormatHandler),
        "slimpajama": (os.path.join(input_path_root, "slimpajama/train"), SlimpajamaFormatHandler),
        "openr1": (os.path.join(input_path_root, "openr1/all"), OpenR1FormatHandler),
        "ape210k": (os.path.join(input_path_root, "ape210k/data"), APE210KFormatHandler)
    }

    for dataset_name, info in dataset_process_info.items():
        input_path = info[0]
        Handler = info[1]
        output_path = os.path.join(output_path_root, f"processed_{dataset_name}.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all(max_lines=5)

    # 代码数据单独处理
    dataset_name = "starcoder"
    input_path = os.path.join(input_path_root, "starcoder")
    output_path = os.path.join(output_path_root, f"processed_{dataset_name}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)
    fh = StarcoderFormatHandler(input_path, output_path, dataset_name, ["sql"])
    fh.process_all(max_lines=5)

def main_run():
    input_path_root = r"/root/autodl-tmp/mini-deepseekv3/data/raw_data"
    output_path_root = r"/root/autodl-tmp/mini-deepseekv3/data/step1/data_step1"

    dataset_process_info = {
        "skypile": (os.path.join(input_path_root, "skypile/data"), SkypileFormatHandler),
        "slimpajama": (os.path.join(input_path_root, "slimpajama/train"), SlimpajamaFormatHandler),
        "openr1": (os.path.join(input_path_root, "openr1math/all"), OpenR1FormatHandler),
        "ape210k": (os.path.join(input_path_root, "ape/data"), APE210KFormatHandler)
    }

    for dataset_name, info in dataset_process_info.items():
        input_path = info[0]
        Handler = info[1]
        output_path = os.path.join(output_path_root, f"step1_{dataset_name}.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()

    # 代码数据单独处理
    dataset_name = "starcoder"
    input_path = input_path_root + "/starcoder"
    output_path = os.path.join(output_path_root, f"step1_{dataset_name}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)
    fh = StarcoderFormatHandler(input_path, output_path, dataset_name, ["python", "sql","matlab"])
    fh.process_all()
    
if __name__ == "__main__":
    test_mode = True
    
    if len(sys.argv) > 1 and sys.argv[1] == "main":
            test_mode = False  # 如果参数是 "main"，就运行 main_run()
    
    if test_mode:
        print("✅ 正在运行测试模式 (test_run)...")
        test_run()
        
    else: 
        print("🚀 正在运行正式模式 (main_run)...")
        main_run()