"""
用于配置Python全局日志记录器的工具文件，专门为分布式训练环境设计。它的核心作用是确保程序运行时的日志信息（如调试信息、状态报告、错误详情等）能同时输出到文件和控制台。其中，所有级别的日志都会保存到traininglogs目录下按进程号区分的文件中（例如train_log_rank0.log），便于事后追溯分析；而控制台则只显示INFO级别以上的关键信息，保持终端界面的简洁。通过识别环境变量中的RANK值，它能自动为不同进程创建独立的日志器，有效避免多进程日志写入冲突
"""
# logger_utils.py
import logging
import os
import time

_logger_instance = None

def get_rank_for_logger():
    return int(os.environ.get("RANK", 0))

def setup_logger():
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    rank = get_rank_for_logger()
    logger = logging.getLogger(f"train_logger_rank{rank}")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # 创建目录
        log_dir = "traininglogs"
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"train_log_rank{rank}.log")
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            f"[%(asctime)s][%(levelname)s][RANK {rank}] %(message)s",
            "%H:%M:%S"
        )
        formatter.converter = time.localtime
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream Handler（终端只打印 INFO 及以上）
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)  # ✅ 只显示 INFO 以上的日志
        sh_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        sh.setFormatter(sh_formatter)
        logger.addHandler(sh)

    _logger_instance = logger
    return logger