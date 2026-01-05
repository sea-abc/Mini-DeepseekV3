import os
import gc
import platform
import argparse
import time
import math
import warnings
import glob
import json
import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from contextlib import nullcontext
from model.deepseekv3_mtp_model import Transformer, ColumnParallelLinear, RowParallelLinear
from model.deepseekv3_mtp_model import ModelArgs
from data.step4.step4_dataset import PretrainDataset
from pathlib import Path
from model.deepseekv3_mtp_model import get_rank, get_world_size
from logger_utils import setup_logger
import logging

torch.autograd.set_detect_anomaly(True)

logger = setup_logger()

#不需要调试时用这一行
logger.setLevel(logging.INFO)

#需要调试时启用下面这一行
#logger.setLevel(logging.DEBUG)


logger.info("🚀 Logger initialized")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

warnings.filterwarnings('ignore')

def create_ds_config():
    # ds_config动态生成函数
    ds_config = {
        "gradient_accumulation_steps": args.accumulation_steps,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 1},  # ✅ 启用 ZeRO-1
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": 10000
            }
        }
    }

    # ✅ 将 JSON 数据写入文件
    config_path = os.path.join(args.save_dir, "ds_config.json")
    with open(config_path, "w") as f:
        json.dump(ds_config, f, indent=4) 

    return config_path

def print_memory(tag=""):
    logger.debug(f"\n📌 CUDA Memory Status {tag}:")
    logger.debug(f"   Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    logger.debug(f"   Reserved:  {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    logger.debug(f"   Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    logger.debug(f"   Max Reserved:  {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB\n")

def Logger_main_rank(msg):
    # ✅ 让 DeepSpeed 处理 rank
    rank = int(os.environ.get("RANK", "0"))  
    if rank == 0:
        logger.info(msg)

def get_lr(it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def reduce_loss_across_gpus(loss):
    """对 loss 进行 all-reduce 并取平均，返回的是一个 detached 的 tensor，仅用于 log"""
    if dist.is_available() and dist.is_initialized():
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss /= get_world_size()
        return reduced_loss
    else:
        return loss

def compute_loss(logits, mtp_logits, targets, lambda_mtp, vocab_size):
    """
    计算 DeepSeek-V3 预训练的总损失 (L = L_main + L_MTP)

    Args:
        logits: [Bs, seq_len, vocab_size] —— 主模型输出
        targets: [Bs, seq_len] —— 已偏移好的目标（dataset 中已是 input+1）
        mtp_logits: list of [Bs, seq_len-k-1, vocab_size] —— 每个 k 的 MTP 输出
        lambda_mtp: MTP 损失的权重因子 λ

    Returns:
        总损失 (scalar tensor)
    """

    logger.debug(f"[DEBUG] logits max: {logits.max().item()}, min: {logits.min().item()}")
    
    for k, mtp_logit in enumerate(mtp_logits):
        logger.debug(f"[DEBUG] mtp_logits[{k}] max: {mtp_logit.max().item()}, min: {mtp_logit.min().item()}")
        
        if torch.isnan(mtp_logit).any():
            logger.error(f"[NaN] ❌ mtp_logits[{k}] contains NaN!")

        if torch.isinf(mtp_logit).any():
            logger.warning(f"[Inf] ⚠️ mtp_logits[{k}] contains Inf!")
    
    # ====== 主 loss ======
    main_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    logger.debug(f"[DEBUG] main_loss: {main_loss.item()}, requires_grad: {main_loss.requires_grad}, grad_fn: {main_loss.grad_fn}")

    # ====== MTP loss ======
    mtp_loss = None
    for k, mtp_logit in enumerate(mtp_logits):
        if targets.shape[1] <= k + 1:
            continue

        mtp_target = targets[:, k + 1:]
        expected_len = mtp_logit.size(1)
        mtp_target = mtp_target[:, :expected_len]

        # 检查是否有 NaN 或非法值
        if torch.isnan(mtp_logit).any():
            logger.error(f"[ERROR] NaN detected in mtp_logits[{k}]")
            continue
        if mtp_target.max() >= vocab_size:
            logger.error(f"[ERROR] mtp_target contains token >= vocab_size at k={k}")
            continue
        
        this_loss = F.cross_entropy(mtp_logit.reshape(-1, vocab_size), mtp_target.reshape(-1))
        logger.debug(f"[DEBUG] mtp_loss[{k}]: {this_loss.item()}, requires_grad: {this_loss.requires_grad}, grad_fn: {this_loss.grad_fn}")

        if mtp_loss is None:
            mtp_loss = this_loss
        else:
            mtp_loss = mtp_loss + this_loss

    if mtp_loss is not None:
        mtp_loss = (lambda_mtp / len(mtp_logits)) * mtp_loss
    else:
        mtp_loss = torch.tensor(0.0, device=main_loss.device, dtype=main_loss.dtype)
    logger.debug(f"[DEBUG] final mtp_loss: {mtp_loss.item()}, requires_grad: {mtp_loss.requires_grad}, grad_fn: {mtp_loss.grad_fn}")

    total_loss = main_loss + mtp_loss
    logger.debug(f"[DEBUG] total_loss: {total_loss.item()}, requires_grad: {total_loss.requires_grad}, grad_fn: {total_loss.grad_fn}")

    return total_loss

def train_epoch(epoch, wandb):
    epoch_start_time = time.time()
    for step, (X, Y) in enumerate(train_loader):
        step_start_time = time.time()
        X = X.to(args.device)
        Y = Y.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            #logger.debug("="*20 + " 当前张量显存信息 " + "="*20)
            #print_memory("Before forward")
            forward_start = time.time()
            logits, mtp_logits = model(X,Y)
            logger.debug(f"logits grad check: requires_grad = {logits.requires_grad} | grad_fn = {logits.grad_fn}")
            logger.debug(f"mtp_logits grad check: requires_grad = {mtp_logits[0].requires_grad} | grad_fn = {mtp_logits[0].grad_fn}")
            #print_memory("After forward")
            if torch.isnan(logits).any():
                logger.debug("🚨 [NaN DETECTED] logits contains NaN")
            else:
                logger.debug("✅ logits looks OK")
            forward_end_1 = time.time()
            
            for i, mtp in enumerate(mtp_logits):
                if torch.isnan(mtp).any():
                    logger.debug(f"🚨 [NaN DETECTED] mtp_logits[{i}] contains NaN")
                else:
                    logger.debug(f"✅ mtp_logits[{i}] looks OK")
            
            loss = compute_loss(logits, mtp_logits, Y
                                , args.lambda_mtp, lm_config.vocab_size) / args.accumulation_steps
            avg_loss = reduce_loss_across_gpus(loss)
            logger.debug(f"LOSS grad check: requires_grad = {loss.requires_grad} | grad_fn = {loss.grad_fn}")
            forward_end_2 = time.time()
                            
            print_memory("After loss")
            
        with torch.autograd.set_detect_anomaly(True):
            scaler.scale(loss).backward()
            backward_end = time.time()
            
        del logits, Y

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        
        update_end = time.time()
        
        if step % args.log_interval == 0:
            current_time = time.time()
            epoch_spend_time = current_time - epoch_start_time

            if step > 100:
                # ✅ 新增：估算剩余 epoch 时间
                avg_time_per_step = epoch_spend_time / (step + 1)
                remaining_steps = iter_per_epoch - step - 1
                estimated_remaining_time = avg_time_per_step * remaining_steps
                estimated_remaining_min = estimated_remaining_time / 60
            else:
                estimated_remaining_min = -1 #暂不估计

            # ✅ 打印进度
            Logger_main_rank(
                f"Epoch:[{epoch}/{args.epochs}]({step}/{iter_per_epoch}) "
                f"loss:{avg_loss.item() * args.accumulation_steps:.3f} "
                f"lr:{optimizer.param_groups[-1]['lr']:.7f} "
                f"epoch_RemainingTime:{estimated_remaining_min / 60:.1f}min"
            )

            # ✅ 可选：记录 wandb
            if (wandb is not None) and (os.environ.get("RANK", "0") == "0"):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_RemainingTime(min)": estimated_remaining_time / 60
                })
                
        if (step + 1) % args.save_interval == 0 and (os.environ.get("RANK", "0") == "0"):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.model_name}.pth'  # 使用用户提供的模型名称

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            Logger_main_rank(f"保存模型到 {ckp}")
            optimizer_state_path = f'{args.save_dir}/{args.model_name}_optimizer.pth'
            torch.save(optimizer.state_dict(), optimizer_state_path)
            Logger_main_rank(f"保存优化器状态到 {optimizer_state_path}")
            
            model.train()

def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = Transformer(lm_config, logger=logger)
    moe_path = '_moe' if lm_config.use_moe else ''

    def clean_state_dict(state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    # 加入恢复训练的逻辑
    checkpoint_path = f'{args.save_dir}/{args.model_name}.pth'
    if os.path.exists(checkpoint_path):
        Logger_main_rank(f"加载模型检查点 {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(clean_state_dict(state_dict), strict=True)
    else:
        Logger_main_rank(f"没有找到模型检查点，开始从头训练")

    # ✅ 注册所有你手动做了分布式的模块的参数
    def register_external_parameters(model, optimizer):
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = model.optimizer  # 拿到 ZeRO 优化器

        def try_register(param):
            try:
                deepspeed.zero.register_external_parameter(optimizer,param)
            except Exception as e:
                Logger_main_rank(f"⚠️ 注册参数失败: {e}")              

        # 注册 embedding 层
        try_register(raw_model.embed.weight)

        # 注册输出 head 层
        try_register(raw_model.head.weight)

        # 注册主模型 layers 中的 parallel linear
        for layer in raw_model.layers:
            for mod in layer.modules():
                if isinstance(mod, (ColumnParallelLinear, RowParallelLinear)):
                    try_register(mod.weight)
                    if mod.bias is not None:
                        try_register(mod.bias)
                    if hasattr(mod, 'scale') and mod.scale is not None:
                        try_register(mod.scale)

        # 注册 MTP 中的 projection 和 layers
        for proj in raw_model.mtp_projections:
            for param in proj.parameters():
                try_register(param)

        for layer in raw_model.mtp_layers:
            for mod in layer.modules():
                if isinstance(mod, (ColumnParallelLinear, RowParallelLinear)):
                    try_register(mod.weight)
                    if mod.bias is not None:
                        try_register(mod.bias)
                    if hasattr(mod, 'scale') and mod.scale is not None:
                        try_register(mod.scale)

        # 注册 MoE 模块中的 gate 和 experts
        for layer in raw_model.layers:
            if hasattr(layer.ffn, 'gate'):
                gate = layer.ffn.gate
                try_register(gate.weight)
                if gate.bias is not None:
                    try_register(gate.bias)

            if hasattr(layer.ffn, 'experts'):
                for expert in layer.ffn.experts:
                    if expert is not None:
                        for param in expert.parameters():
                            try_register(param)

    # ✅ 使用 DeepSpeed ZeRO-1 数据并行
    logger.debug("🚀 Initializing model with DeepSpeed...")
    Logger_main_rank("使用 DeepSpeed ZeRO-1")
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=args.deepspeed,
        model_parameters=model.parameters()
    )
    
    # ✅ 执行注册
    register_external_parameters(model, optimizer)
    logger.debug(f"NaN check passed on rank {get_rank()}")
    return model

# torchrun --nproc_per_node 2 pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Pretraining")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro batch size per GPU")
    parser.add_argument("--accumulation_steps", type=int, default=6, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="DeepSeek-Pretrain", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/mini-deepseekv3/data/step3/data_step3", help="Path to training data")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument("--model_name", type=str, default="minideepseekbase", help="模型名称，用于保存和加载检查点")
    # 新增关于MTP的相关参数lambda
    parser.add_argument("--lambda_mtp", type=float, default=0.5, help="MTP loss weight λ")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    lm_config = ModelArgs()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = f'{args.save_dir}/{args.model_name}.pth'

    # 动态创建DeepSpeed配置
    ds_config_path = create_ds_config()
    args.deepspeed = ds_config_path
    
    tokens_per_iter = args.micro_batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.wandb_run_name = f"minideepseek-Pretrain-Epoch-{args.epochs}-BatchSize-{args.micro_batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    if args.use_wandb and (os.environ.get("RANK", "0") == "0"):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    data_path_list = sorted([str(p) for p in Path(args.data_path).glob("*.bin")])
    logger.debug(f"🔍 Looking for .bin files in: {args.data_path}")
    logger.debug(f"📦 Found {len(data_path_list)} bin files")
    for path in data_path_list[:5]:
        logger.debug(f"  - {path}")
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)
    #train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False, # 在step4里使用shuffle了
        num_workers=args.num_workers,
        prefetch_factor=8,  # 预取 8 个 batch
        persistent_workers=True  # 让 worker 持续运行
    )

    model = init_model()
    logger.debug(f"[CHECK] head weight stats: max={model.head.weight.max().item()}, min={model.head.weight.min().item()}, mean={model.head.weight.mean().item()}")
    if hasattr(model.head, "scale") and model.head.scale is not None:
        logger.debug(f"[CHECK] head scale stats: max={model.head.scale.max().item()}, min={model.head.scale.min().item()}")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️ Parameter {name} contains NaN!")
    # torch 2.2.0版本不支持更智能的bf16自动混合精度训练，因此在设置中只能保留float16
    # 但是由于我们是基于bf16进行训练，因此本质上AMP不会被触发
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    #scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 恢复优化器状态
    optimizer_state_path = f'{args.save_dir}/{args.model_name}_optimizer.pth'
    if os.path.exists(checkpoint_path):
        optimizer_state_path = f'{args.save_dir}/{args.model_name}_optimizer.pth'
        if os.path.exists(optimizer_state_path):
            Logger_main_rank(f"加载优化器状态 {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=args.device))
        else:
            Logger_main_rank(f"没有找到优化器状态，使用新的优化器")

    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger_main_rank("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)