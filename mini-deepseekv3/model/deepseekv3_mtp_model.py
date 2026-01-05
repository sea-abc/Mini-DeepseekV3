import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List
import torch
import torch.distributed as dist
import math
import torch.nn.init as init
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PretrainedConfig
from model.kernel import act_quant, weight_dequant, fp8_gemm
from logger_utils import setup_logger
import logging

#分块量化的块尺寸
block_size = 128

#设置矩阵乘法所使用的精度
gemm_impl: Literal["bf16", "fp8"] = "bf16"
#设置注意力的计算方法
attn_impl: Literal["naive", "absorb"] = "absorb"
#设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@dataclass
class ModelArgs(PretrainedConfig):
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA self.rank for query projections.
        kv_lora_rank (int): LoRA self.rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 增加kwargs.get以兼容代码中的属性引用
        self.max_batch_size = kwargs.get("max_batch_size", 8)
        self.max_seq_len = kwargs.get("max_seq_len", 4096)
        self.dtype = kwargs.get("dtype", "bf16")
        self.vocab_size = kwargs.get("vocab_size", 128000) # 按照deepseek的tokenizer重新修改
        self.dim = kwargs.get("dim", 1024) 
        self.inter_dim = kwargs.get("inter_dim", 2048)  
        self.moe_inter_dim = kwargs.get("moe_inter_dim", 512) #MoE的inter_dim
        self.n_layers = kwargs.get("n_layers", 8) 
        self.n_dense_layers = kwargs.get("n_dense_layers", 1)
        self.n_heads = kwargs.get("n_heads", 8) 
        # MoE
        self.use_moe = kwargs.get("use_moe", 1)
        self.n_routed_experts = kwargs.get("n_routed_experts", 4) #必须是gpu的倍数 
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)  
        self.n_activated_experts = kwargs.get("n_activated_experts", 2) 
        self.n_expert_groups = kwargs.get("n_expert_groups", 1)
        self.n_limited_groups = kwargs.get("n_limited_groups", 1)
        self.score_func = kwargs.get("score_func", "softmax")
        self.route_scale = kwargs.get("route_scale", 1.0)
        # MLA
        self.q_lora_rank = kwargs.get("q_lora_rank", 0)
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 256)  
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", 64)  
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 32)  
        self.v_head_dim = kwargs.get("v_head_dim", 64)
        # YARN
        self.original_seq_len = kwargs.get("original_seq_len", 4096)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_factor = kwargs.get("rope_factor", 40)
        self.beta_fast = kwargs.get("beta_fast", 32)
        self.beta_slow = kwargs.get("beta_slow", 1)
        self.mscale = kwargs.get("mscale", 1.0)
        # MTP
        self.n_mtp_depths = kwargs.get("n_mtp_depths", 2)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 增加kwargs.get以兼容代码中的属性引用
        self.max_batch_size = kwargs.get("max_batch_size", 8)
        self.max_seq_len = kwargs.get("max_seq_len", 2048)
        self.dtype = kwargs.get("dtype", "bf16")
        self.vocab_size = kwargs.get("vocab_size", 128000) # 按照deepseek的tokenizer重新修改
        self.dim = kwargs.get("dim", 512) 
        self.inter_dim = kwargs.get("inter_dim", 1024)  
        self.moe_inter_dim = kwargs.get("moe_inter_dim", 256) #MoE的inter_dim
        self.n_layers = kwargs.get("n_layers", 4) 
        self.n_dense_layers = kwargs.get("n_dense_layers", 1)
        self.n_heads = kwargs.get("n_heads", 8)
        # MoE
        self.use_moe = kwargs.get("use_moe", 1)
        self.n_routed_experts = kwargs.get("n_routed_experts", 4) #必须是gpu的倍数 
        self.n_shared_experts = kwargs.get("n_shared_experts", 1)  
        self.n_activated_experts = kwargs.get("n_activated_experts", 2) 
        self.n_expert_groups = kwargs.get("n_expert_groups", 1)
        self.n_limited_groups = kwargs.get("n_limited_groups", 1)
        self.score_func = kwargs.get("score_func", "softmax")
        self.route_scale = kwargs.get("route_scale", 1.0)
        # MLA
        self.q_lora_rank = kwargs.get("q_lora_rank", 0)
        self.kv_lora_rank = kwargs.get("kv_lora_rank", 64)  
        self.qk_nope_head_dim = kwargs.get("qk_nope_head_dim", 64)  
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim", 32)  
        self.v_head_dim = kwargs.get("v_head_dim", 64)
        # YARN
        self.original_seq_len = kwargs.get("original_seq_len", 2048)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_factor = kwargs.get("rope_factor", 40)
        self.beta_fast = kwargs.get("beta_fast", 32)
        self.beta_slow = kwargs.get("beta_slow", 1)
        self.mscale = kwargs.get("mscale", 1.0)
        # MTP
        self.n_mtp_depths = kwargs.get("n_mtp_depths", 1)

def get_rank_world_size():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        return 0, 1

def get_rank():
    return get_rank_world_size()[0]

def get_world_size():
    return get_rank_world_size()[1]

def get_logger(logger=None):
    return logger or logging.getLogger("fallback_logger")

def print_memory(tag=""):
    print(f"\n📌 CUDA Memory Status {tag}:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"   Reserved:  {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    print(f"   Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"   Max Reserved:  {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB\n")

class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        gathered = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, input)
        ctx.rank = torch.distributed.get_rank()
        ctx.world_size = torch.distributed.get_world_size()
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.chunk(ctx.world_size, dim=-1)[ctx.rank]

class AllReduceWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        torch.distributed.all_reduce(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，同样 all_reduce，保持一致性
        torch.distributed.all_reduce(grad_output)
        return grad_output

#===================
# 1. 遵循张量并行的Embedding层
#===================

class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, vocab_size: int, dim: int, logger=None):
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.vocab_size = vocab_size
        self.dim = dim

        #将需要向量化的token分到不同GPU上
        assert self.vocab_size % self.world_size == 0, f"Vocabulary size must be divisible by world size (world_size={self.world_size})"
        self.part_vocab_size = (self.vocab_size // self.world_size)

        #划分为 [0,500],[500,1000],[1000,1500]这样的区间
        self.vocab_start_idx = self.rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size

        #embedding层的权重等于一个GPU上的vocab_size乘以相应的权重
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=self.dtype or Linear.dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.logger.debug(f"[DEBUG] ParallelEmbedding weight initialized: max={self.weight.max().item()}, min={self.weight.min().item()}, mean={self.weight.mean().item()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.max().item() < self.vocab_size, \
            f"Token ID {x.max().item()} 超过 vocab_size {self.vocab_size}"
    
        if self.world_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
    
            # 初始化输出
            y = torch.zeros(x.shape + (self.dim,), device=x.device, dtype=self.weight.dtype)
    
            # 只处理 mask 内合法的局部 index
            y[mask] = F.embedding(x[mask] - self.vocab_start_idx, self.weight)
    
            dist.all_reduce(y)
        else:
            y = F.embedding(x, self.weight)
        return y

#===================
# 2. 支持混合精度 + 细粒度量化操作的线性函数与线性层
#===================

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    自定义的、能够实现FP8、BF16、FP16、FP32各种精度之间的转化和GEMM的线性运算函数

    这个函数的输入与nn.Linear层不同，函数需的是：
    - 需要被线性变换的数据X，以及
    - 线性变换所使用的权重weights，此时weights的结构就等同于
    
    补充：BF16和FP16的区别在于、都是16位数，但BF16是8个指数7个尾数（8E7M），FP16是5E10M，指数越大可表示的数值范围越大、尾数越大越精确。因此，BF16范围更大适用于梯度计算、激活值存储等场景、可以稳定训练；FP16更加精确、适用于权重存储等存储场景，更适合保存信息。

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() > 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    # 如果字节大于1（精度大于fp8），则应该是fp16或者fp32、直接使用非量化计算
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    # 如果GEMM精度是bf16，那要将权重解量化为bf16
    # 当我们默认训练使用BF16的时候，不会进入elif
    # 下面的elif只能适应推理流程、不能用于训练
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    # 如果element_size不大于1，同时gemm_impl不是bf16（也就是fp8）
    # 就说明是支持fp8计算的，则无论什么数据输入都量化成fp8
    # 使用fp8精度进行fp8的GEMM
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

class Linear(nn.Module):
    """
    自定义的、能够支持FP8精度的细粒度量化（主要是创建缩放因子进行缩放）的线性层。

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, logger=None):
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=self.dtype or Linear.dtype or Linear.dtype))

        # 权重初始化：kaiming_uniform
        with torch.no_grad():
            tmp_weight = torch.empty_like(self.weight, dtype=torch.float32)
            init.xavier_uniform_(tmp_weight, gain=1.0 / math.sqrt(2))
            self.weight.copy_(tmp_weight.to(self.weight.dtype))
            if torch.isnan(self.weight).any():
                self.logger.warning(f"[WARNING] Linear weight contains NaN after kaiming init! rank={self.rank}")
        
        # 如果字节 == 1，即weight精度是FP8，则执行归一化
        # 将FP8权重缩放到FP8的动态范围内、则会需要对每一个“块”构建缩放因子
        # 其中block_size是我们规定的“块”的大小，用下面的公式来确认缩放因子的具体数量
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            #如果不是fp8，则直接将scale设置为None来表示该参数不存在
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features),dtype=self.dtype or Linear.dtype)
            with torch.no_grad():
                init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)

#===================
# 3. 支持分布式并行（张量并行）的线性层
#===================

class ColumnParallelLinear(Linear):
    """
    具有列并行功能的线性层，将输出的features分布到不同的进程（GPU）上完成。

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
        gather_output (bool): 是否在前向中聚合输出（保持梯度计算图）
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None, gather_output: bool = False, logger=None):
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.gather_output = gather_output

        assert out_features % self.world_size == 0, \
            f"Output features must be divisible by world size (world_size={self.world_size})"
        self.part_out_features = out_features // self.world_size

        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor. Shape: [B, T, in_features]

        Returns:
            torch.Tensor: Output tensor. If gather_output=True, shape = [B, T, out_features]
                          Otherwise, shape = [B, T, out_features / world_size]
        """
        y = F.linear(x, self.weight, self.bias)  # y: [B, T, out_features / world_size]
        #self.logger.debug(f"[DEBUG] ColumnParallelLinear rank {self.rank} output shape: {y.shape}")

        if self.gather_output and self.world_size > 1:
            y = AllGatherWithGrad.apply(y)
        return y

class RowParallelLinear(Linear):
    """
    具有行并行功能的线性层，将输入的features分布到不同的进程（GPU）上完成。

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, logger=None):
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        assert in_features % self.world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // self.world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        #print("Row",self.rank, self.weight.shape)
        y = linear(x, self.weight)

        #如果是多线程运行，还需要进行all_reduce聚合，并加上偏置
        if self.world_size > 1:
            y = AllReduceWithGrad.apply(y)
        if self.bias is not None:
            y += self.bias
        return y

#===================
# 4. 并行化的RMS Norm 根均方归一化
#===================

class RMSNorm(nn.Module):
    """
    huggngface-style RMSNorm
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, dim, eps=1e-5, logger=None):
        super().__init__()
        self.logger = get_logger(logger)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim,dtype=self.dtype or Linear.dtype))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(norm + self.eps)
        return self.weight * x

#===================
# 5. 旋转位置编码
# deepseek从序列长度、频率生成、平滑校正等多个角度改进了旋转位置编码
# 其核心与llama所使用的旋转位置编码相同
#===================

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    对旋转位置变啊中所需要的指数权重进行预计算，方便后续使用。

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

#===================
# 6. MLA 潜在注意力机制
#===================

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): self.rank for low-rank query projection.
        kv_lora_rank (int): self.rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, args: ModelArgs, logger=None):
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // self.world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim, logger=self.logger)
            self.logger.debug("[DEBUG] wq weight NaN: %s", torch.isnan(self.wq.weight).any())
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank, logger=self.logger)
            self.q_norm = RMSNorm(self.q_lora_rank, logger=self.logger)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim, logger=self.logger)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, logger=self.logger)
        self.kv_norm = RMSNorm(self.kv_lora_rank, logger=self.logger)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), logger=self.logger)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim, logger=self.logger)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
    
        # 1. Q projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        self.logger.debug("[DEBUG] q before nope NaN: %s, inf: %s", torch.isnan(q).any(), torch.isinf(q).any())
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
    
        # 2. KV projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    
        if self.training:
            # === Training Mode ===
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

            if torch.isnan(q).any():
                logger.error("[NaN] ❌ Q input contains NaN before einsum")
            scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
            if mask is not None:
                scores += mask.unsqueeze(1)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
            x = torch.einsum("bsht,bthd->bshd", scores, v)
    
        else:
            # === Inference Mode ===
            if attn_impl == "naive":
                q = torch.cat([q_nope, q_pe], dim=-1)
                kv = self.wkv_b(self.kv_norm(kv))
                kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
                self.k_cache[:bsz, start_pos:end_pos] = k.detach()
                self.v_cache[:bsz, start_pos:end_pos] = v.detach()
    
                scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
            else:
                wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
                wkv_b = wkv_b.view(self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
                q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
                self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv).detach()
                self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2).detach()
    
                scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
    
            if mask is not None:
                scores += mask.unsqueeze(1)
            scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
    
            if attn_impl == "naive":
                x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
            else:
                x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
                x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
    
        x = self.wo(x.flatten(2))
        return x

'''    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        self.logger.debug("[DEBUG] q before nope NaN: %s, inf: %s", torch.isnan(q).any(), torch.isinf(q).any())
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 该行代码在FP8状态下会斩断计算图，BF16状态下安全
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            #wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            expected_shape = self.n_local_heads * (self.qk_nope_head_dim + self.v_head_dim)
            assert wkv_b.shape[0] == expected_shape, f"wkv_b.shape[0]={wkv_b.shape[0]} does not match expected {expected_shape}"
            wkv_b = wkv_b.view(self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            self.logger.debug("[DEBUG] q_nope NaN: %s max: %f", torch.isnan(q_nope).any(), torch.isinf(q_nope).any())
            self.logger.debug("[DEBUG] q_pe NaN: %s max: %f", torch.isnan(q_pe).any(), torch.isinf(q_pe).any())
            self.logger.debug("[DEBUG] kv NaN: %s max: %f", torch.isnan(self.kv_cache).any(),  torch.isinf(self.kv_cache).any())
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        self.logger.debug("[DEBUG] attn score NaN: %s max: %f", torch.isnan(scores).any(), scores.max().item())
        if mask is not None:
            scores += mask.unsqueeze(1)
        self.logger.debug("[DEBUG] attn score after mask NaN: %s max: %f", torch.isnan(scores).any(), scores.max().item())
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        self.logger.debug("[DEBUG] attn score after softmax NaN: %s max: %f", torch.isnan(scores).any(), scores.max().item())

        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x
'''

#===================
# 7. DeepSeekMOE
#===================

class MLP(nn.Module):
    """
    支持张量并行的SwiGLU FFN前馈神经网络，并行方式为切分features，用于支持共享专家的构建。
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, dim: int, inter_dim: int, logger=None):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.w1 = ColumnParallelLinear(dim, inter_dim, logger=self.logger)
        self.w2 = RowParallelLinear(inter_dim, dim, logger=self.logger)
        self.w3 = ColumnParallelLinear(dim, inter_dim, logger=self.logger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, args: ModelArgs, logger=None):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=self.dtype or Linear.dtype))
        with torch.no_grad():
            tmp = torch.empty_like(self.weight, dtype=torch.float32)
            init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight.copy_(tmp.to(self.dtype or Linear.dtype))
            if torch.isnan(self.weight).any():
                self.logger.warning(f"[WARNING] Gate weight contains NaN! rank={self.rank}")
        
        if self.dim == 7168:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=self.dtype or Linear.dtype))
            with torch.no_grad():
                init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.to(torch.float32).softmax(dim=-1).to(x.dtype)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        self.logger.debug(f"[DEBUG] Gate weights grad_fn: {weights.grad_fn}, requires_grad: {weights.requires_grad}")
        return weights.type_as(x), indices

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, dim: int, inter_dim: int, logger=None):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.w1 = Linear(dim, inter_dim, logger=self.logger)
        self.w2 = Linear(inter_dim, dim, logger=self.logger)
        self.w3 = Linear(dim, inter_dim, logger=self.logger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn

    def __init__(self, args: ModelArgs, logger=None):
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.dim = args.dim

        assert args.n_routed_experts % self.world_size == 0, \
            f"Number of experts must be divisible by world size (world_size={self.world_size})"

        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // self.world_size
        self.n_activated_experts = args.n_activated_experts

        self.experts_start_idx = self.rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(args, logger=self.logger)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, logger=self.logger)
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim, logger=self.logger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_train(x)
        else:
            with torch.no_grad():
                return self._forward_infer(x)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)  # shape: [B*T, topk]
        y = torch.zeros_like(x)

        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            x_selected = x[idx]
            self.logger.debug(f"[Expert {i}] x_selected stats: max={x_selected.max().item():.4f}, min={x_selected.min().item():.4f}")
            expert_out = expert(x_selected)
            if torch.isnan(expert_out).any():
                self.logger.error("[NaN] ❌ expert_out contains NaN before index_add")
            self.logger.debug(f"[Expert {i}] weights stats: max={weights[idx, top].max().item():.4f}, min={weights[idx, top].min().item():.4f}")
            y = y.index_add(0, idx, expert_out * weights[idx, top].unsqueeze(-1))

        z = self.shared_experts(x)

        if self.world_size > 1:
            dist.all_reduce(y)

        return (y + z).view(shape)

    def _forward_infer(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)

        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1)
        total_experts = self.n_routed_experts

        counts = torch.bincount(flat_indices, minlength=total_experts)
        tokens_per_expert = counts.tolist()
        idxs = flat_indices.argsort()
        sorted_x = x[idxs]

        local_tokens = []
        for i in range(self.experts_start_idx, self.experts_end_idx):
            num_tokens = tokens_per_expert[i]
            if num_tokens == 0:
                continue
            start_idx = sum(tokens_per_expert[:i])
            end_idx = start_idx + num_tokens
            tokens_for_expert = sorted_x[start_idx:end_idx]
            expert_out = self.experts[i](tokens_for_expert)
            local_tokens.append(expert_out)

        local_output = torch.cat(local_tokens, dim=0) if local_tokens else sorted_x.new_empty(0)

        output_gather = [torch.empty_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_gather, local_output)
        gathered_output = torch.cat(output_gather, dim=0)

        gathered_x = torch.empty_like(gathered_output)
        gathered_x[idxs] = gathered_output

        final_out = (
            gathered_x.view(*indices.shape, -1)
            .type(weights.dtype)
            .mul_(weights.unsqueeze(-1))
            .sum(dim=1)
            .type(x.dtype)
        )

        z = self.shared_experts(x)

        if self.world_size > 1:
            dist.all_reduce(final_out)

        return (final_out + z).view(shape)

'''
class MoE(nn.Module):

    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, args: ModelArgs, logger=None):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.dim = args.dim
        assert args.n_routed_experts % self.world_size == 0, f"Number of experts must be divisible by world size (world_size={self.world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // self.world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = self.rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args, logger=self.logger)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, logger=self.logger) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim, logger=self.logger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            #y[idx] += expert(x[idx]) * weights[idx, top, None]
            y = y.index_add(0, idx, expert(x[idx]) * weights[idx, top, None])
        z = self.shared_experts(x)
        if self.world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)
'''

#===================
# 8. Decoder layers
#===================

class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    dtype = torch.bfloat16 if gemm_impl == "bf16" else torch.float8_e4m3fn
    
    def __init__(self, layer_id: int, args: ModelArgs, logger=None):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.logger = get_logger(logger)
        self.rank = get_rank()
        self.layer_id = layer_id
        self.world_size = get_world_size()
        self.attn = MLA(args, logger=self.logger)
        self.ffn = MLP(args.dim, args.inter_dim, logger=self.logger) if layer_id < args.n_dense_layers else MoE(args, logger=self.logger)
        self.attn_norm = RMSNorm(args.dim, logger=self.logger)
        self.ffn_norm = RMSNorm(args.dim, logger=self.logger)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        self.logger.debug(f"[Attention Layer {self.layer_id}]")
        self.logger.debug(f"[DEBUG] input contains NaN: {torch.isnan(x).any()}")
        fn = self.attn_norm(x)
        self.logger.debug(f"[DEBUG] attn_norm NaN: {torch.isnan(fn).any()}")
        fn = self.attn(fn, start_pos, freqs_cis, mask)
        self.logger.debug(f"[DEBUG] attn NaN: {torch.isnan(fn).any()}")
        x = x + fn
        self.logger.debug(f"[DEBUG] first_res NaN: {torch.isnan(x).any()}")
        fn2 = self.ffn_norm(x)
        self.logger.debug(f"[DEBUG] ffn_norm NaN: {torch.isnan(fn2).any()}")
        fn2 = self.ffn(fn2)
        self.logger.debug(f"[DEBUG] ffn NaN: {torch.isnan(fn2).any()}")
        x = x + fn2
        self.logger.debug(f"[DEBUG] second_res NaN: {torch.isnan(x).any()}")
        return x

#===================
# 9. 完整的DeepSeek
#===================

class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """    
    def __init__(self, args: ModelArgs, logger=None):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        self.rank = get_rank()
        self.world_size = get_world_size()
        Linear.dtype = torch.bfloat16
        #Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.logger = get_logger(logger)
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim, logger=self.logger) #在MTP模块中共享
        self.layers = nn.ModuleList([Block(i, args, logger=self.logger) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, logger=self.logger)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype(), gather_output=True, logger=self.logger)
        self.logger.debug(f"[CHECK] head weight stats: max={self.head.weight.max().item()}, min={self.head.weight.min().item()}, mean={self.head.weight.mean().item()}")
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # ✅ 构建MTPlayers所所需的组件
        # 其中embedding与output层是与之前的trans共享
        # 定义预测未来token的数量
        self.n_mtp_depths = args.n_mtp_depths
        self.mtp_rms_norm1 = RMSNorm(args.dim)  # 归一化 h_i^{k-1}
        self.mtp_rms_norm2 = RMSNorm(args.dim)  # 归一化 Emb(t_{i+k})
        # 按照未来token数量确定投影层的数量
        self.mtp_projections = nn.ModuleList([
            Linear(args.dim*2, args.dim, logger=self.logger) for _ in range(self.n_mtp_depths) 
        ])
        # 按照未来token数量确定要使用的decoder layer数量
        self.mtp_layers = nn.ModuleList([
            Block(args.n_layers + i, args, logger=self.logger) for i in range(self.n_mtp_depths) 
        ])

        def register_nan_hook(module, name, logger):
            def hook(module, grad_input, grad_output):
                logger.debug(f"[HOOK ✅] {name} backward triggered")
                for i, g in enumerate(grad_output):
                    if g is not None and torch.isnan(g).any():
                        logger.error(f"🚨 NaN detected in {name} grad_output[{i}]")
        
            def forward_hook(module, input, output):
                logger.debug(f"[HOOK 🚀] {name} forward triggered")
        
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(hook)
        
        for i, layer in enumerate(self.layers):
            register_nan_hook(layer.ffn, f"main_ffn_{i}", self.logger)
            register_nan_hook(layer.attn.wo, f"main_attn_wo_{i}", self.logger)
        
        for i, layer in enumerate(self.mtp_layers):
            register_nan_hook(layer.ffn, f"mtp_ffn_{i}", self.logger)
            register_nan_hook(layer.attn.wo, f"mtp_attn_wo_{i}", self.logger)
        
        for i, proj in enumerate(self.mtp_projections):
            register_nan_hook(proj, f"mtp_proj_{i}", self.logger)
        
        register_nan_hook(self.head, "head", self.logger)
    
    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        
        self.logger.debug(f"[DEBUG] input check: max={tokens.max().item()}, min={tokens.min().item()}, mean={tokens.float().mean().item()}")
        
        #print_memory("🔹 Before embedding")
        h = self.embed(tokens)
        self.logger.debug(f"[DEBUG] embed weight stats: max={self.embed.weight.max().item()}, min={self.embed.weight.min().item()}, mean={self.embed.weight.mean().item()}")
        self.logger.debug(f"[DEBUG] logits embd_h NaN: {torch.isnan(h).any()}")
        self.logger.debug(f"[DEBUG] h after emb: max={h.max().item()}, min={h.min().item()}, mean={h.mean().item()}")
        #print_memory("🔹 After embedding")
        
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=tokens.device), diagonal=1)
        
        # ✅ Main model计算
        # Decoderlayers输出结果
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis, mask)
            self.logger.debug(f"[DEBUG] logits layer_mask_h NaN: {torch.isnan(h).any()}")
            self.logger.debug(f"[DEBUG] h in {i} layer: max={h.max().item()}, min={h.min().item()}, mean={h.mean().item()}")
            #print_memory(f"🔹 After layer {i}")

        self.logger.debug(f"[DEBUG] h before norm: max={h.max().item()}, min={h.min().item()}, mean={h.mean().item()}")
        # 获取模型输出，结构为[bs, seq_len, D]
        h = self.norm(h)  # [B, T, D]
        self.logger.debug(f"[DEBUG] logits norm_h NaN: {torch.isnan(h).any()}")
        self.logger.debug(f"[DEBUG] h before head: max={h.max().item()}, min={h.min().item()}, mean={h.mean().item()}")
        logits = self.head(h)  # [B, T, V]
        self.logger.debug(f"[DEBUG] logits head_NaN: {torch.isnan(h).any()}")
        #print_memory("🔹 After main model")
        
        mtp_logits = []
        for k in range(self.n_mtp_depths):
            if start_pos + k + 1 >= tokens.size(1):
                break
            h_k = h[:, :-k-1]  # [B, T-k-1, D]
            self.logger.debug(f"[DEBUG] mtp_index_h NaN: {torch.isnan(h_k).any()}")
            emb_k = self.embed(tokens[:, k+1:])  # [B, T-k-1, D]
            self.logger.debug(f"[DEBUG] mtp_emb_k NaN: {torch.isnan(emb_k).any()}")
            
            norm_h = self.mtp_rms_norm1(h_k)
            self.logger.debug(f"[DEBUG] mtp_norm_h NaN: {torch.isnan(norm_h).any()}")
            norm_emb = self.mtp_rms_norm2(emb_k)
            self.logger.debug(f"[DEBUG] mtp_norm_k NaN: {torch.isnan(norm_emb).any()}")
            combined_h = torch.cat([norm_h, norm_emb], dim=-1)  # [B, T-k-1, 2D]
            self.logger.debug(f"[DEBUG] mtp_combined_h NaN: {torch.isnan(combined_h).any()}")
            
            mtp_h = self.mtp_projections[k](combined_h)
            mtp_h = torch.clamp(mtp_h, min=-20.0, max=20.0)
            self.logger.debug(f"[DEBUG] mtp_h NaN check after projection: {torch.isnan(mtp_h).any()}")
            self.logger.debug(f"[DEBUG] mtp_h before head[{k}]: max={mtp_h.max().item()}, min={mtp_h.min().item()}, mean={mtp_h.mean().item()}")
            freqs_cis_k = self.freqs_cis[:mtp_h.size(1)]
            mask_k = torch.triu(torch.ones(mtp_h.size(1), mtp_h.size(1), dtype=torch.bool, device=tokens.device), diagonal=1)
            
            mtp_h = self.mtp_layers[k](mtp_h, 0, freqs_cis_k, mask_k)
            self.logger.debug(f"[DEBUG] mtp_h NaN check after layer: {torch.isnan(mtp_h).any()}")
            #print_memory(f"🔸 After MTP layer {k} - before head")
            mtp_logit = self.head(mtp_h)  # [B, T-k-1, V]
            #print_memory(f"🔸 After MTP layer {k} - after head")
            self.logger.debug(f"[DEBUG] mtp_logits[{k}] stats: max={mtp_logit.max().item()}, min={mtp_logit.min().item()}, mean={mtp_logit.mean().item()}")
            if torch.isinf(mtp_logit).any():
                self.logger.warning(f"[Inf] ⚠️ mtp_logits[{k}] contains Inf! Saving inputs for debug...")
                torch.save({
                    "mtp_h": mtp_h.detach().cpu(),
                    "head_weight": self.head.weight.detach().cpu(),
                    "head_bias": getattr(self.head, "bias", None).detach().cpu() if hasattr(self.head, "bias") else None
                }, f"debug_mtp_inf_k{k}.pt")
                
            mtp_logits.append(mtp_logit)
        
        return logits, mtp_logits

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())