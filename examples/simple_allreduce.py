"""
Simple AllReduce Example

演示 MACO 计算-通信重叠的基本用法。

运行方式:
    # 单 GPU（功能验证）
    python examples/simple_allreduce.py

    # 多 GPU（实际重叠）
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 examples/simple_allreduce.py
"""

import torch
import maco


def main():
    # 初始化 MACO（自动检测分布式环境）
    maco.init(backend="nccl")

    rank = maco.get_rank()
    world_size = maco.get_world_size()
    print(f"[Rank {rank}] World Size: {world_size}")

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # 创建张量
    x = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)

    print(f"[Rank {rank}] 张量准备完成，设备: {device}")

    # ========================================
    # 核心 API：计算-通信重叠
    # ========================================
    # 在执行 AllReduce 的同时，并行执行矩阵乘法
    def compute_fn():
        return torch.matmul(a, b)

    tensor_to_reduce = x.clone()

    print(f"[Rank {rank}] 开始计算-通信重叠...")
    compute_result, reduced_tensor = maco.overlap_compute_and_comm(
        compute_fn=compute_fn,
        comm_tensor=tensor_to_reduce,
        comm_op="allreduce"
    )

    print(f"[Rank {rank}] 计算结果 shape: {compute_result.shape}")
    print(f"[Rank {rank}] 通信结果 shape: {reduced_tensor.shape}")

    # 同步并验证
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\n[Rank {rank}] MACO 示例完成!")

    # 清理
    maco.cleanup()


if __name__ == "__main__":
    main()
