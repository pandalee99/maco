#!/usr/bin/env python3
"""
Benchmark: Compare PyTorch baseline, vLLM, and MACO performance.

This script benchmarks:
1. Pure PyTorch inference (baseline)
2. vLLM inference
3. MACO with compute-comm overlap (multi-GPU)

Usage:
    # Single GPU baseline + vLLM
    CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_inference.py --mode baseline
    CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_inference.py --mode vllm

    # Multi-GPU MACO
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 benchmarks/benchmark_inference.py --mode maco
"""

import argparse
import time
import os
import sys

import torch
import torch.nn as nn

# Add maco to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_pytorch_baseline(model_path: str, num_iterations: int = 10):
    """Benchmark pure PyTorch inference."""
    print("\n" + "="*60)
    print("PyTorch Baseline Benchmark")
    print("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Warmup
    print("Warming up...")
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    latencies = []

    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        latency = (end - start) * 1000  # ms
        latencies.append(latency)

        if i == 0:
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Sample output: {generated[:100]}...")

    # Results
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\nResults (32 new tokens):")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Throughput: {32 * 1000 / avg_latency:.2f} tokens/sec")

    return {
        "method": "pytorch_baseline",
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_tokens_sec": 32 * 1000 / avg_latency,
    }


def benchmark_vllm(model_path: str, num_iterations: int = 10):
    """Benchmark vLLM inference."""
    print("\n" + "="*60)
    print("vLLM Benchmark")
    print("="*60)

    from vllm import LLM, SamplingParams

    # Initialize vLLM
    print(f"Loading model with vLLM from {model_path}...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
    )

    prompt = "The future of artificial intelligence is"

    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = llm.generate([prompt], sampling_params)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    latencies = []

    for i in range(num_iterations):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        end = time.perf_counter()

        latency = (end - start) * 1000  # ms
        latencies.append(latency)

        if i == 0:
            print(f"Sample output: {outputs[0].outputs[0].text[:100]}...")

    # Results
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"\nResults (32 new tokens):")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Min latency: {min_latency:.2f} ms")
    print(f"  Max latency: {max_latency:.2f} ms")
    print(f"  Throughput: {32 * 1000 / avg_latency:.2f} tokens/sec")

    return {
        "method": "vllm",
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_tokens_sec": 32 * 1000 / avg_latency,
    }


def benchmark_maco_overlap(model_path: str, num_iterations: int = 10):
    """
    Benchmark MACO with compute-communication overlap.

    This simulates tensor parallel inference with overlapped allreduce.
    """
    print("\n" + "="*60)
    print("MACO Compute-Comm Overlap Benchmark")
    print("="*60)

    import torch.distributed as dist
    import maco

    # Initialize distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size: {world_size}, Rank: {rank}, Device: {device}")

    # Initialize MACO
    maco.init(backend="nccl", rank=rank, world_size=world_size)

    # Load model (each rank loads full model for simplicity)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if rank == 0:
        print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Get overlap manager
    overlap_manager = maco.get_overlap_manager()

    # Warmup
    if rank == 0:
        print("Warming up...")

    with torch.no_grad():
        for _ in range(3):
            outputs = model(**inputs)
            if world_size > 1:
                # Simulate TP allreduce
                dist.all_reduce(outputs.logits)

    torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()

    # Benchmark with overlap
    if rank == 0:
        print(f"Running {num_iterations} iterations with compute-comm overlap...")

    latencies = []
    overlap_latencies = []

    for i in range(num_iterations):
        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()

        # Sequential version (for comparison)
        start_seq = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.clone()
            if world_size > 1:
                dist.all_reduce(logits)
        torch.cuda.synchronize()
        seq_time = (time.perf_counter() - start_seq) * 1000

        # Overlapped version
        if world_size > 1:
            dist.barrier()

        start_overlap = time.perf_counter()
        with torch.no_grad():
            # First forward pass produces logits
            outputs1 = model(**inputs)
            logits_to_reduce = outputs1.logits.clone()

            # Overlap: allreduce logits while doing another forward
            # (In real TP, this would be overlapping layer N's allreduce with layer N+1's compute)
            def compute_fn():
                return model(**inputs)

            if world_size > 1 and overlap_manager is not None:
                _, logits_reduced = overlap_manager.overlap_compute_and_comm(
                    compute_fn,
                    logits_to_reduce,
                    comm_op="allreduce"
                )
            else:
                compute_fn()
                if world_size > 1:
                    dist.all_reduce(logits_to_reduce)
                logits_reduced = logits_to_reduce

        torch.cuda.synchronize()
        overlap_time = (time.perf_counter() - start_overlap) * 1000

        latencies.append(seq_time)
        overlap_latencies.append(overlap_time)

    # Gather results
    if rank == 0:
        avg_seq = sum(latencies) / len(latencies)
        avg_overlap = sum(overlap_latencies) / len(overlap_latencies)

        print(f"\nResults:")
        print(f"  Sequential avg latency: {avg_seq:.2f} ms")
        print(f"  Overlapped avg latency: {avg_overlap:.2f} ms")
        print(f"  Speedup from overlap: {avg_seq / avg_overlap:.2f}x")

    # Cleanup
    maco.cleanup()

    if rank == 0:
        return {
            "method": "maco_overlap",
            "world_size": world_size,
            "avg_sequential_ms": sum(latencies) / len(latencies),
            "avg_overlap_ms": sum(overlap_latencies) / len(overlap_latencies),
        }
    return None


def benchmark_comm_only(num_iterations: int = 100):
    """
    Benchmark communication patterns only.

    This measures the overhead of different communication strategies.
    """
    print("\n" + "="*60)
    print("Communication-Only Benchmark")
    print("="*60)

    import torch.distributed as dist
    import maco

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size < 2:
        print("This benchmark requires at least 2 GPUs")
        return None

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size: {world_size}, Device: {device}")

    # Initialize MACO
    maco.init(backend="nccl", rank=rank, world_size=world_size)

    # Test different tensor sizes
    sizes = [
        (1024, 1024),      # 4MB
        (2048, 2048),      # 16MB
        (4096, 4096),      # 64MB
    ]

    results = []

    for size in sizes:
        tensor = torch.randn(size, dtype=torch.bfloat16, device=device)
        tensor_size_mb = tensor.numel() * 2 / (1024 * 1024)

        if rank == 0:
            print(f"\nTensor size: {size} ({tensor_size_mb:.1f} MB)")

        # Warmup
        for _ in range(5):
            t = tensor.clone()
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dist.barrier()

        # Sequential allreduce
        seq_times = []
        for _ in range(num_iterations):
            t = tensor.clone()
            torch.cuda.synchronize()
            dist.barrier()

            start = time.perf_counter()
            dist.all_reduce(t)
            torch.cuda.synchronize()
            end = time.perf_counter()

            seq_times.append((end - start) * 1000)

        avg_seq = sum(seq_times) / len(seq_times)

        # Overlapped allreduce + compute
        overlap_manager = maco.get_overlap_manager()

        overlap_times = []
        for _ in range(num_iterations):
            t = tensor.clone()
            torch.cuda.synchronize()
            dist.barrier()

            def dummy_compute():
                # Simulate some compute work
                x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
                return torch.matmul(x, x.T)

            start = time.perf_counter()
            if overlap_manager is not None:
                _, _ = overlap_manager.overlap_compute_and_comm(
                    dummy_compute, t, "allreduce"
                )
            else:
                dist.all_reduce(t)
                dummy_compute()
            torch.cuda.synchronize()
            end = time.perf_counter()

            overlap_times.append((end - start) * 1000)

        avg_overlap = sum(overlap_times) / len(overlap_times)

        if rank == 0:
            print(f"  Sequential allreduce: {avg_seq:.3f} ms")
            print(f"  Overlapped (allreduce + compute): {avg_overlap:.3f} ms")
            print(f"  Bandwidth: {tensor_size_mb * 2 / (avg_seq / 1000):.1f} GB/s")

            results.append({
                "size": size,
                "size_mb": tensor_size_mb,
                "sequential_ms": avg_seq,
                "overlap_ms": avg_overlap,
            })

    maco.cleanup()

    if rank == 0:
        return results
    return None


def main():
    parser = argparse.ArgumentParser(description="MACO Benchmark")
    parser.add_argument(
        "--mode",
        choices=["baseline", "vllm", "maco", "comm"],
        default="baseline",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--model",
        default="/vllm-workspace/Qwen3-0.6B-Base/",
        help="Model path"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations"
    )

    args = parser.parse_args()

    results = None

    if args.mode == "baseline":
        results = benchmark_pytorch_baseline(args.model, args.iterations)
    elif args.mode == "vllm":
        results = benchmark_vllm(args.model, args.iterations)
    elif args.mode == "maco":
        results = benchmark_maco_overlap(args.model, args.iterations)
    elif args.mode == "comm":
        results = benchmark_comm_only(args.iterations)

    if results:
        print("\n" + "="*60)
        print("Final Results:")
        print("="*60)
        for k, v in results.items() if isinstance(results, dict) else enumerate(results):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
