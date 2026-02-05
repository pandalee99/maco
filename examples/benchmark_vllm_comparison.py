#!/usr/bin/env python3
"""
MACO vs vLLM TP=2 性能对比

对比 MACO TaskGraph 与 vLLM tensor parallel 的性能：
1. 单卡推理基线
2. vLLM TP=2 推理
3. MACO TaskGraph MLP 执行

运行方式:
    cd /mini_mirage/maco

    # 单卡测试
    CUDA_VISIBLE_DEVICES=1 python3 examples/benchmark_vllm_comparison.py --mode single

    # vLLM TP=2 测试
    CUDA_VISIBLE_DEVICES=1,2 python3 examples/benchmark_vllm_comparison.py --mode vllm

    # MACO 测试
    CUDA_VISIBLE_DEVICES=1 python3 examples/benchmark_vllm_comparison.py --mode maco

    # 全部测试
    CUDA_VISIBLE_DEVICES=1,2 python3 examples/benchmark_vllm_comparison.py --mode all
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def benchmark_single_gpu(model_path: str, num_tokens: int = 20):
    """单卡推理基线"""
    print("\n" + "=" * 60)
    print("Benchmark: Single GPU Inference (transformers)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 5
    for _ in range(iterations):
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / iterations * 1000

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Generation time: {elapsed:.2f} ms")
    print(f"Tokens/sec: {tokens_generated / (elapsed / 1000):.1f}")

    del model
    torch.cuda.empty_cache()

    return elapsed, tokens_generated


def benchmark_vllm_tp2(model_path: str, num_tokens: int = 20):
    """vLLM TP=2 推理"""
    print("\n" + "=" * 60)
    print("Benchmark: vLLM Tensor Parallel (TP=2)")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed, skipping...")
        return None, None

    # 检查 GPU 数量
    if torch.cuda.device_count() < 2:
        print(f"Need 2 GPUs for TP=2, but only {torch.cuda.device_count()} available")
        return None, None

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",
    )

    prompt = "The capital of France is"
    sampling_params = SamplingParams(
        max_tokens=num_tokens,
        temperature=0,
    )

    # Warmup
    for _ in range(3):
        _ = llm.generate([prompt], sampling_params)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 5
    for _ in range(iterations):
        outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / iterations * 1000

    response = outputs[0].outputs[0].text
    tokens_generated = len(outputs[0].outputs[0].token_ids)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Generation time: {elapsed:.2f} ms")
    print(f"Tokens/sec: {tokens_generated / (elapsed / 1000):.1f}")

    del llm
    torch.cuda.empty_cache()

    return elapsed, tokens_generated


def benchmark_maco_mlp(model_path: str, num_layers: int = 4):
    """MACO TaskGraph MLP 执行"""
    print("\n" + "=" * 60)
    print(f"Benchmark: MACO TaskGraph MLP ({num_layers} layers)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoConfig
    from maco import TaskGraph

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    hidden_size = config.hidden_size
    batch_size = 8
    seq_len = 128

    print(f"Hidden size: {hidden_size}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")

    # 提取权重
    all_weights = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        weights = {
            "gate_proj": layer.mlp.gate_proj.weight.data,
            "up_proj": layer.mlp.up_proj.weight.data,
            "down_proj": layer.mlp.down_proj.weight.data,
        }
        all_weights.append(weights)

    # MACO TaskGraph
    x = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

    graph = TaskGraph(num_workers=8)

    current = x
    for layer_idx in range(num_layers):
        weights = all_weights[layer_idx]
        gate = graph.linear(current, weights["gate_proj"], name=f"l{layer_idx}_gate")
        up = graph.linear(current, weights["up_proj"], name=f"l{layer_idx}_up")
        down = graph.linear(gate.output, weights["down_proj"], name=f"l{layer_idx}_down")
        current = down.output

    graph.compile()

    # Warmup
    for _ in range(5):
        graph.execute()
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    maco_time = (time.perf_counter() - start) / iterations * 1000

    # PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        h = x
        for layer_idx in range(num_layers):
            weights = all_weights[layer_idx]
            gate_out = torch.nn.functional.linear(h, weights["gate_proj"])
            up_out = torch.nn.functional.linear(h, weights["up_proj"])
            down_out = torch.nn.functional.linear(gate_out, weights["down_proj"])
            h = down_out

    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\nMACO TaskGraph time: {maco_time:.3f} ms")
    print(f"PyTorch baseline time: {pytorch_time:.3f} ms")
    print(f"Speedup: {pytorch_time / maco_time:.2f}x")
    print(f"\n{graph.summary()}")

    del model
    torch.cuda.empty_cache()

    return maco_time, pytorch_time


def benchmark_maco_full_layer(model_path: str, num_layers: int = 4):
    """MACO TaskGraph 完整层执行（包含 QKV）"""
    print("\n" + "=" * 60)
    print(f"Benchmark: MACO TaskGraph Full Layer ({num_layers} layers)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoConfig
    from maco import TaskGraph

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    hidden_size = config.hidden_size
    batch_size = 8
    seq_len = 128

    print(f"Hidden size: {hidden_size}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")

    # 提取权重
    all_weights = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        weights = {
            "q_proj": layer.self_attn.q_proj.weight.data,
            "k_proj": layer.self_attn.k_proj.weight.data,
            "v_proj": layer.self_attn.v_proj.weight.data,
            "o_proj": layer.self_attn.o_proj.weight.data,
            "gate_proj": layer.mlp.gate_proj.weight.data,
            "up_proj": layer.mlp.up_proj.weight.data,
            "down_proj": layer.mlp.down_proj.weight.data,
        }
        all_weights.append(weights)

    # MACO TaskGraph - 只执行线性投影，跳过 attention 计算
    x = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

    graph = TaskGraph(num_workers=8)

    current = x
    for layer_idx in range(num_layers):
        weights = all_weights[layer_idx]

        # QKV 投影
        q = graph.linear(current, weights["q_proj"], name=f"l{layer_idx}_q")
        k = graph.linear(current, weights["k_proj"], name=f"l{layer_idx}_k")
        v = graph.linear(current, weights["v_proj"], name=f"l{layer_idx}_v")

        # 简化：使用 q 作为 attention 输出（跳过实际 attention 计算）
        # 因为 q 的输出维度与 o_proj 的输入维度匹配
        o = graph.linear(q.output, weights["o_proj"], name=f"l{layer_idx}_o")

        # MLP
        gate = graph.linear(o.output, weights["gate_proj"], name=f"l{layer_idx}_gate")
        up = graph.linear(o.output, weights["up_proj"], name=f"l{layer_idx}_up")
        down = graph.linear(gate.output, weights["down_proj"], name=f"l{layer_idx}_down")

        current = down.output

    graph.compile()

    # Warmup
    for _ in range(5):
        graph.execute()
    torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    maco_time = (time.perf_counter() - start) / iterations * 1000

    # PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        h = x
        for layer_idx in range(num_layers):
            weights = all_weights[layer_idx]
            q_out = torch.nn.functional.linear(h, weights["q_proj"])
            k_out = torch.nn.functional.linear(h, weights["k_proj"])
            v_out = torch.nn.functional.linear(h, weights["v_proj"])
            o_out = torch.nn.functional.linear(q_out, weights["o_proj"])
            gate_out = torch.nn.functional.linear(o_out, weights["gate_proj"])
            up_out = torch.nn.functional.linear(o_out, weights["up_proj"])
            down_out = torch.nn.functional.linear(gate_out, weights["down_proj"])
            h = down_out

    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\nMACO TaskGraph time: {maco_time:.3f} ms")
    print(f"PyTorch baseline time: {pytorch_time:.3f} ms")
    print(f"Speedup: {pytorch_time / maco_time:.2f}x")
    print(f"\n{graph.summary()}")

    del model
    torch.cuda.empty_cache()

    return maco_time, pytorch_time


def main():
    parser = argparse.ArgumentParser(description="MACO vs vLLM Benchmark")
    parser.add_argument("--mode", choices=["single", "vllm", "maco", "full", "all"],
                        default="all", help="Benchmark mode")
    parser.add_argument("--model", default="/vllm-workspace/Qwen3-0.6B-Base",
                        help="Model path")
    parser.add_argument("--tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--layers", type=int, default=4, help="Layers for MACO test")
    args = parser.parse_args()

    print("=" * 60)
    print("MACO vs vLLM Performance Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Model: {args.model}")

    results = {}

    if args.mode in ["single", "all"]:
        results["single"] = benchmark_single_gpu(args.model, args.tokens)

    if args.mode in ["vllm", "all"]:
        results["vllm"] = benchmark_vllm_tp2(args.model, args.tokens)

    if args.mode in ["maco", "all"]:
        results["maco_mlp"] = benchmark_maco_mlp(args.model, args.layers)

    if args.mode in ["full", "all"]:
        results["maco_full"] = benchmark_maco_full_layer(args.model, args.layers)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results.items():
        if result[0] is not None:
            print(f"{name}: {result}")


if __name__ == "__main__":
    main()
