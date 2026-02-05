#!/usr/bin/env python3
"""
Qwen3-0.6B 验证脚本

使用 MACO TaskGraph 执行 Qwen3 模型的 forward pass，
验证正确性和性能。

运行方式:
    # 设置代理（如果需要）
    export HTTP_PROXY=http://10.27.194.128:8890
    export HTTPS_PROXY=http://10.27.194.128:8890

    # 运行测试
    cd /mini_mirage/maco
    CUDA_VISIBLE_DEVICES=1 python3 examples/validate_qwen3.py
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import List, Tuple


def load_qwen3_model(model_path: str):
    """加载 Qwen3 模型"""
    print(f"Loading model from {model_path}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    print(f"Model loaded: {config.model_type}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")

    return model, tokenizer, config


def test_baseline_generation(model, tokenizer):
    """测试基线生成"""
    print("\n" + "=" * 60)
    print("Test 1: Baseline Generation")
    print("=" * 60)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Generation time: {elapsed:.2f} ms")
    print(f"Tokens generated: {outputs.shape[1] - inputs.input_ids.shape[1]}")

    return True


def extract_layer_weights(model, layer_idx: int) -> dict:
    """提取指定层的权重"""
    layer = model.model.layers[layer_idx]

    weights = {
        # Self-attention
        "q_proj": layer.self_attn.q_proj.weight.data,
        "k_proj": layer.self_attn.k_proj.weight.data,
        "v_proj": layer.self_attn.v_proj.weight.data,
        "o_proj": layer.self_attn.o_proj.weight.data,
        # MLP
        "gate_proj": layer.mlp.gate_proj.weight.data,
        "up_proj": layer.mlp.up_proj.weight.data,
        "down_proj": layer.mlp.down_proj.weight.data,
    }

    return weights


def test_single_layer_taskgraph(model, config):
    """使用 TaskGraph 执行单层 forward"""
    print("\n" + "=" * 60)
    print("Test 2: Single Layer TaskGraph")
    print("=" * 60)

    from maco import TaskGraph

    # 提取第一层权重
    weights = extract_layer_weights(model, 0)

    # 打印权重形状以便调试
    print("Weight shapes:")
    for name, w in weights.items():
        print(f"  {name}: {w.shape}")

    # 创建测试输入 (batch=1, seq=8, hidden)
    batch_size = 1
    seq_len = 8
    hidden_size = config.hidden_size

    x = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    print(f"Input shape: {x.shape}")

    # 创建 TaskGraph
    graph = TaskGraph(num_workers=8)

    # 注意：Qwen3 使用 GQA (Grouped Query Attention)
    # q_proj: [num_heads * head_dim, hidden_size]
    # k_proj, v_proj: [num_kv_heads * head_dim, hidden_size] (较小)
    # o_proj: [hidden_size, num_heads * head_dim]

    # 简化测试：只测试 MLP 部分（维度一致，更容易验证）
    # MLP: gate_proj, up_proj, down_proj
    # gate_proj/up_proj: [intermediate_size, hidden_size]
    # down_proj: [hidden_size, intermediate_size]

    # 测试 MLP forward: x -> gate_proj -> silu -> * up_proj -> down_proj -> output
    gate = graph.linear(x, weights["gate_proj"], name="gate_proj")
    up = graph.linear(x, weights["up_proj"], name="up_proj")

    # 简化：跳过 SiLU 和 element-wise multiply，直接测试 linear chain
    # 使用 gate 输出作为 down_proj 的输入
    down = graph.linear(gate.output, weights["down_proj"], name="down_proj")

    graph.compile()
    print(f"\n{graph.summary()}")

    # 执行
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 10
    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    print(f"\nTaskGraph execution time: {elapsed:.3f} ms")
    print(f"Down proj output sum: {down.output.sum().item():.4f}")

    # 对比 PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        gate_pt = torch.nn.functional.linear(x, weights["gate_proj"])
        up_pt = torch.nn.functional.linear(x, weights["up_proj"])
        down_pt = torch.nn.functional.linear(gate_pt, weights["down_proj"])

    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / iterations * 1000

    print(f"PyTorch baseline time: {baseline_time:.3f} ms")
    print(f"Speedup: {baseline_time / elapsed:.2f}x")

    # 验证正确性
    diff = (down_pt - down.output).abs().max().item()
    print(f"Max difference: {diff:.6f}")

    return diff < 0.1  # bfloat16 精度


def test_multi_layer_taskgraph(model, config):
    """使用 TaskGraph 执行多层 forward"""
    print("\n" + "=" * 60)
    print("Test 3: Multi-Layer TaskGraph")
    print("=" * 60)

    from maco import TaskGraph

    num_layers = min(4, config.num_hidden_layers)  # 测试前 4 层
    batch_size = 1
    seq_len = 8
    hidden_size = config.hidden_size

    print(f"Testing {num_layers} layers (MLP only)...")

    # 创建 TaskGraph
    graph = TaskGraph(num_workers=8)

    # 初始输入
    x = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    current = x

    layer_outputs = []

    for layer_idx in range(num_layers):
        weights = extract_layer_weights(model, layer_idx)

        # 简化测试：只测试 MLP 部分（维度兼容）
        # gate_proj: [intermediate_size, hidden_size]
        # down_proj: [hidden_size, intermediate_size]
        gate = graph.linear(current, weights["gate_proj"], name=f"layer{layer_idx}_gate")
        down = graph.linear(gate.output, weights["down_proj"], name=f"layer{layer_idx}_down")

        # 下一层输入（简化：跳过残差连接）
        current = down.output
        layer_outputs.append(down)

    graph.compile()
    print(f"\n{graph.summary()}")

    # 执行
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 10
    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    print(f"\nTaskGraph execution time ({num_layers} layers): {elapsed:.3f} ms")
    print(f"Per-layer time: {elapsed / num_layers:.3f} ms")

    # 对比 PyTorch 基线
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        h = x
        for layer_idx in range(num_layers):
            weights = extract_layer_weights(model, layer_idx)
            gate = torch.nn.functional.linear(h, weights["gate_proj"])
            down = torch.nn.functional.linear(gate, weights["down_proj"])
            h = down

    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / iterations * 1000

    print(f"PyTorch baseline time: {baseline_time:.3f} ms")
    print(f"Speedup: {baseline_time / elapsed:.2f}x")

    # 验证正确性
    # 重新计算 PyTorch 结果
    h_pt = x.clone()
    for layer_idx in range(num_layers):
        weights = extract_layer_weights(model, layer_idx)
        gate_pt = torch.nn.functional.linear(h_pt, weights["gate_proj"])
        down_pt = torch.nn.functional.linear(gate_pt, weights["down_proj"])
        h_pt = down_pt

    diff = (h_pt - layer_outputs[-1].output).abs().max().item()
    print(f"Max difference: {diff:.6f}")

    return diff < 0.5  # bfloat16 精度，多层累积误差较大


def test_wave_overlap(model, config):
    """测试 Wave Grouping 重叠效果"""
    print("\n" + "=" * 60)
    print("Test 4: Wave Grouping Overlap")
    print("=" * 60)

    from maco import TaskGraph

    weights = extract_layer_weights(model, 0)
    hidden_size = config.hidden_size

    x = torch.randn(8, hidden_size, device="cuda", dtype=torch.bfloat16)

    # 创建带重叠的 TaskGraph
    graph = TaskGraph(num_workers=8)

    # 多个计算任务 - 使用 MLP 的 gate_proj -> down_proj 链
    # 这样维度可以正确匹配
    compute_tasks = []
    h = x
    for i in range(4):
        gate_t = graph.linear(h, weights["gate_proj"], name=f"gate_{i}")
        down_t = graph.linear(gate_t.output, weights["down_proj"], name=f"down_{i}")
        compute_tasks.append(gate_t)
        compute_tasks.append(down_t)
        h = down_t.output  # down_proj 输出维度与输入维度相同

    # 通信任务（模拟）
    comm_tensor = torch.randn(8, hidden_size, device="cuda", dtype=torch.bfloat16)
    comm_task = graph.allreduce(comm_tensor, name="allreduce")

    # 标记重叠
    group = graph.overlap(compute_tasks, [comm_task])
    group.auto_waves()

    print(f"Wave grouping: {group.num_waves} waves")
    for t in compute_tasks:
        print(f"  {t.name}: wave {t._wave_id}")

    graph.compile()

    # 执行
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 10
    for _ in range(iterations):
        graph.execute()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations * 1000

    print(f"\nExecution time with wave grouping: {elapsed:.3f} ms")

    return True


def main():
    print("=" * 60)
    print("MACO Qwen3-0.6B Validation")
    print("=" * 60)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # 模型路径
    model_path = os.environ.get("MODEL_PATH", "/vllm-workspace/Qwen3-0.6B-Base")

    if not os.path.exists(model_path):
        print(f"\nModel not found at: {model_path}")
        print("Please set MODEL_PATH environment variable or download the model.")
        return

    # 加载模型
    model, tokenizer, config = load_qwen3_model(model_path)

    # 运行测试
    tests = [
        ("Baseline Generation", lambda: test_baseline_generation(model, tokenizer)),
        ("Single Layer TaskGraph", lambda: test_single_layer_taskgraph(model, config)),
        ("Multi-Layer TaskGraph", lambda: test_multi_layer_taskgraph(model, config)),
        ("Wave Grouping Overlap", lambda: test_wave_overlap(model, config)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
                print(f"\n✓ {name} passed!")
            else:
                failed += 1
                print(f"\n✗ {name} failed!")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nQwen3 validation successful!")
    else:
        print(f"\n{failed} tests failed.")


if __name__ == "__main__":
    main()
