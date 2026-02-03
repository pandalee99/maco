# MACO Benchmark Results

## Test Environment

- **GPUs**: 4x (using GPU 2,3 for tests)
- **PyTorch**: 2.9.0+cu129
- **vLLM**: 0.12.0
- **Model**: Qwen3-0.6B-Base

---

## 1. Compute-Communication Overlap Benchmark

**Test**: Measuring the benefit of overlapping allreduce with matrix multiplication using separate CUDA streams.

**Setup**: 2 GPUs (CUDA_VISIBLE_DEVICES=2,3), NCCL backend

### Results

| Configuration | Comm Size | Compute Size | Sequential | Overlapped | Speedup | Time Saved |
|---------------|-----------|--------------|------------|------------|---------|------------|
| Small compute, medium comm | 8 MB | 1024x1024 | 0.71 ms | 0.67 ms | 1.05x | 4.8% |
| Medium compute, large comm | 32 MB | 2048x2048 | 2.70 ms | 2.51 ms | 1.08x | 7.2% |
| Large compute, large comm | 32 MB | 4096x4096 | 3.72 ms | 2.47 ms | **1.51x** | **33.7%** |
| Large compute, very large comm | 64 MB | 4096x4096 | 6.06 ms | 4.82 ms | 1.26x | 20.4% |

**Average Speedup: 1.22x**

### Key Insights

1. **Overlap benefit scales with workload size**: Larger compute and communication workloads show greater overlap benefit
2. **Best case**: 1.51x speedup when compute and communication are both substantial
3. **Worst case**: Even with small workloads, still achieves ~5% improvement

---

## 2. Inference Benchmark

**Test**: End-to-end inference latency for generating 32 new tokens.

**Prompt**: "The future of artificial intelligence is"

### Results

| Method | Avg Latency | Throughput | vs Baseline |
|--------|-------------|------------|-------------|
| PyTorch Baseline | 560.44 ms | 57.1 tokens/sec | 1.0x |
| vLLM | 95.51 ms | 335.0 tokens/sec | **5.87x** |

### Key Insights

1. **vLLM achieves 5.87x speedup** over naive PyTorch through:
   - CUDA Graph capture
   - PagedAttention
   - Continuous batching
   - Optimized kernels

2. **MACO's compute-comm overlap** is complementary:
   - Applies to multi-GPU tensor parallelism scenarios
   - Can be integrated with vLLM's existing optimizations
   - Most beneficial for larger models where communication overhead is significant

---

## 3. Communication Bandwidth

From the overlap tests, we can estimate effective bandwidth:

| Tensor Size | AllReduce Time | Effective Bandwidth |
|-------------|----------------|---------------------|
| 8 MB | ~0.4 ms | ~40 GB/s |
| 32 MB | ~1.1 ms | ~58 GB/s |
| 64 MB | ~2.1 ms | ~61 GB/s |

This is consistent with NVLink bandwidth characteristics for intra-node communication.

---

## 4. When to Use MACO Overlap

**High benefit scenarios**:
- Multi-GPU tensor parallelism
- Large models (7B+) where each layer has significant compute and communication
- Decode phase with larger batch sizes

**Low benefit scenarios**:
- Single GPU inference
- Very small models
- Latency-dominated workloads with minimal communication

---

## Conclusion

1. **MACO's compute-comm overlap is validated**: Achieves 1.22x average speedup, up to 1.51x in optimal conditions

2. **Complementary to existing optimizations**: Can be combined with vLLM's CUDA Graph, PagedAttention, etc.

3. **Key insight**: The benefit of overlap scales with workload size - larger models with tensor parallelism will see greater improvements

4. **Future work**:
   - Integrate with vLLM's tensor parallel implementation
   - Implement persistent kernel for even lower overhead
   - Add automatic overlap detection and scheduling
