# Mirage Persistent Kernel (MPK) - Principles and Architecture

## Overview

Mirage Persistent Kernel (MPK) is a compiler and runtime system that automatically transforms LLM inference into a **single megakernel** - a fused GPU kernel that performs all necessary computation and communication within a single kernel launch. This end-to-end GPU fusion approach reduces LLM inference latency by 1.2× to 6.7×.

## Core Concepts

### 1. Megakernel Approach

Traditional LLM inference involves launching many separate GPU kernels sequentially:
- Each kernel launch has overhead (scheduling, memory synchronization)
- Context switching between kernels wastes time
- Memory bandwidth is underutilized

MPK solves this by **fusing all operations into one persistent kernel** that:
- Launches once and runs continuously
- Handles all computation (attention, linear layers, normalization, etc.)
- Manages inter-GPU communication internally
- Eliminates kernel launch overhead

### 2. Task Graph Model

MPK represents LLM computation as a **task graph**:

```
Task Graph = (Tasks, Events, Dependencies)
```

- **Tasks**: Individual compute operations (embedding, attention, linear, etc.)
- **Events**: Synchronization points between tasks
- **Dependencies**: Data flow relationships between tasks

Each task is defined by:
```cpp
struct TaskDesc {
    TaskType task_type;        // Type of computation
    unsigned variant_id;       // Hardware-specific variant
    EventId trigger_event;     // Wait for this event before starting
    EventId dependent_event;   // Signal this event when done
    void* input_ptrs[];        // Input tensor pointers
    void* output_ptrs[];       // Output tensor pointers
};
```

### 3. Worker-Scheduler Architecture

The megakernel uses a **distributed scheduling model**:

```
                    ┌─────────────────┐
                    │   Schedulers    │
                    │  (dispatch      │
                    │   events/tasks) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ Worker  │         │ Worker  │         │ Worker  │
   │   SM0   │         │   SM1   │         │  SM_N   │
   └─────────┘         └─────────┘         └─────────┘
```

- **Workers**: GPU SMs that execute compute tasks
- **Local Schedulers**: Manage task queues within a single GPU
- **Remote Schedulers**: Handle cross-GPU communication (NVSHMEM)

Configuration parameters:
- `num_workers`: Number of worker SMs
- `num_local_schedulers`: Number of local scheduling SMs
- `num_remote_schedulers`: Number of remote scheduling SMs (for multi-GPU)

### 4. Memory Hierarchy Optimization

MPK leverages GPU memory hierarchy efficiently:

```
┌─────────────────────────────────────────────┐
│              Global Memory (HBM)            │
│   - Model weights                           │
│   - KV cache (paged)                        │
│   - Input/output tensors                    │
└─────────────────────────────────────────────┘
                    ▲
                    │
┌─────────────────────────────────────────────┐
│            Shared Memory (SRAM)             │
│   - Intermediate activations                │
│   - Task descriptors                        │
│   - Reduction buffers                       │
└─────────────────────────────────────────────┘
                    ▲
                    │
┌─────────────────────────────────────────────┐
│               Registers                     │
│   - Thread-local computation               │
└─────────────────────────────────────────────┘
```

For modern GPUs (Hopper/Blackwell), MPK uses:
- **TMA (Tensor Memory Accelerator)**: Hardware-assisted async memory copies
- **Dynamic shared memory**: Up to 225KB per SM on Hopper

## Compilation Pipeline

### Step 1: Graph Definition

Users define computation using Python API:

```python
mpk = mi.PersistentKernel(
    world_size=1,
    num_workers=96,
    num_local_schedulers=48,
    ...
)

# Attach tensors
x = mpk.attach_input(torch_tensor, name="input")

# Define layers
mpk.rmsnorm_linear_layer(input=x, weight_norm=w, ...)
mpk.attention_layer(...)
mpk.linear_layer(...)
```

### Step 2: Task Graph Generation

The compiler converts the high-level graph to a task graph:

```python
results = kn_graph.generate_task_graph(num_gpus, my_gpu_id)
```

This produces:
- CUDA source code for the megakernel
- JSON task graph description
- TMA descriptors for async memory operations

### Step 3: CUDA Compilation

The generated CUDA code is compiled with nvcc:

```bash
nvcc -arch=sm_90a -O3 -shared ... -o megakernel.so
```

### Step 4: Runtime Initialization

```cpp
init_persistent_kernel(
    meta_tensors,      // Step counter, token buffer, etc.
    profiler_buffer,   // Optional profiling
    num_workers,
    num_schedulers,
    ...
);
```

## Supported Operations

### Fused Layers
| Layer Type | Description |
|------------|-------------|
| `embed_layer` | Token embedding lookup |
| `rmsnorm_layer` | RMS normalization |
| `rmsnorm_linear_layer` | Fused RMSNorm + Linear |
| `attention_layer` | Multi-head/grouped-query attention |
| `paged_attention_layer` | Paged KV-cache attention |
| `linear_layer` | Dense matrix multiplication |
| `linear_with_residual_layer` | Linear + residual addition |
| `silu_mul_layer` | SiLU activation with gating |
| `allreduce_layer` | Multi-GPU all-reduce |
| `argmax_layer` | Token sampling |

### MoE Support
| Layer Type | Description |
|------------|-------------|
| `moe_topk_softmax_routing_layer` | Expert routing |
| `moe_w13_linear_layer` | MoE gate/up projection |
| `moe_w2_linear_layer` | MoE down projection |
| `moe_mul_sum_add_layer` | Expert output combination |

## Multi-GPU Support

MPK supports tensor parallelism across multiple GPUs using NVSHMEM:

```
GPU 0                    GPU 1
┌────────────┐          ┌────────────┐
│  Worker    │◄────────►│  Worker    │
│  SMs       │ NVSHMEM  │  SMs       │
├────────────┤          ├────────────┤
│  Remote    │◄────────►│  Remote    │
│ Scheduler  │          │ Scheduler  │
└────────────┘          └────────────┘
```

Communication primitives:
- `nvshmem_allgather_strided_put`: Distributed gather
- Custom reduction kernels for all-reduce

## Paged Attention

For serving scenarios, MPK implements **paged KV-cache**:

```
┌─────────────────────────────────────────────┐
│              Paged KV Cache                 │
│  ┌────┬────┬────┬────┬────┬────┐          │
│  │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ ...      │
│  └────┴────┴────┴────┴────┴────┘          │
│    │         │                             │
│    ▼         ▼                             │
│  Request 0  Request 1                      │
│  pages=[0,2] pages=[1,3,4]                 │
└─────────────────────────────────────────────┘
```

Key parameters:
- `max_num_pages`: Total pages in cache
- `page_size`: Tokens per page
- `max_seq_length`: Maximum sequence length

## Speculative Decoding

MPK supports speculative decoding for faster inference:

```python
# Prompt lookup speculation
spec_config = PromptLookupConfig(
    ngram_size=3,
    spec_length=5
)

# Draft then verify
spec_tokens = mpk.draft_forward_layer_dispatcher(spec_config, tokens, ...)
verified = mpk.verify_layer_dispatcher(spec_config, spec_tokens, target_output, ...)
```

## Hardware Support

| GPU Architecture | Compute Capability | Features |
|-----------------|-------------------|----------|
| Ampere (A100)   | SM 80             | Basic megakernel |
| Ada (RTX 4090)  | SM 86/89          | Improved shared memory |
| Hopper (H100)   | SM 90             | TMA, 225KB SRAM |
| Blackwell (B200)| SM 100            | Enhanced TMA |

## Performance Benefits

1. **Reduced Launch Overhead**: Single kernel launch vs. hundreds
2. **Better Memory Reuse**: Activations stay in shared memory
3. **Overlapped Compute/Communication**: Async scheduling
4. **Hardware-Optimized**: TMA for memory, Tensor Cores for compute

## References

- Paper: [Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://arxiv.org/abs/2405.05751) (OSDI 2025)
- Paper: [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219) (arXiv 2025)
- Repository: https://github.com/mirage-project/mirage
