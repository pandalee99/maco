# MACO: Multi-GPU Async Communication Optimizer

## Design Goals

1. **Communication as First-Class Citizen** - 通信不是事后想法，而是核心抽象
2. **Explicit SM Control** - 用户可以控制 SM 的角色分配
3. **Minimal API Surface** - 简单易用，渐进式复杂度
4. **PyTorch Compatible** - 无缝集成现有 PyTorch 代码

---

## Core API Design

### Level 1: Simple API (最简单的使用方式)

```python
import maco

# 初始化（自动检测 GPU 拓扑）
maco.init(backend="nvshmem")  # or "nccl_optimized"

# 最简单的用法：包装现有通信操作
@maco.optimize_comm
def my_forward(x, weight):
    y = torch.matmul(x, weight)
    y = maco.allreduce(y)  # 替代 dist.all_reduce
    return y

# 执行
output = my_forward(input_tensor, weight_tensor)
```

### Level 2: Explicit SM Control (显式控制 SM 分配)

```python
import maco

# 创建通信优化的执行上下文
ctx = maco.Context(
    world_size=8,
    compute_sms=96,      # 96 个 SM 做计算
    comm_sms=32,         # 32 个 SM 做通信
    overlap=True,        # 允许计算-通信重叠
)

with ctx:
    # 定义计算阶段
    y = ctx.compute(lambda: torch.matmul(x, w1))

    # 定义通信（与下一个计算重叠）
    y_reduced = ctx.allreduce(y, async_op=True)

    # 下一个计算（与通信重叠执行）
    z = ctx.compute(lambda: torch.matmul(x, w2))

    # 同步
    y_reduced.wait()
```

### Level 3: Persistent Kernel Mode (最大性能)

```python
import maco

# 定义持久化 kernel 配置
kernel = maco.PersistentKernel(
    world_size=8,
    rank=maco.get_rank(),
    num_workers=96,
    num_local_schedulers=24,
    num_remote_schedulers=8,
)

# 注册张量
x = kernel.register_tensor(input_tensor, name="input")
w = kernel.register_tensor(weight_tensor, name="weight")
out = kernel.register_output(shape=(M, N), dtype=torch.bfloat16, name="output")

# 定义任务图
with kernel.task_graph() as graph:
    # 计算任务
    y = graph.matmul(x, w, output=out)

    # 通信任务（自动与计算重叠）
    y_reduced = graph.allreduce(y)

    # 更多计算...
    z = graph.matmul(y_reduced, w2)

# 编译
kernel.compile()

# 执行（单次 kernel launch）
for step in range(num_steps):
    kernel()
```

---

## Communication Patterns

### Supported Patterns

```python
# 1. AllReduce (最常用)
y = maco.allreduce(x, op=maco.SUM)

# 2. AllGather
gathered = maco.allgather(x)  # [world_size, ...]

# 3. ReduceScatter
scattered = maco.reduce_scatter(x, op=maco.SUM)

# 4. Point-to-Point
maco.send(x, dst=1)
y = maco.recv(src=0, shape=x.shape, dtype=x.dtype)

# 5. Ring Pattern (for pipeline parallelism)
with maco.ring(direction="forward"):
    maco.send(activation, dst=(rank+1) % world_size)
    recv_activation = maco.recv(src=(rank-1) % world_size)
```

### Communication Scheduling

```python
# 显式定义通信调度
scheduler = maco.CommScheduler(num_sms=32)

# 定义通信阶段
phase1 = scheduler.phase("allreduce_qkv")
phase1.allreduce(q_proj)
phase1.allreduce(k_proj)
phase1.allreduce(v_proj)

phase2 = scheduler.phase("allreduce_output", depends_on=phase1)
phase2.allreduce(attn_output)

# 编译调度
scheduler.compile()
```

---

## Overlap Strategies

### Strategy 1: Automatic Overlap

```python
# 框架自动决定重叠策略
with maco.auto_overlap():
    y1 = matmul(x, w1)
    y1_reduced = maco.allreduce(y1)  # 自动与下一个 matmul 重叠
    y2 = matmul(x, w2)
    y2_reduced = maco.allreduce(y2)
```

### Strategy 2: Manual Overlap Control

```python
# 手动控制重叠
with maco.overlap_region() as region:
    # 这些操作会被重叠执行
    compute_future = region.submit_compute(matmul, x, w)
    comm_future = region.submit_comm(maco.allreduce, prev_y)

# 离开 region 时自动同步
y = compute_future.result()
prev_y_reduced = comm_future.result()
```

### Strategy 3: Pipeline Overlap

```python
# 流水线式重叠（适合 Transformer layers）
pipeline = maco.Pipeline(num_stages=3)

@pipeline.stage(0)
def stage_attention(x):
    return attention(x)

@pipeline.stage(1)
def stage_allreduce(x):
    return maco.allreduce(x)

@pipeline.stage(2)
def stage_ffn(x):
    return ffn(x)

# 三个 stage 流水线执行
output = pipeline(input)
```

---

## SM Role Assignment

### Automatic Assignment

```python
# 根据工作负载自动分配
config = maco.auto_config(
    compute_intensity="high",  # high/medium/low
    comm_intensity="medium",
    gpu_arch="hopper",
)
# 返回推荐的 SM 分配
# config.compute_sms = 100
# config.comm_sms = 32
```

### Manual Assignment

```python
# 精细控制每个 SM 的角色
sm_config = maco.SMConfig(total_sms=132)

# 分配 SM 角色
sm_config.assign_role(
    sm_range=(0, 96),
    role=maco.Role.COMPUTE,
    task_types=["matmul", "attention"]
)

sm_config.assign_role(
    sm_range=(96, 120),
    role=maco.Role.LOCAL_SCHEDULER,
)

sm_config.assign_role(
    sm_range=(120, 132),
    role=maco.Role.REMOTE_SCHEDULER,  # 处理跨 GPU 通信
)
```

### Dynamic Assignment

```python
# 根据运行时负载动态调整
@maco.dynamic_sm_config
def forward(x):
    # 在 prefill 阶段：更多计算 SM
    if is_prefill:
        maco.set_config(compute_sms=120, comm_sms=12)
    # 在 decode 阶段：更多通信 SM（因为 batch 小）
    else:
        maco.set_config(compute_sms=80, comm_sms=52)

    return model(x)
```

---

## Integration with PyTorch

### Decorator-based Integration

```python
import torch
import torch.distributed as dist
import maco

# 方式 1：装饰器替换
@maco.optimize
class MyModel(nn.Module):
    def forward(self, x):
        y = self.linear1(x)
        # dist.all_reduce 自动被优化
        dist.all_reduce(y)
        z = self.linear2(y)
        return z

# 方式 2：Context Manager
with maco.optimize_context():
    output = model(input)
```

### Monkey Patching (零代码修改)

```python
# 直接替换 torch.distributed
maco.patch_torch_distributed()

# 现有代码无需修改
dist.all_reduce(tensor)  # 自动使用优化版本
```

### Compile Mode Integration

```python
# 与 torch.compile 结合
model = torch.compile(model, backend="maco")

# 或者
@torch.compile(backend="maco")
def forward(x):
    ...
```

---

## Example: Optimized Tensor Parallel Linear

```python
import maco
import torch

class TPLinear(maco.Module):
    """Tensor Parallel Linear with optimized communication"""

    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        # 按列切分权重
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features // world_size)
        )

    def forward(self, x):
        # 本地 matmul
        y = torch.matmul(x, self.weight.T)

        # 优化的 allreduce（与其他计算重叠）
        y = maco.allreduce(y, overlap_with_next=True)

        return y


# 使用示例
model = TPLinear(4096, 4096, world_size=8)

# 创建优化上下文
with maco.PersistentContext(model) as ctx:
    for batch in dataloader:
        output = ctx.forward(batch)
```

---

## Example: Full Transformer Layer

```python
import maco

class OptimizedTransformerLayer(maco.Module):
    def __init__(self, config):
        self.config = config
        # ... 初始化权重 ...

    @maco.task_graph
    def forward(self, x, kv_cache=None):
        # 定义计算和通信的任务图

        # Phase 1: Attention 输入投影（可与上一层通信重叠）
        qkv = self.qkv_proj(x)

        # Phase 2: Attention 计算
        attn_out = self.attention(qkv, kv_cache)

        # Phase 3: Attention 输出投影 + AllReduce
        attn_out = self.o_proj(attn_out)
        attn_out = maco.allreduce(attn_out)  # 标记为可重叠

        # Phase 4: Residual + Norm（与 AllReduce 重叠）
        x = x + attn_out
        x = self.norm1(x)

        # Phase 5: FFN
        ffn_out = self.ffn(x)
        ffn_out = maco.allreduce(ffn_out)

        # Phase 6: Final Residual
        x = x + ffn_out
        x = self.norm2(x)

        return x


# 编译整个模型
model = OptimizedTransformerModel(config)
compiled_model = maco.compile(model)

# 单次 kernel launch 执行所有层
output = compiled_model(input_ids)
```

---

## Configuration

### Global Configuration

```python
maco.config.set(
    # 通信后端
    backend="nvshmem",  # "nvshmem" | "nccl" | "gloo"

    # SM 分配策略
    sm_policy="auto",  # "auto" | "manual" | "dynamic"

    # 重叠策略
    overlap_strategy="aggressive",  # "none" | "conservative" | "aggressive"

    # 内存管理
    comm_buffer_size="256MB",
    use_pinned_memory=True,

    # 调试
    debug=False,
    profile=True,
)
```

### Per-Operation Configuration

```python
# 细粒度控制单个操作
maco.allreduce(
    tensor,
    op=maco.SUM,
    async_op=True,
    priority="high",        # 通信优先级
    overlap_group="attn",   # 重叠分组
    use_ring=False,         # 是否使用 ring allreduce
)
```

---

## Profiling & Debugging

```python
# 启用性能分析
with maco.profiler() as prof:
    output = model(input)

# 打印报告
prof.print_summary()
# Output:
# ┌─────────────────┬──────────┬──────────┬──────────┐
# │ Operation       │ Time(ms) │ Overlap% │ SM Usage │
# ├─────────────────┼──────────┼──────────┼──────────┤
# │ matmul_qkv      │ 0.42     │ 85%      │ 96 SMs   │
# │ allreduce_attn  │ 0.18     │ 85%      │ 32 SMs   │
# │ matmul_ffn      │ 0.38     │ 72%      │ 96 SMs   │
# │ allreduce_ffn   │ 0.15     │ 72%      │ 32 SMs   │
# └─────────────────┴──────────┴──────────┴──────────┘
# Total overlap efficiency: 78%

# 导出 trace
prof.export_chrome_trace("maco_trace.json")
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] NVSHMEM wrapper
- [ ] SM role manager
- [ ] Basic persistent kernel framework
- [ ] Simple allreduce implementation

### Phase 2: PyTorch Integration (Week 3-4)
- [ ] Tensor registration
- [ ] Autograd integration
- [ ] dist.all_reduce replacement
- [ ] Basic overlap support

### Phase 3: Advanced Features (Week 5-6)
- [ ] Task graph compiler
- [ ] Automatic overlap detection
- [ ] Dynamic SM assignment
- [ ] Profiler

### Phase 4: Optimization (Week 7-8)
- [ ] Ring allreduce optimization
- [ ] Memory pooling
- [ ] Multi-node support
- [ ] Performance tuning

---

## File Structure

```
maco/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── context.py          # 执行上下文
│   ├── sm_manager.py       # SM 角色管理
│   ├── tensor_registry.py  # 张量注册
│   └── persistent_kernel.py # 持久化 kernel
├── comm/
│   ├── __init__.py
│   ├── nvshmem_backend.py  # NVSHMEM 后端
│   ├── patterns.py         # 通信模式
│   └── scheduler.py        # 通信调度器
├── overlap/
│   ├── __init__.py
│   ├── analyzer.py         # 重叠分析
│   └── strategies.py       # 重叠策略
├── integration/
│   ├── __init__.py
│   ├── torch_patch.py      # PyTorch 集成
│   └── compile_backend.py  # torch.compile 后端
├── profiler/
│   ├── __init__.py
│   └── profiler.py         # 性能分析
└── codegen/
    ├── __init__.py
    ├── cuda_gen.py         # CUDA 代码生成
    └── task_graph.py       # 任务图编译
```
