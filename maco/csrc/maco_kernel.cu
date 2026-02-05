/**
 * MACO Persistent Kernel - SM 调度主入口
 *
 * 这是 MACO SM 调度系统的核心。
 * 单次 kernel launch，整个推理过程都在 GPU 上运行。
 *
 * 架构:
 * - Workers (blockIdx.x < num_workers): 执行计算任务
 * - Schedulers (blockIdx.x >= num_workers): 分发任务
 *
 * 不依赖 NVSHMEM，只使用标准 CUDA + PyTorch。
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include "maco_types.h"
#include "maco_atoms.cuh"
#include "maco_worker.cuh"
#include "maco_scheduler.cuh"

namespace maco {

// ============================================================================
// Persistent Kernel 主入口
// ============================================================================

/**
 * MACO Persistent Kernel
 *
 * 根据 blockIdx.x 决定 CTA 角色:
 * - blockIdx.x < num_workers: Worker CTA
 * - blockIdx.x >= num_workers: Scheduler CTA
 */
__global__ void maco_persistent_kernel(MacoRuntimeConfig config) {
    if (blockIdx.x < config.num_workers) {
        // Worker CTA
        execute_worker(&config);
    } else {
        // Scheduler CTA
        execute_scheduler(&config);
    }
}

// ============================================================================
// 简化版 Persistent Kernel (用于演示)
// ============================================================================

/**
 * 简化版 Persistent Kernel - 不使用 Scheduler
 *
 * 每个 Worker 直接从自己的队列获取任务。
 * 任务由 CPU 预先分配好。
 *
 * 这个版本更简单，适合验证 GPU Atomics 的正确性。
 */
__global__ void maco_simple_persistent_kernel(
    MacoTaskDesc* task_queues,      // [num_workers][queue_size]
    uint64_t* queue_heads,          // [num_workers] - 每个队列的头
    uint64_t* queue_tails,          // [num_workers] - 每个队列的尾
    uint64_t* event_counters,       // [num_events] - 事件计数器
    MacoEventDesc* events,          // 事件描述符
    int* terminate_flag,            // 终止标志
    int* tasks_completed,           // 完成计数
    int queue_size,
    int num_workers
) {
    int worker_id = blockIdx.x;
    if (worker_id >= num_workers) return;

    // 获取这个 Worker 的队列
    MacoTaskDesc* my_queue = task_queues + worker_id * queue_size;
    uint64_t local_tail = 0;

    // Shared memory for task caching
    __shared__ MacoTaskDesc cached_task;

    while (true) {
        // 1. Polling: 等待新任务
        uint64_t head;
        int spin = 0;
        while (true) {
            head = ld_acquire_gpu_u64(&queue_heads[worker_id]);
            if (local_tail < head) break;

            // 检查终止
            if (ld_acquire_gpu_u32(reinterpret_cast<unsigned int*>(
                    terminate_flag)) != 0) {
                return;
            }

            // 检查队列中是否有终止任务
            if (my_queue[local_tail % queue_size].task_type == MACO_TASK_TERMINATE) {
                return;
            }

            maco_nanosleep(10);
            if (++spin > 1000000) spin = 0;
        }

        // 2. 获取任务
        if (threadIdx.x == 0) {
            cached_task = my_queue[local_tail % queue_size];
        }
        __syncthreads();

        // 3. 终止检查
        if (cached_task.task_type == MACO_TASK_TERMINATE) {
            return;
        }

        // 4. 等待依赖事件
        if (cached_task.dependent_event != EVENT_INVALID_ID) {
            uint64_t evt_id = cached_task.dependent_event;
            uint64_t needed = events[evt_id].needed_triggers;
            while (ld_acquire_gpu_u64(&event_counters[evt_id]) < needed) {
                maco_nanosleep(10);
            }
        }
        __syncthreads();

        // 5. 执行任务
        execute_task(&cached_task);
        __syncthreads();

        // 6. 触发事件
        if (cached_task.trigger_event != EVENT_INVALID_ID && threadIdx.x == 0) {
            atom_add_release_gpu_u64(&event_counters[cached_task.trigger_event], 1);
            atomicAdd(tasks_completed, 1);
        }
        __syncthreads();

        // 7. 移动到下一个
        local_tail++;
        if (threadIdx.x == 0) {
            st_release_gpu_u64(&queue_tails[worker_id], local_tail);
        }
    }
}

// ============================================================================
// 演示用的简单 Kernel
// ============================================================================

/**
 * 演示真正的 SM 调度效果
 *
 * 这个 kernel 执行一系列任务，展示:
 * 1. GPU Atomics 同步
 * 2. Lock-free 队列操作
 * 3. 任务依赖管理
 */
__global__ void demo_sm_scheduling_kernel(
    float* data,                    // 工作数据
    uint64_t* sync_counter,         // 同步计数器
    int* task_completion_order,     // 记录任务完成顺序
    int* completion_index,          // 完成索引
    int data_size,
    int num_tasks_per_worker,
    int num_workers
) {
    int worker_id = blockIdx.x;
    if (worker_id >= num_workers) return;

    // 每个 Worker 处理一部分数据
    int chunk_size = data_size / num_workers;
    int start = worker_id * chunk_size;
    int end = (worker_id == num_workers - 1) ? data_size : start + chunk_size;

    for (int task = 0; task < num_tasks_per_worker; task++) {
        // 执行计算 (简单的向量操作)
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            data[i] = data[i] * 1.5f + 0.1f;
        }
        __syncthreads();

        // 使用 GPU Atomics 记录完成
        if (threadIdx.x == 0) {
            // 获取当前完成索引
            int idx = atomicAdd(completion_index, 1);
            task_completion_order[idx] = worker_id * 1000 + task;

            // 通知其他 Workers
            atom_add_release_gpu_u64(sync_counter, 1);
        }
        __syncthreads();

        // 等待其他 Workers 完成这个阶段 (屏障)
        if (threadIdx.x == 0) {
            uint64_t expected = (uint64_t)(task + 1) * num_workers;
            while (ld_acquire_gpu_u64(sync_counter) < expected) {
                maco_nanosleep(10);
            }
        }
        __syncthreads();
    }
}

}  // namespace maco

// ============================================================================
// Python 绑定
// ============================================================================

/**
 * 运行 SM 调度演示
 *
 * 返回:
 * - data: 处理后的数据
 * - completion_order: 任务完成顺序
 */
std::tuple<torch::Tensor, torch::Tensor> run_sm_scheduling_demo(
    torch::Tensor data,
    int num_tasks_per_worker,
    int num_workers,
    int threads_per_worker
) {
    TORCH_CHECK(data.is_cuda(), "Data must be a CUDA tensor");
    TORCH_CHECK(data.is_contiguous(), "Data must be contiguous");

    // 准备同步变量
    auto sync_counter = torch::zeros({1}, data.options().dtype(torch::kInt64));
    auto completion_index = torch::zeros({1}, data.options().dtype(torch::kInt32));

    int total_tasks = num_tasks_per_worker * num_workers;
    auto completion_order = torch::zeros({total_tasks}, data.options().dtype(torch::kInt32));

    // 启动 kernel
    maco::demo_sm_scheduling_kernel<<<num_workers, threads_per_worker>>>(
        data.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(sync_counter.data_ptr<int64_t>()),
        completion_order.data_ptr<int32_t>(),
        completion_index.data_ptr<int32_t>(),
        data.numel(),
        num_tasks_per_worker,
        num_workers
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    return std::make_tuple(data, completion_order);
}

/**
 * 初始化简化版 Persistent Kernel 的运行时结构
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
init_simple_persistent_kernel_runtime(
    int num_workers,
    int queue_size,
    int num_events,
    torch::Device device
) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    // 任务队列 [num_workers][queue_size] * sizeof(MacoTaskDesc)
    size_t task_queue_bytes = num_workers * queue_size * sizeof(maco::MacoTaskDesc);
    auto task_queues = torch::zeros({(int64_t)task_queue_bytes}, options);

    // 队列头尾指针
    auto queue_heads = torch::zeros({num_workers}, options.dtype(torch::kInt64));
    auto queue_tails = torch::zeros({num_workers}, options.dtype(torch::kInt64));

    // 事件计数器
    auto event_counters = torch::zeros({num_events}, options.dtype(torch::kInt64));

    // 事件描述符
    size_t event_desc_bytes = num_events * sizeof(maco::MacoEventDesc);
    auto event_descs = torch::zeros({(int64_t)event_desc_bytes}, options);

    // 控制变量
    auto control = torch::zeros({2}, options.dtype(torch::kInt32));  // [terminate_flag, tasks_completed]

    return std::make_tuple(
        task_queues, queue_heads, queue_tails,
        event_counters, event_descs, control
    );
}

/**
 * 添加任务到指定 Worker 的队列
 */
void add_task_to_queue(
    torch::Tensor task_queues,      // 任务队列 buffer
    torch::Tensor queue_heads,      // 队列头指针
    int worker_id,
    int queue_size,
    int task_type,
    torch::Tensor input0,
    torch::Tensor output0,
    std::vector<int> dims,
    int64_t dependent_event,
    int64_t trigger_event
) {
    TORCH_CHECK(task_queues.is_cuda(), "task_queues must be on CUDA");

    // 获取队列头位置
    int64_t head = queue_heads[worker_id].item<int64_t>();
    int pos = head % queue_size;

    // 构造任务描述符
    maco::MacoTaskDesc task;
    task.task_type = task_type;
    task.task_id = static_cast<int>(head);
    task.inputs[0] = input0.defined() ? input0.data_ptr() : nullptr;
    task.inputs[1] = nullptr;
    task.inputs[2] = nullptr;
    task.inputs[3] = nullptr;
    task.outputs[0] = output0.defined() ? output0.data_ptr() : nullptr;
    task.outputs[1] = nullptr;

    for (int i = 0; i < 8 && i < dims.size(); i++) {
        task.dims[i] = dims[i];
    }

    task.dependent_event = dependent_event < 0 ? maco::EVENT_INVALID_ID : dependent_event;
    task.trigger_event = trigger_event < 0 ? maco::EVENT_INVALID_ID : trigger_event;

    // 复制到 GPU
    maco::MacoTaskDesc* queue_ptr = reinterpret_cast<maco::MacoTaskDesc*>(
        task_queues.data_ptr()) + worker_id * queue_size + pos;

    cudaMemcpy(queue_ptr, &task, sizeof(maco::MacoTaskDesc), cudaMemcpyHostToDevice);

    // 更新队列头 (原子操作在 GPU 上进行，这里简化为 CPU 更新)
    queue_heads[worker_id] = head + 1;
}

/**
 * 启动简化版 Persistent Kernel
 */
void launch_simple_persistent_kernel(
    torch::Tensor task_queues,
    torch::Tensor queue_heads,
    torch::Tensor queue_tails,
    torch::Tensor event_counters,
    torch::Tensor event_descs,
    torch::Tensor control,
    int queue_size,
    int num_workers,
    int threads_per_worker
) {
    TORCH_CHECK(task_queues.is_cuda(), "All tensors must be on CUDA");

    maco::maco_simple_persistent_kernel<<<num_workers, threads_per_worker>>>(
        reinterpret_cast<maco::MacoTaskDesc*>(task_queues.data_ptr()),
        reinterpret_cast<uint64_t*>(queue_heads.data_ptr()),
        reinterpret_cast<uint64_t*>(queue_tails.data_ptr()),
        reinterpret_cast<uint64_t*>(event_counters.data_ptr()),
        reinterpret_cast<maco::MacoEventDesc*>(event_descs.data_ptr()),
        control.data_ptr<int32_t>(),      // terminate_flag
        control.data_ptr<int32_t>() + 1,  // tasks_completed
        queue_size,
        num_workers
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
}

/**
 * 设置终止标志
 */
void set_terminate_flag(torch::Tensor control) {
    control[0] = 1;  // terminate_flag = 1
}

/**
 * 获取已完成任务数
 */
int get_tasks_completed(torch::Tensor control) {
    return control[1].item<int32_t>();
}

// ============================================================================
// PyBind11 模块
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MACO SM Scheduling Kernel - No NVSHMEM Required";

    // 演示函数
    m.def("run_sm_scheduling_demo", &run_sm_scheduling_demo,
          "Run SM scheduling demonstration",
          py::arg("data"),
          py::arg("num_tasks_per_worker") = 10,
          py::arg("num_workers") = 4,
          py::arg("threads_per_worker") = 128);

    // 运行时管理
    m.def("init_simple_persistent_kernel_runtime",
          &init_simple_persistent_kernel_runtime,
          "Initialize runtime structures for simple persistent kernel",
          py::arg("num_workers"),
          py::arg("queue_size"),
          py::arg("num_events"),
          py::arg("device"));

    m.def("add_task_to_queue", &add_task_to_queue,
          "Add a task to worker's queue",
          py::arg("task_queues"),
          py::arg("queue_heads"),
          py::arg("worker_id"),
          py::arg("queue_size"),
          py::arg("task_type"),
          py::arg("input0"),
          py::arg("output0"),
          py::arg("dims"),
          py::arg("dependent_event") = -1,
          py::arg("trigger_event") = -1);

    m.def("launch_simple_persistent_kernel",
          &launch_simple_persistent_kernel,
          "Launch the simple persistent kernel",
          py::arg("task_queues"),
          py::arg("queue_heads"),
          py::arg("queue_tails"),
          py::arg("event_counters"),
          py::arg("event_descs"),
          py::arg("control"),
          py::arg("queue_size"),
          py::arg("num_workers"),
          py::arg("threads_per_worker") = 128);

    m.def("set_terminate_flag", &set_terminate_flag,
          "Set terminate flag to stop persistent kernel");

    m.def("get_tasks_completed", &get_tasks_completed,
          "Get number of completed tasks");

    // 常量导出 (显式转换为 int)
    m.attr("TASK_TERMINATE") = static_cast<int>(maco::MACO_TASK_TERMINATE);
    m.attr("TASK_NOP") = static_cast<int>(maco::MACO_TASK_NOP);
    m.attr("TASK_MATMUL") = static_cast<int>(maco::MACO_TASK_MATMUL);
    m.attr("TASK_LINEAR") = static_cast<int>(maco::MACO_TASK_LINEAR);
    m.attr("TASK_LAYERNORM") = static_cast<int>(maco::MACO_TASK_LAYERNORM);
    m.attr("TASK_GELU") = static_cast<int>(maco::MACO_TASK_GELU);
    m.attr("TASK_SILU") = static_cast<int>(maco::MACO_TASK_SILU);
    m.attr("TASK_CUSTOM_COMPUTE") = static_cast<int>(maco::MACO_TASK_CUSTOM_COMPUTE);
    m.attr("EVENT_INVALID_ID") = static_cast<uint64_t>(maco::EVENT_INVALID_ID);
}
