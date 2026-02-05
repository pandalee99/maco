/**
 * MACO Worker CTA - 任务执行单元
 *
 * Worker 是 Persistent Kernel 中的计算执行单元。
 * 每个 Worker 是一个 CTA (blockIdx.x < num_workers)。
 *
 * 工作流程:
 * 1. 从任务队列获取任务 (polling with GPU atomics)
 * 2. 等待依赖事件 (if any)
 * 3. 执行任务
 * 4. 触发完成事件
 * 5. 继续下一个任务
 *
 * 参考: Mirage persistent_kernel.cuh Line 527-752
 */

#pragma once

#include "maco_atoms.cuh"
#include "maco_types.h"

namespace maco {

// ============================================================================
// 任务执行函数
// ============================================================================

/**
 * 执行 MatMul 任务
 *
 * dims 解释:
 * - dims[0]: M (output rows)
 * - dims[1]: N (output cols)
 * - dims[2]: K (inner dimension)
 */
__device__ void execute_matmul_task(MacoTaskDesc* task) {
    float* A = static_cast<float*>(task->inputs[0]);
    float* B = static_cast<float*>(task->inputs[1]);
    float* C = static_cast<float*>(task->outputs[0]);

    int M = task->dims[0];
    int N = task->dims[1];
    int K = task->dims[2];

    // 简化实现: 每个线程计算一个输出元素
    // 实际应该调用 cuBLAS 或 CUTLASS
    for (int idx = threadIdx.x; idx < M * N; idx += blockDim.x) {
        int i = idx / N;  // row
        int j = idx % N;  // col

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

/**
 * 执行简单的向量运算任务 (用于演示)
 *
 * dims 解释:
 * - dims[0]: vector size
 * - aux_float[0]: scalar multiplier
 */
__device__ void execute_vector_scale_task(MacoTaskDesc* task) {
    float* input = static_cast<float*>(task->inputs[0]);
    float* output = static_cast<float*>(task->outputs[0]);
    int size = task->dims[0];
    float scale = task->aux_float[0];

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = input[i] * scale;
    }
}

/**
 * 执行 LayerNorm 任务 (简化版)
 */
__device__ void execute_layernorm_task(MacoTaskDesc* task) {
    float* input = static_cast<float*>(task->inputs[0]);
    float* gamma = static_cast<float*>(task->inputs[1]);
    float* beta = static_cast<float*>(task->inputs[2]);
    float* output = static_cast<float*>(task->outputs[0]);

    int batch = task->dims[0];
    int hidden = task->dims[1];
    float eps = task->aux_float[0];

    // 简化实现: 每个 block 处理一行
    // 实际应该使用更优化的实现
    __shared__ float shared_mean;
    __shared__ float shared_var;

    for (int b = blockIdx.x; b < batch; b += gridDim.x) {
        float* row_in = input + b * hidden;
        float* row_out = output + b * hidden;

        // 计算均值
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            local_sum += row_in[i];
        }

        // Block reduce (简化版)
        __shared__ float reduce_buf[256];
        reduce_buf[threadIdx.x] = local_sum;
        __syncthreads();

        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int i = 0; i < blockDim.x && i < hidden; i++) {
                total += reduce_buf[i];
            }
            shared_mean = total / hidden;
        }
        __syncthreads();

        // 计算方差
        float local_var = 0.0f;
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            float diff = row_in[i] - shared_mean;
            local_var += diff * diff;
        }

        reduce_buf[threadIdx.x] = local_var;
        __syncthreads();

        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int i = 0; i < blockDim.x && i < hidden; i++) {
                total += reduce_buf[i];
            }
            shared_var = total / hidden;
        }
        __syncthreads();

        // 归一化
        float inv_std = rsqrtf(shared_var + eps);
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
            float normalized = (row_in[i] - shared_mean) * inv_std;
            row_out[i] = normalized * gamma[i] + beta[i];
        }
    }
}

/**
 * 执行 GELU 激活
 */
__device__ void execute_gelu_task(MacoTaskDesc* task) {
    float* input = static_cast<float*>(task->inputs[0]);
    float* output = static_cast<float*>(task->outputs[0]);
    int size = task->dims[0];

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float x = input[i];
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x * x * x)));
        output[i] = x * cdf;
    }
}

/**
 * 执行 SiLU (Swish) 激活
 */
__device__ void execute_silu_task(MacoTaskDesc* task) {
    float* input = static_cast<float*>(task->inputs[0]);
    float* output = static_cast<float*>(task->outputs[0]);
    int size = task->dims[0];

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

/**
 * 空操作 (用于同步)
 */
__device__ void execute_nop_task(MacoTaskDesc* task) {
    // Do nothing
    __syncthreads();
}

// ============================================================================
// 任务分发
// ============================================================================

/**
 * 执行任务 (根据任务类型分发)
 */
__device__ void execute_task(MacoTaskDesc* task) {
    switch (task->task_type) {
        case MACO_TASK_MATMUL:
            execute_matmul_task(task);
            break;

        case MACO_TASK_LINEAR:
            execute_matmul_task(task);  // Linear 本质是 MatMul
            break;

        case MACO_TASK_LAYERNORM:
            execute_layernorm_task(task);
            break;

        case MACO_TASK_GELU:
            execute_gelu_task(task);
            break;

        case MACO_TASK_SILU:
            execute_silu_task(task);
            break;

        case MACO_TASK_NOP:
            execute_nop_task(task);
            break;

        case MACO_TASK_CUSTOM_COMPUTE:
            // 用户自定义计算通过函数指针调用
            // 简化: 作为 vector scale
            execute_vector_scale_task(task);
            break;

        default:
            // 未知任务类型，跳过
            break;
    }
}

// ============================================================================
// Worker 主循环
// ============================================================================

/**
 * Worker CTA 执行函数
 *
 * 这是 Worker 的核心逻辑:
 * 1. 持续从队列获取任务
 * 2. 等待依赖
 * 3. 执行
 * 4. 触发完成事件
 *
 * @param config 运行时配置
 */
__device__ void execute_worker(MacoRuntimeConfig* config) {
    int worker_id = blockIdx.x;

    // 获取这个 Worker 的队列
    MacoTaskDesc* my_queue = config->worker_queues +
                             worker_id * WORKER_QUEUE_SIZE;

    // 本地队列位置追踪
    uint64_t local_tail = 0;  // 下一个要处理的任务位置

    // Shared memory for task desc caching
    __shared__ MacoTaskDesc cached_task;

    while (true) {
        // 1. Polling: 等待新任务
        uint64_t head;
        int spin_count = 0;
        while (true) {
            // 原子读取队列头 (Scheduler 写入的位置)
            head = ld_acquire_gpu_u64(&config->worker_queue_head[worker_id]);

            if (local_tail < head) {
                break;  // 有新任务
            }

            // 检查终止标志
            if (ld_acquire_gpu_u32(reinterpret_cast<unsigned int*>(
                    config->terminate_flag)) != 0) {
                return;
            }

            // 检查队列中是否有终止任务
            MacoTaskDesc* peek_task = &my_queue[local_tail % WORKER_QUEUE_SIZE];
            if (peek_task->task_type == MACO_TASK_TERMINATE) {
                return;
            }

            // 休眠减少竞争
            maco_nanosleep(10);

            spin_count++;
            if (spin_count > 1000000) {
                // 防止无限等待
                spin_count = 0;
            }
        }

        // 2. 获取任务 (加载到 shared memory)
        MacoTaskDesc* task = &my_queue[local_tail % WORKER_QUEUE_SIZE];

        // 由 thread 0 加载到 shared memory
        if (threadIdx.x == 0) {
            cached_task = *task;
        }
        __syncthreads();

        // 3. 检查终止任务
        if (cached_task.task_type == MACO_TASK_TERMINATE) {
            return;
        }

        // 4. 等待依赖事件
        if (cached_task.dependent_event != EVENT_INVALID_ID) {
            uint64_t event_id = cached_task.dependent_event;
            MacoEventDesc* event_desc = &config->all_events[event_id];
            uint64_t needed = event_desc->needed_triggers;

            // Polling 等待事件计数器
            while (true) {
                uint64_t current = ld_acquire_gpu_u64(
                    &config->event_counters[event_id]);
                if (current >= needed) {
                    break;
                }
                maco_nanosleep(10);
            }
        }
        __syncthreads();

        // 5. 执行任务
        execute_task(&cached_task);
        __syncthreads();

        // 6. 触发完成事件
        if (cached_task.trigger_event != EVENT_INVALID_ID && threadIdx.x == 0) {
            uint64_t event_id = cached_task.trigger_event;

            // 原子增加事件计数器
            uint64_t old_count = atom_add_release_gpu_u64(
                &config->event_counters[event_id], 1);

            // 检查是否需要通知 Scheduler
            MacoEventDesc* event_desc = &config->all_events[event_id];
            if (old_count + 1 >= event_desc->needed_triggers) {
                // 将事件加入 Scheduler 队列
                uint64_t sched_pos = atom_add_release_gpu_u64(
                    config->scheduler_queue_head, 1);
                config->scheduler_queue[sched_pos % SCHEDULER_QUEUE_SIZE] = event_id;
            }

            // 更新完成任务计数
            atomicAdd(config->tasks_completed, 1);
        }
        __syncthreads();

        // 7. 移动到下一个任务
        local_tail++;

        // 更新队列尾指针 (告诉 Scheduler 可以复用这个位置)
        if (threadIdx.x == 0) {
            st_release_gpu_u64(&config->worker_queue_tail[worker_id], local_tail);
        }
    }
}

}  // namespace maco
