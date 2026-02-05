/**
 * MACO Scheduler CTA - 任务分发单元
 *
 * Scheduler 负责根据事件触发来分发任务到 Workers。
 * 每个 Scheduler 是一个 CTA (blockIdx.x >= num_workers)。
 *
 * 工作流程:
 * 1. 等待事件到达 (通过 Scheduler 队列)
 * 2. 根据事件类型分发任务到 Workers
 * 3. 继续处理下一个事件
 *
 * 参考: Mirage persistent_kernel.cuh Line 755-1000
 */

#pragma once

#include "maco_atoms.cuh"
#include "maco_types.h"

namespace maco {

// ============================================================================
// Scheduler 主循环
// ============================================================================

/**
 * Scheduler CTA 执行函数
 *
 * Scheduler 的核心职责:
 * 1. 监听事件完成
 * 2. 将后续任务分发到空闲的 Workers
 * 3. 实现负载均衡
 *
 * @param config 运行时配置
 */
__device__ void execute_scheduler(MacoRuntimeConfig* config) {
    int scheduler_id = blockIdx.x - config->num_workers;

    // 只有 thread 0 执行调度逻辑
    if (threadIdx.x != 0) {
        // 其他线程等待，或者可以做辅助工作
        while (true) {
            if (ld_acquire_gpu_u32(reinterpret_cast<unsigned int*>(
                    config->terminate_flag)) != 0) {
                return;
            }
            maco_nanosleep(100);
        }
    }

    // 本地状态
    uint64_t local_tail = 0;  // 下一个要处理的事件位置
    int next_worker = 0;      // Round-robin 分配的下一个 Worker

    while (true) {
        // 1. 检查终止标志
        if (ld_acquire_gpu_u32(reinterpret_cast<unsigned int*>(
                config->terminate_flag)) != 0) {
            // 向所有 Workers 发送终止信号
            for (int w = 0; w < config->num_workers; w++) {
                MacoTaskDesc* queue = config->worker_queues +
                                     w * WORKER_QUEUE_SIZE;
                uint64_t head = ld_acquire_gpu_u64(
                    &config->worker_queue_head[w]);

                MacoTaskDesc* task = &queue[head % WORKER_QUEUE_SIZE];
                task->task_type = MACO_TASK_TERMINATE;
                task->task_id = -1;

                // 增加队列头
                st_release_gpu_u64(&config->worker_queue_head[w], head + 1);
            }
            return;
        }

        // 2. Polling: 等待事件到达
        uint64_t head = ld_acquire_gpu_u64(config->scheduler_queue_head);

        if (local_tail >= head) {
            // 没有新事件，休眠
            maco_nanosleep(10);
            continue;
        }

        // 3. 获取事件
        uint64_t event_id = ld_relaxed_gpu_u64(
            &config->scheduler_queue[local_tail % SCHEDULER_QUEUE_SIZE]);

        MacoEventDesc* event_desc = &config->all_events[event_id];

        // 4. 根据事件类型处理
        switch (event_desc->event_type) {
            case EVENT_TYPE_LAUNCH_TASK: {
                // 分发单个任务到指定或下一个 Worker
                int target = event_desc->target_worker;
                if (target < 0 || target >= config->num_workers) {
                    // Round-robin 分配
                    target = next_worker;
                    next_worker = (next_worker + 1) % config->num_workers;
                }

                // 检查 Worker 队列是否有空间
                uint64_t worker_head = ld_acquire_gpu_u64(
                    &config->worker_queue_head[target]);
                uint64_t worker_tail = ld_acquire_gpu_u64(
                    &config->worker_queue_tail[target]);

                if (worker_head - worker_tail >= WORKER_QUEUE_SIZE - 1) {
                    // 队列满，等待
                    maco_nanosleep(10);
                    continue;  // 重试这个事件
                }

                // 将任务加入 Worker 队列
                // 注意: 任务描述符应该已经准备好了
                // 这里只是更新队列头
                st_release_gpu_u64(&config->worker_queue_head[target],
                                  worker_head + 1);
                break;
            }

            case EVENT_TYPE_LAUNCH_MASSIVE: {
                // 分发多个任务到多个 Workers
                int num_tasks = event_desc->num_tasks;
                int first_task = event_desc->first_task_id;

                for (int i = 0; i < num_tasks; i++) {
                    // Round-robin 分配
                    int target = (next_worker + i) % config->num_workers;

                    // 检查 Worker 队列空间
                    uint64_t worker_head = ld_acquire_gpu_u64(
                        &config->worker_queue_head[target]);
                    uint64_t worker_tail = ld_acquire_gpu_u64(
                        &config->worker_queue_tail[target]);

                    while (worker_head - worker_tail >= WORKER_QUEUE_SIZE - 1) {
                        // 队列满，等待
                        maco_nanosleep(10);
                        worker_tail = ld_acquire_gpu_u64(
                            &config->worker_queue_tail[target]);
                    }

                    // 更新队列头
                    st_release_gpu_u64(&config->worker_queue_head[target],
                                      worker_head + 1);
                }

                next_worker = (next_worker + num_tasks) % config->num_workers;
                break;
            }

            case EVENT_TYPE_BARRIER: {
                // 屏障同步 - 等待所有 Workers 完成当前任务
                // 简化实现: 检查所有 Worker 队列是否为空

                bool all_empty = false;
                while (!all_empty) {
                    all_empty = true;
                    for (int w = 0; w < config->num_workers; w++) {
                        uint64_t head = ld_acquire_gpu_u64(
                            &config->worker_queue_head[w]);
                        uint64_t tail = ld_acquire_gpu_u64(
                            &config->worker_queue_tail[w]);
                        if (head != tail) {
                            all_empty = false;
                            break;
                        }
                    }
                    if (!all_empty) {
                        maco_nanosleep(10);
                    }
                }
                break;
            }

            case EVENT_TYPE_TASK_DONE: {
                // 任务完成事件 - 可能需要触发后续任务
                // 这个通常由 Worker 自己处理
                break;
            }

            default:
                break;
        }

        // 5. 移动到下一个事件
        local_tail++;
        st_release_gpu_u64(config->scheduler_queue_tail, local_tail);
    }
}

// ============================================================================
// 初始任务分发
// ============================================================================

/**
 * 初始化阶段的任务分发
 *
 * 在 Persistent Kernel 启动时，将初始任务分发到 Workers。
 * 这个函数由 Scheduler CTA 调用一次。
 *
 * @param config 运行时配置
 * @param initial_tasks 初始任务数组
 * @param num_initial_tasks 初始任务数量
 */
__device__ void distribute_initial_tasks(
    MacoRuntimeConfig* config,
    MacoTaskDesc* initial_tasks,
    int num_initial_tasks
) {
    // 只有 thread 0 执行分发
    if (threadIdx.x != 0) return;

    // Round-robin 分发初始任务
    for (int i = 0; i < num_initial_tasks; i++) {
        int target = i % config->num_workers;

        // 复制任务到 Worker 队列
        MacoTaskDesc* queue = config->worker_queues +
                             target * WORKER_QUEUE_SIZE;
        uint64_t head = ld_acquire_gpu_u64(
            &config->worker_queue_head[target]);

        queue[head % WORKER_QUEUE_SIZE] = initial_tasks[i];

        // 更新队列头
        st_release_gpu_u64(&config->worker_queue_head[target], head + 1);
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 获取最空闲的 Worker
 *
 * 返回队列中待处理任务最少的 Worker ID
 */
__device__ int get_least_loaded_worker(MacoRuntimeConfig* config) {
    int best_worker = 0;
    uint64_t min_pending = UINT64_MAX;

    for (int w = 0; w < config->num_workers; w++) {
        uint64_t head = ld_acquire_gpu_u64(&config->worker_queue_head[w]);
        uint64_t tail = ld_acquire_gpu_u64(&config->worker_queue_tail[w]);
        uint64_t pending = head - tail;

        if (pending < min_pending) {
            min_pending = pending;
            best_worker = w;
        }
    }

    return best_worker;
}

/**
 * 检查是否所有任务都已完成
 */
__device__ bool all_tasks_completed(MacoRuntimeConfig* config, int expected) {
    int completed = atomicAdd(config->tasks_completed, 0);
    return completed >= expected;
}

}  // namespace maco
