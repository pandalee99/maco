/**
 * MACO Types - 任务和事件定义
 *
 * 定义 SM 调度系统使用的数据结构。
 * 参考 Mirage 的 persistent_kernel.cuh 设计。
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace maco {

// ============================================================================
// 常量定义
// ============================================================================

// 无效事件 ID
constexpr uint64_t EVENT_INVALID_ID = 0xFFFFFFFFFFFFFFFFULL;

// 最大支持的 Worker 数量
constexpr int MAX_WORKERS = 128;

// 最大支持的 Scheduler 数量
constexpr int MAX_SCHEDULERS = 16;

// 每个 Worker 的任务队列大小
constexpr int WORKER_QUEUE_SIZE = 256;

// Scheduler 队列大小
constexpr int SCHEDULER_QUEUE_SIZE = 512;

// 最大事件数量
constexpr int MAX_EVENTS = 4096;

// ============================================================================
// 任务类型枚举
// ============================================================================

enum MacoTaskType : int {
    // 控制任务
    MACO_TASK_TERMINATE = 0,        // 终止信号
    MACO_TASK_NOP = 1,              // 空操作 (用于同步)

    // 计算任务 (100-199)
    MACO_TASK_MATMUL = 100,         // 矩阵乘法
    MACO_TASK_ATTENTION = 101,      // Flash Attention
    MACO_TASK_LINEAR = 102,         // 线性层
    MACO_TASK_LAYERNORM = 103,      // Layer Normalization
    MACO_TASK_GELU = 104,           // GELU 激活
    MACO_TASK_SILU = 105,           // SiLU 激活
    MACO_TASK_SOFTMAX = 106,        // Softmax
    MACO_TASK_CUSTOM_COMPUTE = 150, // 自定义计算 (用户扩展)

    // 通信任务 (200-299) - 使用 NCCL
    MACO_TASK_ALLREDUCE = 200,      // AllReduce
    MACO_TASK_ALLGATHER = 201,      // AllGather
    MACO_TASK_REDUCE_SCATTER = 202, // ReduceScatter
    MACO_TASK_ALL_TO_ALL = 203,     // All-to-All
    MACO_TASK_SEND = 204,           // Point-to-point Send
    MACO_TASK_RECV = 205,           // Point-to-point Recv
    MACO_TASK_CUSTOM_COMM = 250,    // 自定义通信

    // 内存任务 (300-399)
    MACO_TASK_MEMCPY = 300,         // 内存复制
    MACO_TASK_MEMSET = 301,         // 内存设置
};

// ============================================================================
// 事件类型枚举
// ============================================================================

enum MacoEventType : int {
    EVENT_TYPE_INVALID = 0,
    EVENT_TYPE_TASK_DONE = 1,           // 单个任务完成
    EVENT_TYPE_BARRIER = 2,             // 屏障同步
    EVENT_TYPE_LAUNCH_TASK = 3,         // 启动单个任务
    EVENT_TYPE_LAUNCH_MASSIVE = 4,      // 启动大批量任务
};

// ============================================================================
// 任务描述符
// ============================================================================

/**
 * 任务描述符 - 描述一个要执行的任务
 *
 * 设计原则:
 * 1. 固定大小，便于队列管理
 * 2. 包含任务类型、输入输出指针、维度信息
 * 3. 支持事件依赖和触发
 */
struct MacoTaskDesc {
    // 任务标识
    int task_type;                  // MacoTaskType
    int task_id;                    // 唯一任务 ID

    // 输入输出指针 (最多 4 个输入，2 个输出)
    void* inputs[4];
    void* outputs[2];

    // 维度信息 (根据任务类型解释)
    int dims[8];

    // 依赖事件 (在执行前等待这个事件)
    uint64_t dependent_event;       // EVENT_INVALID_ID 表示无依赖

    // 触发事件 (执行完成后触发)
    uint64_t trigger_event;         // EVENT_INVALID_ID 表示不触发

    // 辅助数据 (用于 reduction 操作类型等)
    int aux_int[4];
    float aux_float[4];
};

// ============================================================================
// 事件描述符
// ============================================================================

/**
 * 事件描述符 - 描述一个事件
 *
 * 事件用于任务间的依赖管理:
 * - 任务完成时触发事件 (增加 counter)
 * - 任务等待事件 (轮询 counter)
 */
struct MacoEventDesc {
    int event_type;                 // MacoEventType
    int event_id;                   // 唯一事件 ID

    // 触发次数 (需要多少次触发才算完成)
    int needed_triggers;

    // 关联的任务信息
    int num_tasks;                  // 此事件关联的任务数
    int first_task_id;              // 第一个任务 ID
    int target_worker;              // 目标 Worker ID (-1 表示任意)
    int target_scheduler;           // 目标 Scheduler ID
};

// ============================================================================
// 运行时配置
// ============================================================================

/**
 * SM 调度运行时配置
 *
 * 这个结构体传递给 Persistent Kernel，包含所有必要的指针和配置。
 */
struct MacoRuntimeConfig {
    // Worker 配置
    int num_workers;                // Worker CTA 数量
    int num_schedulers;             // Scheduler 数量 (通常 = 1)
    int threads_per_worker;         // 每个 Worker 的线程数

    // Worker 任务队列
    // 布局: [num_workers][WORKER_QUEUE_SIZE]
    MacoTaskDesc* worker_queues;

    // 队列控制指针 (用于 lock-free 操作)
    // worker_queue_head[i]: Worker i 的下一个可写位置 (Scheduler 写)
    // worker_queue_tail[i]: Worker i 的下一个可读位置 (Worker 读)
    uint64_t* worker_queue_head;
    uint64_t* worker_queue_tail;

    // 事件系统
    MacoEventDesc* all_events;      // 所有事件描述符
    uint64_t* event_counters;       // 事件计数器 [MAX_EVENTS]
    int num_events;

    // Scheduler 队列 (事件触发后通知 Scheduler)
    uint64_t* scheduler_queue;      // [SCHEDULER_QUEUE_SIZE]
    uint64_t* scheduler_queue_head;
    uint64_t* scheduler_queue_tail;

    // 全局状态
    int* terminate_flag;            // 终止标志
    int* tasks_completed;           // 已完成任务计数

    // 设备信息
    int device_id;
    int sm_count;
};

// ============================================================================
// 任务图节点 (Python 端使用)
// ============================================================================

/**
 * 任务图节点 - 用于在 Python 端构建任务图
 */
struct TaskGraphNode {
    int node_id;
    int task_type;

    // 依赖关系
    int num_dependencies;
    int dependencies[8];            // 依赖的节点 ID

    // 生成的事件 ID
    int generated_event;
};

// ============================================================================
// 辅助结构
// ============================================================================

/**
 * Tensor 描述符 - 描述一个张量的形状和数据类型
 */
struct TensorDesc {
    void* data_ptr;
    int dtype;                      // 0=float32, 1=float16, 2=bfloat16
    int ndim;
    int shape[8];
    int strides[8];
};

/**
 * 通信配置 - 用于通信任务
 */
struct CommConfig {
    int world_size;
    int rank;
    int group_id;                   // -1 表示默认 group
    int reduction_op;               // 0=sum, 1=avg, 2=min, 3=max
};

}  // namespace maco
