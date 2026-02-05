/**
 * NVSHMEM Operations for MACO
 *
 * 实现 NVSHMEM 版本的 allreduce 和 all_to_all_4D。
 * 参考 Mirage 的 NVSHMEM 实现模式 (tasks/ampere/allreduce.cuh)。
 *
 * NVSHMEM 核心 API:
 * - nvshmemx_putmem_nbi_block: 非阻塞 put (GPU 直连内存传输)
 * - nvshmem_quiet: 等待所有 put 完成
 * - nvshmemx_signal_op: 信号操作 (用于同步)
 * - nvshmem_barrier_all: 全局栅栏
 *
 * 编译要求:
 * - NVSHMEM >= 3.5.19
 * - MPI (用于 bootstrap)
 * - nvcc -rdc=true (用于设备链接)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// NVSHMEM 头文件
#include <nvshmem.h>
#include <nvshmemx.h>

// MPI 头文件 (用于 NVSHMEM bootstrap)
#include <mpi.h>

#include <vector>
#include <iostream>
#include <stdexcept>

// ============================================================================
// 全局状态
// ============================================================================

static bool g_nvshmem_initialized = false;
static int g_my_pe = -1;
static int g_n_pes = -1;

// 对称内存缓冲区 (用于 NVSHMEM 通信)
static void* g_symmetric_buffer = nullptr;
static size_t g_symmetric_buffer_size = 0;

// 信号数组 (用于同步)
static uint64_t* g_signal_array = nullptr;

// ============================================================================
// 辅助函数
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define NVSHMEM_CHECK(call) do { \
    int ret = call; \
    if (ret != 0) { \
        throw std::runtime_error("NVSHMEM error: " + std::to_string(ret)); \
    } \
} while(0)

/**
 * 确保对称内存缓冲区足够大
 */
static void ensure_symmetric_buffer(size_t required_size) {
    if (g_symmetric_buffer_size >= required_size) {
        return;
    }

    // 释放旧缓冲区
    if (g_symmetric_buffer != nullptr) {
        nvshmem_free(g_symmetric_buffer);
    }

    // 分配新的对称内存
    // 注意: nvshmem_malloc 分配的内存在所有 PE 上有相同的虚拟地址
    g_symmetric_buffer = nvshmem_malloc(required_size);
    if (g_symmetric_buffer == nullptr) {
        throw std::runtime_error("Failed to allocate NVSHMEM symmetric memory");
    }

    g_symmetric_buffer_size = required_size;
}

// ============================================================================
// NVSHMEM 初始化/清理
// ============================================================================

/**
 * 初始化 NVSHMEM
 *
 * 使用 MPI 作为 bootstrap 机制。
 * 必须在 MPI 初始化之后调用。
 */
void nvshmem_init_wrapper() {
    if (g_nvshmem_initialized) {
        return;
    }

    // 初始化 MPI (如果尚未初始化)
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(nullptr, nullptr);
    }

    // 获取 MPI rank 用于设置 CUDA 设备
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // 设置 CUDA 设备
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = mpi_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    // 使用 MPI communicator 初始化 NVSHMEM
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // 保存 PE 信息
    g_my_pe = nvshmem_my_pe();
    g_n_pes = nvshmem_n_pes();

    // 分配信号数组 (用于同步)
    g_signal_array = (uint64_t*)nvshmem_malloc(g_n_pes * sizeof(uint64_t));
    if (g_signal_array == nullptr) {
        throw std::runtime_error("Failed to allocate NVSHMEM signal array");
    }
    CUDA_CHECK(cudaMemset(g_signal_array, 0, g_n_pes * sizeof(uint64_t)));

    // 同步所有 PE
    nvshmem_barrier_all();

    g_nvshmem_initialized = true;

    if (g_my_pe == 0) {
        std::cout << "[MACO NVSHMEM] Initialized with " << g_n_pes
                  << " PEs on device " << device_id << std::endl;
    }
}

/**
 * 清理 NVSHMEM 资源
 */
void nvshmem_finalize_wrapper() {
    if (!g_nvshmem_initialized) {
        return;
    }

    // 释放对称内存
    if (g_symmetric_buffer != nullptr) {
        nvshmem_free(g_symmetric_buffer);
        g_symmetric_buffer = nullptr;
        g_symmetric_buffer_size = 0;
    }

    if (g_signal_array != nullptr) {
        nvshmem_free(g_signal_array);
        g_signal_array = nullptr;
    }

    nvshmem_finalize();
    g_nvshmem_initialized = false;
}

/**
 * 获取当前 PE ID
 */
int nvshmem_my_pe_wrapper() {
    if (!g_nvshmem_initialized) {
        throw std::runtime_error("NVSHMEM not initialized");
    }
    return g_my_pe;
}

/**
 * 获取总 PE 数量
 */
int nvshmem_n_pes_wrapper() {
    if (!g_nvshmem_initialized) {
        throw std::runtime_error("NVSHMEM not initialized");
    }
    return g_n_pes;
}

/**
 * 全局栅栏同步
 */
void nvshmem_barrier_wrapper() {
    if (!g_nvshmem_initialized) {
        throw std::runtime_error("NVSHMEM not initialized");
    }
    nvshmem_barrier_all();
}

// ============================================================================
// AllReduce 内核
// ============================================================================

/**
 * NVSHMEM AllReduce (Sum) 内核
 *
 * 实现思路 (Ring AllReduce):
 * 1. 每个 PE 将数据切分成 n_pes 份
 * 2. 使用 ring pattern 进行 reduce-scatter
 * 3. 使用 ring pattern 进行 allgather
 *
 * 简化实现 (适用于小数据):
 * 1. 每个 PE 将自己的数据 put 到所有其他 PE 的 buffer 中
 * 2. 每个 PE 在本地执行 reduction
 */
template<typename T>
__global__ void nvshmem_allreduce_sum_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    T* __restrict__ buffer,  // 对称内存缓冲区 [n_pes, count]
    size_t count,
    int n_pes,
    int my_pe
) {
    // Grid-stride loop
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += gridDim.x * blockDim.x) {

        // 将本地数据复制到自己的 buffer 槽位
        buffer[my_pe * count + idx] = input[idx];
    }

    // 同步确保本地复制完成
    __syncthreads();

    // 只由第一个线程块执行 NVSHMEM put
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // 将本地数据 put 到所有其他 PE 的 buffer 中
        for (int pe = 0; pe < n_pes; pe++) {
            if (pe != my_pe) {
                // 非阻塞 put: 将 input 发送到 pe 的 buffer[my_pe * count]
                nvshmemx_putmem_nbi_block(
                    buffer + my_pe * count,  // 目标地址 (在 pe 上)
                    input,                    // 源地址 (本地)
                    count * sizeof(T),        // 大小
                    pe                        // 目标 PE
                );
            }
        }

        // 等待所有 put 完成
        nvshmem_quiet();
    }

    // 同步所有 PE
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_barrier_all();
    }

    // 等待 barrier 完成
    __syncthreads();

    // 本地 reduction
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < count;
         idx += gridDim.x * blockDim.x) {

        T sum = T(0);
        for (int pe = 0; pe < n_pes; pe++) {
            sum += buffer[pe * count + idx];
        }
        output[idx] = sum;
    }
}

/**
 * 优化版 AllReduce (使用 Ring 算法)
 *
 * 这个版本更适合大数据量，减少通信开销
 */
template<typename T>
__global__ void nvshmem_allreduce_ring_kernel(
    T* __restrict__ data,
    T* __restrict__ send_buf,
    T* __restrict__ recv_buf,
    size_t chunk_size,
    int n_pes,
    int my_pe,
    int step,
    bool is_reduce_scatter
) {
    int send_to = (my_pe + 1) % n_pes;
    int recv_from = (my_pe - 1 + n_pes) % n_pes;

    int send_chunk = (my_pe - step + n_pes) % n_pes;
    int recv_chunk = (my_pe - step - 1 + n_pes) % n_pes;

    size_t send_offset = send_chunk * chunk_size;
    size_t recv_offset = recv_chunk * chunk_size;

    // Grid-stride loop
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < chunk_size;
         idx += gridDim.x * blockDim.x) {

        send_buf[idx] = data[send_offset + idx];
    }

    __syncthreads();

    // NVSHMEM put
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmemx_putmem_nbi_block(
            recv_buf,
            send_buf,
            chunk_size * sizeof(T),
            send_to
        );
        nvshmem_quiet();
        nvshmem_barrier_all();
    }

    __syncthreads();

    // Reduce or copy
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < chunk_size;
         idx += gridDim.x * blockDim.x) {

        if (is_reduce_scatter) {
            data[recv_offset + idx] += recv_buf[idx];
        } else {
            data[recv_offset + idx] = recv_buf[idx];
        }
    }
}

/**
 * AllReduce Sum 实现
 */
torch::Tensor nvshmem_allreduce_sum(torch::Tensor input) {
    if (!g_nvshmem_initialized) {
        throw std::runtime_error("NVSHMEM not initialized");
    }

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    size_t count = input.numel();

    // 需要的对称内存: n_pes * count * sizeof(element)
    size_t buffer_size = g_n_pes * count * input.element_size();
    ensure_symmetric_buffer(buffer_size);

    // 启动参数
    int threads = 256;
    int blocks = std::min((int)((count + threads - 1) / threads), 1024);

    // 只支持 float32 以避免 NVSHMEM 头文件兼容性问题
    // (half/bfloat16 在 NVSHMEM 3.5 的 reduce.cuh 中有比较运算符问题)
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "nvshmem_allreduce_sum", [&] {
            nvshmem_allreduce_sum_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                static_cast<scalar_t*>(g_symmetric_buffer),
                count,
                g_n_pes,
                g_my_pe
            );
        }
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// All-to-All 4D 内核
// ============================================================================

/**
 * All-to-All 4D 内核
 *
 * 用于 Sequence Parallel 的维度转换:
 * - SP -> HP: [B, S/P, H, D] -> [B, S, H/P, D]
 * - HP -> SP: [B, S, H/P, D] -> [B, S/P, H, D]
 *
 * 实现思路:
 * 1. 将 scatter_dim 按 n_pes 切分
 * 2. 将第 i 块 put 到 PE i
 * 3. 每个 PE 在 gather_dim 上拼接收到的数据
 */
template<typename T>
__global__ void nvshmem_all_to_all_4D_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    T* __restrict__ buffer,
    int batch,
    int dim1,  // scatter_dim 的大小
    int dim2,  // gather_dim 的大小
    int dim3,  // last dim
    int scatter_dim,
    int gather_dim,
    int n_pes,
    int my_pe
) {
    // 计算每个 PE 负责的块大小
    int scatter_chunk = dim1 / n_pes;
    int gather_chunk = dim2 / n_pes;

    // 总元素数
    size_t total_elements = (size_t)batch * dim1 * dim2 * dim3;
    size_t chunk_elements = total_elements / n_pes;

    // 将数据重排并发送
    // 这里简化处理，假设 scatter_dim=1, gather_dim=2 (即 seq 和 heads)

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < chunk_elements;
         idx += gridDim.x * blockDim.x) {

        // 计算源索引和目标索引
        // 源: [b, s_local, h, d] where s_local = s / n_pes
        // 目标 PE p 的数据来自 h 维度的第 p 块

        int d = idx % dim3;
        int h = (idx / dim3) % dim2;
        int s = (idx / (dim3 * dim2)) % scatter_chunk;
        int b = idx / (dim3 * dim2 * scatter_chunk);

        // 确定这个元素应该发送到哪个 PE (基于 h)
        int target_pe = h / gather_chunk;
        int local_h = h % gather_chunk;

        // 在 buffer 中的位置
        size_t buf_idx = ((size_t)b * scatter_chunk * gather_chunk * dim3 +
                          s * gather_chunk * dim3 +
                          local_h * dim3 + d);

        // 复制到发送缓冲区
        buffer[target_pe * chunk_elements + buf_idx] = input[idx];
    }

    __syncthreads();

    // NVSHMEM 交换
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int pe = 0; pe < n_pes; pe++) {
            if (pe != my_pe) {
                // 将发给 pe 的数据 put 过去
                nvshmemx_putmem_nbi_block(
                    buffer + my_pe * chunk_elements,
                    buffer + pe * chunk_elements,
                    chunk_elements * sizeof(T),
                    pe
                );
            }
        }
        nvshmem_quiet();
        nvshmem_barrier_all();
    }

    __syncthreads();

    // 从缓冲区组装输出
    // 输出: [b, s_full, h_local, d]
    int out_dim2 = dim2 / n_pes;           // h/n_pes

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < chunk_elements;
         idx += gridDim.x * blockDim.x) {

        int d = idx % dim3;
        int h_local = (idx / dim3) % out_dim2;
        int s = (idx / (dim3 * out_dim2)) % (dim1);
        int b = idx / (dim3 * out_dim2 * dim1);

        // 这个元素来自哪个 PE
        int src_pe = s / scatter_chunk;
        int local_s = s % scatter_chunk;

        size_t buf_idx = ((size_t)b * scatter_chunk * out_dim2 * dim3 +
                          local_s * out_dim2 * dim3 +
                          h_local * dim3 + d);

        output[idx] = buffer[src_pe * chunk_elements + buf_idx];
    }
}

/**
 * All-to-All 4D 实现
 */
torch::Tensor nvshmem_all_to_all_4D(
    torch::Tensor input,
    int scatter_dim,
    int gather_dim
) {
    if (!g_nvshmem_initialized) {
        throw std::runtime_error("NVSHMEM not initialized");
    }

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(scatter_dim >= 0 && scatter_dim < 4, "Invalid scatter_dim");
    TORCH_CHECK(gather_dim >= 0 && gather_dim < 4, "Invalid gather_dim");
    TORCH_CHECK(scatter_dim != gather_dim, "scatter_dim and gather_dim must be different");

    auto sizes = input.sizes();
    int batch = sizes[0];
    int dim1 = sizes[scatter_dim];
    int dim2 = sizes[gather_dim];
    int dim3 = sizes[3];

    TORCH_CHECK(dim1 % g_n_pes == 0,
                "scatter_dim size must be divisible by n_pes");
    TORCH_CHECK(dim2 % g_n_pes == 0,
                "gather_dim size must be divisible by n_pes");

    // 计算输出形状
    std::vector<int64_t> out_sizes(4);
    out_sizes[0] = batch;
    out_sizes[scatter_dim] = dim1;  // 扩展 scatter dim
    out_sizes[gather_dim] = dim2 / g_n_pes;  // 收缩 gather dim
    out_sizes[3] = dim3;

    auto output = torch::empty(out_sizes, input.options());

    // 需要的对称内存
    size_t buffer_size = g_n_pes * input.numel() * input.element_size();
    ensure_symmetric_buffer(buffer_size);

    // 启动参数
    size_t count = input.numel();
    int threads = 256;
    int blocks = std::min((int)((count + threads - 1) / threads), 1024);

    // 只支持 float32 以避免 NVSHMEM 头文件兼容性问题
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "nvshmem_all_to_all_4D", [&] {
            nvshmem_all_to_all_4D_kernel<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                static_cast<scalar_t*>(g_symmetric_buffer),
                batch,
                dim1,
                dim2,
                dim3,
                scatter_dim,
                gather_dim,
                g_n_pes,
                g_my_pe
            );
        }
    );

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// PyBind11 模块定义
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MACO NVSHMEM Operations";

    // 初始化/清理
    m.def("nvshmem_init", &nvshmem_init_wrapper,
          "Initialize NVSHMEM with MPI bootstrap");
    m.def("nvshmem_finalize", &nvshmem_finalize_wrapper,
          "Finalize NVSHMEM");

    // PE 信息
    m.def("nvshmem_my_pe", &nvshmem_my_pe_wrapper,
          "Get current PE ID");
    m.def("nvshmem_n_pes", &nvshmem_n_pes_wrapper,
          "Get total number of PEs");

    // 同步
    m.def("nvshmem_barrier", &nvshmem_barrier_wrapper,
          "NVSHMEM global barrier");

    // 通信原语
    m.def("nvshmem_allreduce_sum", &nvshmem_allreduce_sum,
          "NVSHMEM AllReduce Sum",
          py::arg("input"));

    m.def("nvshmem_all_to_all_4D", &nvshmem_all_to_all_4D,
          "NVSHMEM All-to-All 4D for Sequence Parallel",
          py::arg("input"),
          py::arg("scatter_dim"),
          py::arg("gather_dim"));
}
