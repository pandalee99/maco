"""
MACO NVSHMEM Extension Build Configuration

编译 NVSHMEM CUDA 扩展模块。

使用方法:
    # 方式 1: 安装到 site-packages
    cd /mini_mirage/maco
    python maco/csrc/setup.py install

    # 方式 2: 开发模式安装
    cd /mini_mirage/maco
    python maco/csrc/setup.py develop

    # 方式 3: 仅编译
    cd /mini_mirage/maco
    python maco/csrc/setup.py build_ext --inplace

环境变量:
    NVSHMEM_HOME: NVSHMEM 安装路径 (默认 /usr/local/nvshmem)
    MPI_HOME: MPI 安装路径 (默认 /usr)
    CUDA_HOME: CUDA 安装路径 (默认 /usr/local/cuda)

依赖要求:
    - NVSHMEM >= 3.5.19
    - MPI (OpenMPI 或 MPICH)
    - PyTorch >= 2.0 (with CUDA support)
    - nvcc (CUDA compiler)

注意事项:
    - 必须使用 -rdc=true 编译以支持 NVSHMEM 设备链接
    - 必须链接 nvshmem_host 和 nvshmem_device 库
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取环境变量或使用默认值
def get_env_path(name, default):
    path = os.environ.get(name, default)
    if not os.path.exists(path):
        print(f"Warning: {name}={path} does not exist")
    return path

# 自动检测 NVSHMEM 安装位置
def find_nvshmem():
    """查找 NVSHMEM 安装路径"""
    # 优先级: 环境变量 > pip安装 > apt安装
    candidates = [
        # 环境变量
        (os.environ.get("NVSHMEM_INC_PATH", ""), os.environ.get("NVSHMEM_LIB_PATH", "")),
        # apt 安装 (CUDA 12) - 优先使用，更稳定
        ("/usr/include/nvshmem_12", "/usr/lib/x86_64-linux-gnu/nvshmem/12"),
        # apt 安装 (CUDA 13)
        ("/usr/include/nvshmem_13", "/usr/lib/x86_64-linux-gnu/nvshmem/13"),
        # pip 安装 (nvidia-nvshmem-cu12) - 有头文件兼容问题
        ("/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/include",
         "/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib"),
        # 传统安装
        ("/usr/local/nvshmem/include", "/usr/local/nvshmem/lib"),
    ]

    for inc, lib in candidates:
        if inc and lib and os.path.exists(os.path.join(inc, "nvshmem.h")):
            print(f"[MACO] Found NVSHMEM at: {inc}")
            return inc, lib

    return None, None

NVSHMEM_INC, NVSHMEM_LIB = find_nvshmem()
MPI_INC = "/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_LIB = "/usr/lib/x86_64-linux-gnu/openmpi/lib"

# Fallback MPI paths
if not os.path.exists(MPI_INC):
    MPI_INC = "/usr/include/openmpi"
if not os.path.exists(MPI_LIB):
    MPI_LIB = "/usr/lib/x86_64-linux-gnu"

# 检查 NVSHMEM 是否存在
if NVSHMEM_INC is None:
    print("=" * 60)
    print("ERROR: NVSHMEM not found")
    print("Please install NVSHMEM: apt-get install nvshmem")
    print("=" * 60)
    if "--help" not in sys.argv and "egg_info" not in sys.argv:
        sys.exit(1)
    NVSHMEM_INC = "/tmp"  # Dummy for help

nvshmem_header = os.path.join(NVSHMEM_INC, "nvshmem.h") if NVSHMEM_INC else ""
if NVSHMEM_INC and not os.path.exists(nvshmem_header):
    print("=" * 60)
    print(f"ERROR: NVSHMEM header not found at {nvshmem_header}")
    print("Please install NVSHMEM: apt-get install nvshmem")
    print("=" * 60)
    if "--help" not in sys.argv and "egg_info" not in sys.argv:
        sys.exit(1)

# 源文件路径
CSRC_DIR = Path(__file__).parent.absolute()
SOURCES = [str(CSRC_DIR / "nvshmem_ops.cu")]

# 编译选项
EXTRA_COMPILE_ARGS = {
    "cxx": [
        "-O3",
        "-std=c++17",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_80,code=sm_80",  # A100/A800
        "-gencode=arch=compute_89,code=sm_89",  # L20/L40/RTX 4090 (Ada)
        "-gencode=arch=compute_90,code=sm_90",  # H100
        "-rdc=true",  # 必需: 用于 NVSHMEM 设备链接
        "-DUSE_NVSHMEM",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler", "-fPIC",
    ],
}

# 链接选项
EXTRA_LINK_ARGS = [
    f"-L{NVSHMEM_LIB}",
    f"-L{MPI_LIB}",
    "-lnvshmem_host",
    "-lnvshmem_device",
    "-lmpi",
    "-lcuda",
    "-lcudart",
]

# 定义扩展模块
ext_modules = [
    CUDAExtension(
        name="maco._C",
        sources=SOURCES,
        include_dirs=[
            NVSHMEM_INC,
            MPI_INC,
            str(CSRC_DIR),
        ],
        library_dirs=[
            NVSHMEM_LIB,
            MPI_LIB,
        ],
        libraries=[
            "nvshmem_host",
            "nvshmem_device",
            "mpi",
        ],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
    )
]

# 设置
setup(
    name="maco-nvshmem",
    version="0.1.0",
    author="MACO Team",
    description="MACO NVSHMEM Extension for GPU-GPU Communication",
    long_description="""
MACO (Multi-GPU Async Communication Optimizer) NVSHMEM Extension.

This extension provides NVSHMEM-based communication primitives for
efficient GPU-GPU data transfer, including:
- AllReduce (Sum)
- All-to-All 4D (for Sequence Parallel)

NVSHMEM enables direct GPU memory access without CPU involvement,
reducing communication latency compared to NCCL.
    """,
    packages=find_packages(where=str(CSRC_DIR.parent.parent)),
    package_dir={"": str(CSRC_DIR.parent.parent)},
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
    ],
)
