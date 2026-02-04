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

NVSHMEM_HOME = get_env_path("NVSHMEM_HOME", "/usr/local/nvshmem")
MPI_HOME = get_env_path("MPI_HOME", "/usr")
CUDA_HOME = get_env_path("CUDA_HOME", "/usr/local/cuda")

# 路径配置
NVSHMEM_INC = os.path.join(NVSHMEM_HOME, "include")
NVSHMEM_LIB = os.path.join(NVSHMEM_HOME, "lib")
MPI_INC = os.path.join(MPI_HOME, "include")
MPI_LIB = os.path.join(MPI_HOME, "lib")

# 也支持从环境变量直接指定
NVSHMEM_INC = os.environ.get("NVSHMEM_INC_PATH", NVSHMEM_INC)
NVSHMEM_LIB = os.environ.get("NVSHMEM_LIB_PATH", NVSHMEM_LIB)
MPI_INC = os.environ.get("MPI_INC_PATH", MPI_INC)
MPI_LIB = os.environ.get("MPI_LIB_PATH", MPI_LIB)

# 检查 NVSHMEM 是否存在
nvshmem_header = os.path.join(NVSHMEM_INC, "nvshmem.h")
if not os.path.exists(nvshmem_header):
    print(f"=" * 60)
    print(f"ERROR: NVSHMEM not found at {NVSHMEM_HOME}")
    print(f"")
    print(f"Please install NVSHMEM first:")
    print(f"  1. Download from: https://developer.nvidia.com/nvshmem")
    print(f"  2. Extract and set: export NVSHMEM_HOME=/path/to/nvshmem")
    print(f"")
    print(f"Or set NVSHMEM_INC_PATH and NVSHMEM_LIB_PATH directly.")
    print(f"=" * 60)
    # 允许 dry-run (例如 setup.py --help)
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
        "-arch=sm_80",  # A100/A800
        "-gencode=arch=compute_80,code=sm_80",  # A100/A800
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
