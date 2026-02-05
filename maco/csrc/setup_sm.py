"""
MACO SM Scheduling Extension Build Configuration

编译 MACO 的 SM 调度模块。
不依赖 NVSHMEM，只需要标准 PyTorch + CUDA。

使用方法:
    cd /mini_mirage/maco
    python maco/csrc/setup_sm.py install

    # 或者开发模式
    pip install -e .

依赖要求:
    - PyTorch >= 2.0 (with CUDA support)
    - CUDA >= 11.0 (sm_70+ for GPU atomics)
    - nvcc (CUDA compiler)

不需要:
    - NVSHMEM ❌
    - MPI ❌
    - 其他特殊硬件配置 ❌
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前目录
CSRC_DIR = Path(__file__).parent.absolute()

# 源文件
SOURCES = [str(CSRC_DIR / "maco_kernel.cu")]

# 检测 CUDA 架构
def get_cuda_arch_flags():
    """
    获取 CUDA 架构编译标志
    支持常见的 GPU 架构
    """
    arch_flags = []

    # 检查环境变量
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        # 使用用户指定的架构
        return []  # PyTorch 会自动处理

    # 默认支持的架构
    default_archs = [
        ("70", "sm_70"),   # V100
        ("75", "sm_75"),   # T4, RTX 2080
        ("80", "sm_80"),   # A100, A800
        ("86", "sm_86"),   # RTX 3090, A6000
        ("89", "sm_89"),   # L40, RTX 4090
        ("90", "sm_90"),   # H100
    ]

    for compute, sm in default_archs:
        arch_flags.append(f"-gencode=arch=compute_{compute},code={sm}")

    return arch_flags

# 编译选项
EXTRA_COMPILE_ARGS = {
    "cxx": [
        "-O3",
        "-std=c++17",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-Xcompiler", "-fPIC",
        # GPU Atomics 需要 sm_70+
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ],
}

# 定义扩展模块
ext_modules = [
    CUDAExtension(
        name="maco._sm",
        sources=SOURCES,
        include_dirs=[
            str(CSRC_DIR),
        ],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )
]

# 主设置
setup(
    name="maco-sm-scheduler",
    version="0.1.0",
    author="MACO Team",
    description="MACO SM Scheduling Extension - GPU Atomics Based Task Scheduling",
    long_description="""
MACO SM Scheduler - Mirage-style SM-level task scheduling.

Features:
- Persistent Kernel: Single launch, continuous execution
- GPU Atomics: Lock-free synchronization using PTX instructions
- Worker/Scheduler Pattern: CTA-based role assignment
- Event System: Task dependency management

This extension provides the low-level infrastructure for
compute-communication overlap optimization.

NO NVSHMEM REQUIRED - Works with standard PyTorch + CUDA.
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
