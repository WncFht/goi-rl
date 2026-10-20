# setup.py
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lidar_gen_cuda',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='lidar_gen_cuda',
            sources=[
                'lidar_gen_cuda.cpp',
                'lidar_gen_kernel.cu'
            ],
            extra_compile_args={
                'nvcc': ['-O3', '--use_fast_math', '-Xptxas=-v']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.10',
    install_requires=['torch==2.4.1+cu124'],  # 确保依赖 torch
)

#  pip install -e .\lidar_gen_cuda --no-build-isolation
