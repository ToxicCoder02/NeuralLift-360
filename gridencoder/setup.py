import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '-gencode', 'arch=compute_75,code=sm_75',  # Update the compute capability based on your GPU
    '-std=c++17'
]

c_flags = ['/O2', '/std:c++17']

ext_modules = [
    CUDAExtension(
        name='grid_encoder',
        sources=[
            'src/gridencoder.cu',
            'src/bindings.cpp',
        ],
        extra_compile_args={
            'cxx': c_flags,
            'nvcc': nvcc_flags,
        },
        include_dirs=[
            os.path.join(os.getenv('CUDA_PATH'), 'include'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include'),
        ]
    )
]

if __name__ == '__main__':
    from setuptools import setup
    setup(
        name='grid_encoder',
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension},
    )
