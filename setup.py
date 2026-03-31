from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            "naive_attn",
            ["csrc/naive_attn.cu", "python/naive_attn_binding.cpp"],
            libraries=["cublas"],
        ),
        CUDAExtension(
            "flash_attn",
            ["csrc/flash_attn.cu", "python/flash_attn_binding.cpp"],
        ),
        CUDAExtension(
            "flash_attn_v2",
            ["csrc/flash_attn_v2.cu", "python/flash_attn_v2_binding.cpp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
