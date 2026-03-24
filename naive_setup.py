from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="naive_attn",
    ext_modules=[
        CUDAExtension(
            name="naive_attn",
            sources=["naive_attn_binding.cpp", "naive_attn.cu"],
            libraries=["cublas"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
