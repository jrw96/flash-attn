from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="flash_attn",
    ext_modules=[
        CUDAExtension(
            name="flash_attn",
            sources=["csrc/flash_attn_binding.cpp", "csrc/flash_attn.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
