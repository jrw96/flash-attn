from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_v2",
    ext_modules=[
        CUDAExtension(
            "flash_attn_v2",
            ["csrc/flash_attn_v2.cu", "csrc/flash_attn_v2_binding.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
