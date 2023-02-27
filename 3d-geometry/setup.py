from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="HitnetModule",
    ext_modules=[
        CUDAExtension(
            "plane_fitting_c",
            [
                "src/plane_fitting/plane_fitting.cpp",
                "src/plane_fitting/plane_fitting_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    package_dir={"": "src"},
    packages=["HitnetModule"],
)
