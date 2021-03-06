import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open("README.md", "r") as fh:
    README = fh.read()

ext_module = [Pybind11Extension(
        "skh_cpp",
        ["sinkhorn_663/skh_cpp.cpp"],
        include_dirs = ['sinkhorn_663', pybind11.get_include()]
        )]

setuptools.setup(
    name="sinkhorn_663",
    version="0.4",
    author="Congwei Yang, Haoliang Zheng, Yijia Zhang",
    author_email="congwei.yang@duke.edu, haoliang.zheng@duke.edu, yijia.zhang912@duke.edu",
    description="Optimized Sinkhorn Optimal transport algorithm",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/congwei-yang/663-Final-Project/sinkhorn_663",
    packages=setuptools.find_packages(),
    ext_modules = ext_module,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"build_ext": build_ext},
    setup_requires=["numpy>=1.16", "cppimport", "numba", "pybind11>=2.6.1"],
    python_requires='>=3.6'
)
