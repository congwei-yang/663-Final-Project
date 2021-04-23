import setuptools

with open("README.md", "r") as fh:
    README = fh.read()

setuptools.setup(
    name="STA-663-Sinkhorn",
    version="0.1",
    author="Congwei Yang, Haoliang Zheng, Yijia Zhang",
    author_email="yijia.zhang912@duke.edu",
    description="Optimized Sinkhorn Optimal transport algorithm",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/congwei-yang/663-Final-Project/Sinkhorn_663",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["numpy>=1.16"],
    python_requires='>=3.6'
)