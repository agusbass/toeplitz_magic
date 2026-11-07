from setuptools import setup, find_packages

setup(
    name="toeplitz-magic-attn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch>=2.0"],
    python_requires=">=3.9",
    author="[Anda]",
    description="Toeplitz + Magic Square positional encoding for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
