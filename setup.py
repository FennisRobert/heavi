from setuptools import setup, find_packages

setup(
    name='heavi',
    version='0.5.1',
    description='A simple python based linear circuit simulator.',
    author="Robert Fennis",
    packages=find_packages(where='src', include=["heavi", "heavi.*"]),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "numba",
        "matplotlib",
        "loguru",
        "sympy",
    ],
    lisence="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
)