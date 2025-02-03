from setuptools import setup, find_packages

setup(
    name='heavi',
    version='0.2.2',
    packages=find_packages(where='src', include=["heavi", "heavi.*"]),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "numba",
        "matplotlib",
        "loguru",
        "sympy",
    ],
    python_requires=">=3.10",
)