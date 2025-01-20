from setuptools import setup, find_packages

setup(
    name='heavi',
    version='0.1.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "numba",
        "matplotlib",
        "loguru",
        "sympy",
    ],
    python_requires=">=3.13",
)