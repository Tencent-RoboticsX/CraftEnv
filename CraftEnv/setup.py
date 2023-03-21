from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

setup(
    name="craftenv",
    version="0.1",
    description="The CraftEnv MARL environment for CRC",
    keywords="Robotics, Reinforcement Learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.5, <4",
    install_requires=[
        "gym",
        "numpy",
        "scipy",
        "wheel",
        "pybullet",
        "absl-py",
        "mpi4py",
        "torch",
        "scipy",
        "cloudpickle",
        "pandas",
        "matplotlib"
    ],
)

print(find_packages(where="src"))
