"""Installation script for the 'kick_task' Python package."""

from setuptools import find_packages, setup

setup(
    name="kick_task",
    version="0.1.0",
    author="Ethan Li",
    description="G1 kick skill task with AMP for IsaacLab",
    license="BSD-3-Clause",
    install_requires=[],
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
    packages=find_packages(),
)
