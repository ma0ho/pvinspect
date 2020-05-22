import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
import sys
import os
from subprocess import check_call


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        # we need to explicitly install shapely using conda in a conda environment on windows
        if sys.platform == "win32" and os.getenv("CONDA_PREFIX", ""):
            check_call("conda install -y shapely".split())


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        # we need to explicitly install shapely using conda in a conda environment on windows
        if sys.platform == "win32" and os.getenv("CONDA_PREFIX", ""):
            check_call("conda install -y shapely".split())


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pvinspect",
    version="0.2.0",
    author="Mathis Hoffmann",
    author_email="mathis.hoffmann@fau.de",
    description="Provides methods for the analysis of PV modules using different modailities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ma0ho/pvinspect",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <3.9",
    install_requires=[
        "numpy >= 1.16.0",
        "scikit-image",
        "matplotlib",
        "pathlib",
        "tqdm",
        "pytest",
        "scikit-image",
        "scipy",
        "opencv-python",
        "requests",
        "googledrivedownloader",
        "opencv-python",
        "shapely",
        "optuna",
        "sklearn",
        "pre-commit",
        "pillow < 7.1.0",
        "pandas >= 1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    cmdclass={"develop": PostDevelopCommand, "install": PostInstallCommand,},
)
