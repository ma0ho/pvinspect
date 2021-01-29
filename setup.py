import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pvinspect",
    version="0.4.0",
    author="Mathis Hoffmann",
    author_email="mathis.hoffmann@fau.de",
    description="Provides methods for the analysis of PV modules using different modalities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ma0ho/pvinspect",
    packages=setuptools.find_packages(),
    python_requires=">=3.7, <3.9",
    install_requires=[
        "numpy >= 1.16.5",
        "scikit-image >= 0.16",
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
        "pillow >= 7.2.0",
        "pandas >= 1.0.0",
        "onnxruntime",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
