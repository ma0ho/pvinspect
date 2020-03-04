import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'pvinspect',
    version = '0.0.1',
    author = 'Mathis Hoffmann',
    author_email = 'mathis.hoffmann@fau.de',
    description = 'Provides methods for the analysis of PV modules using different modailities',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/ma0ho/pvinspect',
    packages = setuptools.find_packages(),
    python_requires = '>=3.6, <3.8',
    install_requires = [
        'numpy',
        'scikit-image',
        'matplotlib',
        'ipywidgets',
        'pathlib',
        'tqdm',
        'pytest',
        'scikit-image',
        'scipy',
        'opencv-python',
        'requests',
        'googledrivedownloader',
        'jupyterlab',
        'opencv-python'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)