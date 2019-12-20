import setuptools

setuptools.setup(
    name = 'pvinspect',
    version = '0.0.1',
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
        'jupyterlab'
    ]
)