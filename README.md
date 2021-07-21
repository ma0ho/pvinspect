# PVInspect

[![travis status](https://travis-ci.com/ma0ho/pvinspect.svg?branch=master "travis status")](https://travis-ci.com/github/ma0ho/pvinspect)
[![Downloads](https://pepy.tech/badge/pvinspect)](https://pepy.tech/project/pvinspect)

This package provides methods for the analysis of solar modules using different imaging modalities. We put huge efforts into providing a clean and easy to use API and additional tooling like the wrapper classes for module images that bundle image with meta data in an expressive way. You are invited to publish you own methods using this API and hence contribute to building a consistent and open tooling that might be useful to others. If you like to integrate your methods directly, please create a merge request.

## Package status

This package is in an early stage of development. Please be aware that the API might change regularily within the next months.

## Installation

We recommend to install `pvinspect` in a Python virtual environment using Python 3.8. This document only supplies a short description on the usage of virtual environments in Python. For a complete guide, please refer to [this](https://realpython.com/python-virtual-environments-a-primer/) tutorial.

**0. Check that you have the correct Python version available:**

```bash
python --version
```

Depending on the OS and configuration of your machine, `python` needs to be substituted with `python3`, `python3.exe` or even the absolute path to the Python interpreter. In any case, the displayed version should be compatible to `pvinspect`. At the moment, `pvinspect` requires `Python 3.7` or `Python 3.8`.

**1. Create a new virtual environment:**

```bash
python -m venv /path/to/your/venv
```

This initializes a new virtual environement in `/path/to/your/venv` (which you should adapt to your needs). All packages that you install within this environment are stored in this folder. Depending on the OS and configuration of your machine, `python` needs to be substituted with `python3` or `python3.exe`.

**2. Activate the new virtual environment:**

```bash
# For Windows users (Powershell):
/path/to/your/venv/Scripts/Activate.ps1

# For Linux users:
/path/to/your/vent/Scripts/activate
```

Please note that the procedure depends on the type of shell that you are using. We've given example for Powershell and Linux-shell users.

**3. Install pvinspect:**

```bash
pip install pvinspect
```

This installs the `pvinspect` package in the currently active environment (which is normally indicated in the shell). So please make sure to have the correct environment activated. After installation, you can use `pvinspect` at any time from within that environment.

### A note to Anaconda users

We do not ship `pvinspect` as an Anaconda package. However, you can of course install it in an Anaconda environment using pip. However, you might come across an error indicating that `geos_c.dll` is missing. In that case, you need to install `shapely` using `conda`:

```bash
conda install -c conda-forge shapely 
```

## Usage

This package contains [example notebooks](examples) that demonstate the most common use cases (to be extended soon). For more details, please refer to the [API docs](https://ma0ho.github.io/pvinspect/).

## Update

You can update `pvinspect` to the newest version simply using `pip`:

```bash
pip install --upgrade pvinspect
```

## Issues

In case you encounter anything that does not work, please [open an issue](https://github.com/ma0ho/pvinspect/issues/new) and provide a precise description (include your OS version, python distribution and the like) as well as a minimal code example to reproduce the issue.

### Known issues

In case you install `pvinspect` in a conda environment using `pip` on Windows, the `shapely` library, which is installed as a dependency of `pvinspect` does not find `geos.dll`. This can be fixed by installing `shapely` using conda:

```bash
conda install shapely
```

This is reported as a bug to shapely: [#1032](https://github.com/Toblerity/Shapely/issues/1032)

## Citations

In case you use methods from this package for research purposes, please make sure to cite the the corresponding papers correctly. Please refer to the [documentation](https://ma0ho.github.io/pvinspect/) for the correct references.

## Acknowledgement

We greatly acknowledge the [HI-ERN](http://www.hi-ern.de/hi-ern/EN/home.html) for providing us a demo dataset of electroluminescense images that is published along with this package.

We gratefully acknowledge funding of the Federal Ministry for Economic Affairs and Energy (BMWi: Grant No. 0324286, iPV4.0).
