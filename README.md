# PVInspect

[![travis status](https://travis-ci.com/ma0ho/pvinspect.svg?branch=master "travis status")](https://travis-ci.com/github/ma0ho/pvinspect)
[![Downloads](https://pepy.tech/badge/pvinspect)](https://pepy.tech/project/pvinspect)

This package provides methods for the analysis of solar modules using different imaging modalities. We put huge efforts into providing a clean and easy to use API and additional tooling like the wrapper classes for module images that bundle image with meta data in an expressive way. You are invited to publish you own methods using this API and hence contribute to building a consistent and open tooling that might be useful to others. If you like to integrate your methods directly, please create a merge request.

## Package status

This package is in an early stage of development. Please be aware that the API might change regularily within the next months.

## Installation

We recommend to use the Anaconda Python distribution and to create a separate environment using Python3.7:

```bash
conda create -n pvinspect pip python=3.7
```

Then you can install this package using `pip`:

```bash
conda activate pvinspect
pip install pvinspect
```

## Usage

Activate the conda environment and start your dev environment (for example Jupyter Lab):

```bash
conda activate pvinspect
jupyter lab
```

This package contains [example notebooks](examples) that demonstate the most common use cases (to be extended soon). For more details, please refer to the [API docs](https://ma0ho.github.io/pvinspect/).

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
