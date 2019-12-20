# PVInspect

![travis status](https://travis-ci.com/ma0ho/pvinspect.svg?branch=master "travis status")

This package provides methods for the analysis of solar modules using different imaging modalities. We put huge efforts into providing a clean and easy to use API and additional tooling like the wrapper classes for module images that bundle image with meta data in an expressive way. You are invited to publish you own methods using this API and hence contribute to building a consistent and open tooling that might be useful to others. If you like to integrate your methods directly, please create a merge request.

## Package status

This package is in an early stage of development. Please be aware that the API might change regularily within the next months.

## Installation

We recommend to use the Anaconda Python distribution and to create a separate environment using Python3.7:
```
conda create -n pvinspect pip python=3.7
```

Then you can install this package using `pip`:
```
conda activate pvinspect
pip install pvinspect
```

## Usage

Activate the conda environment and start your dev environment (for example Jupyter Lab):
```
conda activate pvinspect
jupyter lab
```

This package contains [example notebooks](examples) that demonstate the most common use cases (to be extended soon). In addition, you may refer to the TODO: [TODO](API docs).