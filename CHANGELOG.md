# Changelog

## 0.4.1

### Fixed

* Revert to detection-based stitching, since feature-based stitching turned out instable

## 0.4.0

### New

* ORB-based stitching of solar modules added

## 0.3.3

### New

* `save_images` now has an additional argument `filename_hook` that allows to modify the generation of image filenames
* Meta data is now cached to speed up loading of long sequences
* `ClassificationDataset.result_sequence` now accepts results of type `torch.float` as well

### Fixed

* `PartialModuleImage.cols` and `PartialModuleImage.rows` is now set correctly with `segment_module_part` and negative `first_col`/`first_row`
* Fix semantic of `first_row` `first_col` in combination with module segmentation

## 0.3.2

### New

* Add support for uint8 images

### Fixed

* Fix occasionally failing checkerboard detection

## 0.3.1

### Fixed/Improved

* Add missing dependency

## 0.3.0

### New

* Add performance test for single module detection
* Pandas methods are now (all) available from `ImageSequence.pandas`
* `Image` now supports lazy loading of data (which is now default)
* Introduced convenience method `from_self` to avoid using `type(self).from_other(self, ...)`
* Add PyTorch `Dataset` and `ClassificationDataset` implementation
* Add ELPV cell image dataset

### Fixed/Improved

* Speed up meta data handling

## 0.2.3

* Prefix/suffix for filenames in `save_images`

## 0.2.2

* Improve robustness of reference detection
* Fix plot with visualization for more than one image
* Add `ImageSequence.drop_duplicates`
* Remove unused `copy` argument to `CellImageSequence`
* Store reference to original for `CellImage` with the correct cell marked
* Avoid passing geometric entities to segmented modules
* Save image meta in separate JSON file
* Clip image after FF calibration if it exceeds datatype limits and issue a warning
* Refactor modality such that it is an Enum
* All Image attributes (except data) are meta data now

## 0.2.1

* Fix broken Windows installation

## 0.2.0

* Remove unnecessary copy operations: `Image.data` now always returns an immutable view of the original data. The
  same holds for accessing meta attributes that are `np.ndarray`
* Change semantics of `ImageSequence.apply_image_data`: This method does not modify the original data anymore. Instead
  it returns a copy of the original data
* Shorten too long image titles
* Do no override title for `CellImage` (fixes #2)
* Allow to disable clipping result of flat-field compensation
* Add pandas compability for Image meta
* Automatic extraction of meta data from image path
* Automatic scaling of image intensities using reference cell
* Hierarchical save based on image meta

## 0.1.6

* Implement `preproc.calibration.Calibration` to handle all calibration in a single object
* Sanity check for failed checkerboard detections

## 0.1.5

* Cleanup too verbose imports
* Introduce `DType` enum for uniform typing within the package
* Add `force_dtype` argument to methods from `io`
* Add `list_meta`-method to `Image`

## 0.1.4

* Fix broken pillow version

## 0.1.3

* Refactor tests and introduce some integration tests
* Add `__version__`

## 0.1.2

* Fix missing meta argument to `CellImage`

## 0.1.1

* Fix multi module overlay
* Fix missing modality in multi module dataset
* Fix dropped meta data in `segment_module_part`, `segment_module` and `segment_cells`

## 0.1.0

* Rework + cleanup module detection
* Allow detection of partial modules
  * includes new module type `PartialModuleImage`
  * includes new reading methods `read_partial_module_image`and `read_partial_module_images`
* Automatic clipping of outliers in viewer
  * includes `clip_low` and `clip_high` arguments in viewing methods
* Add `save_image` to allow saving a single image
* Add `segment_module_part` as a more flexible variant of `segment_module` and `segment_cell`
* Add `from_other` method to `Image` and `ImageSequence` to allow partial overwriting of attributes
* Add `as_type` to `Image` and `ImageSequence` to allow datatype conversions with little overhead
* Add `apply_image_data` to `ImageSequence` to enable image manipulation over sequences of images
* Add `calibrate_flatfield` and `compensate_flatfield`
* Add `calibrate_distortion`and `compensate_distortion`
* Implement elementwise operators on Images and ImageSequences
* Implement `locate_multiple_modules`
* Support Python 3.8

## 0.0.1

* Initial version
* Includes module detection
