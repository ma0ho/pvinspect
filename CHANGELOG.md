# Changelog

## In development

* Refactor tests and introduce some integration tests

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
