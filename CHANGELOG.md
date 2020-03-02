# Changelog

## In development
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

## 0.0.1
* Initial version
* Includes module detection