# Changelog

## In development
* Rework + cleanup module detection
* Allow detection of partial modules
  * includes new module type `PartialModuleImage`
  * includes new reading methods `read_partial_module_image`and `read_partial_module_images`
* Automatic clipping of outliers in viewer
  * includes `clip_low` and `clip_high` arguments in viewing methods

## 0.0.1
* Initial version
* Includes module detection