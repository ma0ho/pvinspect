from pvinspect.data import datasets
from pvinspect.preproc.detection import (
    locate_multiple_modules,
    locate_module_and_cells,
    segment_modules,
    segment_cells,
)
from pvinspect.data.image import *
from test.utilities import *


def test_multi_segmentation():
    # load images
    _, imgs = datasets.multi_module_detection(N=2)

    # perform multi module segmentation
    modules = locate_multiple_modules(imgs, rows=6, cols=10)

    # check result
    assert len(modules) == 12
    assert isinstance(modules[0].get_meta("multimodule_original"), Image)
    assert modules[0].get_meta("multimodule_original").path == imgs[0].path

    # check that plotting does not fail
    modules[0].get_meta("multimodule_original").show()

    # perform precise module localization and segmentation
    modules = locate_module_and_cells(modules)
    modules_crop = segment_modules(modules)

    # check result
    assert len(modules) == 12
    assert_equal(
        modules_crop[0].get_meta("transform")(np.array([[0.0, 0.0]])),
        np.array([0.0, 0.0]),
    )

    # make sure that original is preserved
    assert isinstance(modules_crop[0].get_meta("multimodule_original"), Image)
    assert modules_crop[0].get_meta("multimodule_original").path == imgs[0].path

    # check show
    modules_crop.head()

    # check segmentation into cells
    cells = segment_cells(modules_crop[0])
    assert len(cells) == 60

    # make sure that original is preserved
    assert isinstance(cells[0].get_meta("multimodule_original"), Image)
    assert cells[0].get_meta("multimodule_original").path == imgs[0].path


def test_single_segmentation():
    # load images
    img = datasets.poly10x6(N=1)[0]

    # perform detection
    module = locate_module_and_cells(img)

    # check result
    assert module.has_meta("transform")

    # check that show result does not fail
    module.show()

    # perform segmentation into cells
    cells = segment_cells(module)

    # check
    assert len(cells) == 60
