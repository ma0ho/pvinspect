from pvinspect import data, preproc
from pvinspect.data.image import CellImageSequence, ModuleImageSequence
from pathlib import Path
from pvinspect.common.transform import HomographyTransform, FullTransform
import numpy as np
from .utilities import assert_equal

def test_locate_homography():
    seq = data.demo.poly10x6(2)
    seq = preproc.locate_module_and_cells(seq, False)

    assert isinstance(seq[0].transform, HomographyTransform)
    assert isinstance(seq[1].transform, HomographyTransform)
    assert seq[0].transform.valid
    assert seq[1].transform.valid

    # check correct origin
    x = seq[0].transform(np.array([[0.0,0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160
    x = seq[1].transform(np.array([[0.0,0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160

def test_locate_full():
    seq = data.demo.poly10x6(2)
    seq = preproc.locate_module_and_cells(seq, True)

    assert isinstance(seq[0].transform, FullTransform)
    assert isinstance(seq[1].transform, FullTransform)
    assert seq[0].transform.valid
    assert seq[1].transform.valid

    # check correct origin
    x = seq[0].transform(np.array([[0.0,0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160
    x = seq[1].transform(np.array([[0.0,0.0]])).flatten()
    assert x[0] > 1760 and x[0] < 1840
    assert x[1] > 80 and x[1] < 160

def test_segment_cells():
    seq = data.demo.poly10x6(2)
    seq = preproc.locate_module_and_cells(seq, True)
    cells = preproc.segment_cells(seq)

    assert isinstance(cells, CellImageSequence)
    assert len(cells) == 120
    assert cells[0].path == seq[0].path
    assert cells[0].row == 0
    assert cells[0].col == 0
    assert cells[1].col == 1
    assert cells[11].row == 1

def test_segment_modules():
    seq = data.demo.poly10x6(2)
    seq = preproc.locate_module_and_cells(seq, True)
    modules = preproc.segment_modules(seq)
    
    assert isinstance(modules, ModuleImageSequence)
    assert len(modules) == 2
    assert modules[0].path == seq[0].path
    assert modules.same_camera is False
    
    x = modules[0].transform(np.array([[0.0,0.0]])).flatten()
    assert_equal(x[0], 0.0)
    assert_equal(x[1], 0.0)
    x = modules[0].transform(np.array([[10.0,0.0]])).flatten()
    assert_equal(x[0], modules[0].shape[1])
    assert_equal(x[1], 0.0)
    x = modules[0].transform(np.array([[10.0,6.0]])).flatten()
    assert_equal(x[0], modules[0].shape[1])
    assert_equal(x[1], modules[0].shape[0])



