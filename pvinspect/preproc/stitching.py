import numpy as np
from pvinspect.data import ModuleImage, ModuleImageSequence
from pvinspect.data.image import Image, ImageSequence, Modality
from pvinspect.preproc.detection import locate_module_and_cells, segment_module_part


def _stitch_two(
    img1: Image, img2: Image, cell_size: int, overlap: int, direction: str
) -> Image:
    if direction == "hor":
        img1 = segment_module_part(
            img1,
            0,
            0,
            img1.get_meta("cols") - (overlap // 2),
            img1.get_meta("rows"),
            cell_size,
        )
        img2 = segment_module_part(
            img2,
            overlap // 2,
            0,
            img2.get_meta("cols") - (overlap // 2),
            img2.get_meta("rows"),
            cell_size,
        )
        return img1.from_self(
            data=np.concatenate([img1.data, img2.data], axis=1),
            meta=dict(cols=img1.get_meta("cols") + img2.get_meta("cols") - overlap),
        )
    else:
        img1 = segment_module_part(
            img1,
            0,
            0,
            img1.get_meta("cols"),
            img1.get_meta("rows") - (overlap // 2),
            cell_size,
        )
        img2 = segment_module_part(
            img2,
            0,
            overlap // 2,
            img2.get_meta("cols"),
            img2.get_meta("rows") - (overlap // 2),
            cell_size,
        )
        return img1.from_self(
            data=np.concatenate([img1.data, img2.data], axis=0),
            meta=dict(rows=img1.get_meta("rows") + img2.get_meta("rows") - overlap),
        )


def _scale_median_brightness(x: np.ndarray, target: float) -> np.ndarray:
    source = np.median(x)
    return (x * (target / source)).astype(x.dtype)


def stitch_modules(
    images: ImageSequence,
    n_horizontal: int = 1,
    n_vertical: int = 1,
    overlap_horizontal: int = 0,
    overlap_vertical: int = 0,
    equalize_intensity: bool = False,
) -> Image:
    """Locate and stitch partial recordings of a module

    Args:
        images (ImageSequence): Partial recordings to be stitched together (needs to be order left -> right / top -> bottom)
        n_horizontal (int): Number of partial recordings in horizontal direction
        n_vertical (int): Number of partial recordings in vertical direction
        overlap_horizontal (int): Number of fully visible cells that overlap between any two partial recordings
            in horizontal direction
        overlap_vertical (int): Number of fully visible cells that overlap between any two partial recordings
            in vertical direction
        equalize_intenity (bool): Match the median intensity of every partial recording to the median intensity of the first

    Returns:
        image (Image): The stitched image
    """

    if n_horizontal < 1 or n_vertical < 1:
        raise RuntimeError(
            "Number of images for stitching must not be smaller than 1 in any direction"
        )

    if n_horizontal > 2 or n_vertical > 2:
        raise NotImplementedError(
            "Stitching is only implemented for configurations 1x2, 2x1 and 2x2"
        )

    if overlap_horizontal < 0 or overlap_vertical < 0:
        raise RuntimeError("Invalid overlap")

    if overlap_horizontal % 2 != 0 or overlap_vertical % 2 != 0:
        raise NotImplementedError("Stitching is only implemented for even overlap")

    # determine spacing based on first image
    t = images[0].get_meta("transform")
    cell_size = np.linalg.norm(
        t(np.array([[1.0, 1.0]]))[0] - t(np.array([[0.0, 0.0]]))[0]
    )

    # equalize brightness if set
    if equalize_intensity:
        ref = np.median(images[0].data)  # use the first image as reference
        images = images.apply_image_data(_scale_median_brightness, target=ref)

    # stitch
    if n_horizontal == 1 and n_vertical == 1:
        return images[0]
    elif n_horizontal == 2 and n_vertical == 1:
        return _stitch_two(images[0], images[1], cell_size, overlap_horizontal, "hor")
    elif n_horizontal == 1 and n_vertical == 2:
        return _stitch_two(images[0], images[1], cell_size, overlap_vertical, "ver")
    else:  # n_horizontal == 2 and n_vertical == 2
        row1 = _stitch_two(images[0], images[1], cell_size, overlap_horizontal, "hor")
        row2 = _stitch_two(images[2], images[3], cell_size, overlap_horizontal, "hor")
        return row1.from_self(
            data=np.concatenate([row1.data, row2.data], axis=0),
            meta=dict(rows=row1.get_meta("rows") + row2.get_meta("rows")),
        )


# def locate_and_stitch_modules(images: ImageSequence, n_horizontal: int = 1, )
def locate_and_stitch_modules(
    images: ImageSequence,
    rows: int,
    cols: int,
    n_horizontal: int = 1,
    n_vertical: int = 1,
    overlap_horizontal: int = 0,
    overlap_vertical: int = 0,
    equalize_intensity: bool = False,
) -> Image:
    """Locate and stitch partial recordings of a module

    This method applies localization of modules followed by stitching. It relies on an exact specification
    of the visible module geometry (by means of rows, cols, n_horizontal, n_vertical, overlap_horizontal,
    overlap_vertical). Furthermore images need to be given in the specified order.

    This method optionally provides adaptation of intensities of the partial recordings. This is turned off
    by default, since it changes the original intensities, which might be undesirable in some cases.

    Args:
        images (ImageSequence): Partial recordings to be stitched together (needs to be order left -> right / top -> bottom)
        rows (int): Number of fully visible rows of cells in every partial recording
        cols (int): Number of fully visible columns of cells in every partial recording
        n_horizontal (int): Number of partial recordings in horizontal direction
        n_vertical (int): Number of partial recordings in vertical direction
        overlap_horizontal (int): Number of fully visible cells that overlap between any two partial recordings
            in horizontal direction
        overlap_vertical (int): Number of fully visible cells that overlap between any two partial recordings
            in vertical direction
        equalize_intensity (bool): Match the median intensity of every partial recording to the median intensity of the first

    Returns:
        image (Image): The stitched image
    """

    modimages = ModuleImageSequence(
        [
            ModuleImage(x.data, modality=Modality.EL_IMAGE, cols=cols, rows=rows)
            for x in images
        ]
    )

    # locate
    modimages = locate_module_and_cells(modimages, orientation="horizontal")

    # stitch
    return stitch_modules(
        modimages,
        n_horizontal,
        n_vertical,
        overlap_horizontal,
        overlap_vertical,
        equalize_intensity,
    )
