from .image import ModuleImageSequence, ModuleImage, EL_IMAGE
from .io import read_module_image, read_module_images
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd

_ds_ids = {
    '20191219_poly10x6': '1B5fQPLvStuMvuYJ5CxbzyxfwuWQdfNVE'
}

def _check_and_download_ds(name: str):
    ds_path = Path(__file__).parent.absolute() / 'datasets' / name
    if not ds_path.is_dir():
        ds_path.mkdir(parents=True, exist_ok=False)
        gdd.download_file_from_google_drive(_ds_ids[name], str(ds_path / 'data.zip'), unzip=True)
    return ds_path

def poly10x6(N: int = 0) -> ModuleImageSequence:
    '''Read sequence of 10x6 poly modules
    
    Args:
        N (int): Only read first N images
    '''
    p = _check_and_download_ds('20191219_poly10x6')
    return read_module_images(p, EL_IMAGE, True, 10, 6, N = N)
