from pathlib import Path
from typing import Union, Dict, List, Tuple
from shapely.geometry import Polygon

PathOrStr = Union[Path, str]
ObjectAnnotations = Dict[str, List[Tuple[str, Polygon]]]
