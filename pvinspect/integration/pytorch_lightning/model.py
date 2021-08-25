import logging
from abc import abstractmethod
from pathlib import Path
from sys import prefix
from typing import Any, Callable, List, Optional, Type

import numpy as np
import pandas as pd
from pvinspect.data.image.sequence import (
    ImageOrSequence,
    ImageSequence,
    TImageOrSequence,
    TImageSequence,
    sequence,
)
from pvinspect.integration.pytorch.dataset import Dataset
from tqdm.autonotebook import tqdm

try:
    import pytorch_lightning as pl
except ImportError as e:
    logging.error("Missing package pytorch_lightning. Please install.")
    raise e

try:
    import torch as t
except ImportError as e:
    logging.error("Missing package torch. Please install.")
    raise e


class InspectModel:
    def __init__(
        self,
        wrapped_module: Type[pl.LightningModule],
        checkpoint: Path,
        result_names: List[str],
        prefix: str = "pred_",
        data_transform: Optional[Callable[[np.ndarray], Any]] = None,
        use_cuda: bool = False,
        **kwargs
    ):
        self.module = wrapped_module.load_from_checkpoint(
            str(checkpoint), **kwargs
        ).eval()
        self.data_transform = data_transform
        self.prefix = prefix
        self.result_names = result_names
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.module = self.module.cuda()

    def apply(self, data: TImageSequence) -> TImageSequence:

        ds = Dataset(data, self.data_transform)
        results = list()

        with t.no_grad():
            for itm in tqdm(ds):
                if self.use_cuda:
                    itm = itm.cuda().unsqueeze(0)
                else:
                    itm = itm.unsqueeze(0)
                results.append(self.module.forward(itm))

        def apply_label(x: pd.Series):
            x = x.copy()
            res = results[x.name]
            for r, n in zip(list(res), self.result_names):
                x[self.prefix + n] = r.detach().cpu().squeeze(0).numpy()
            return x

        return data.apply_meta(apply_label)
