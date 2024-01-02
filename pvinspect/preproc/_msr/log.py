from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch as t
from torch.autograd.grad_mode import no_grad

from .util import *


def no_debug(func):
    def wrapper(cls, *args, **kwargs):
        if cls.debug:
            func(cls, *args, **kwargs)

    return wrapper


# a logger that does nothing
class DummyLogger:
    def dummymethod(self, *args, **kwargs):
        return

    def __init__(self):
        super(DummyLogger).__init__()

    def __getattr__(self, name):
        return self.dummymethod


class Logger:

    logs: Dict[int, Dict[str, List[Any]]]
    hparams: Optional[pd.Series]
    debug: bool

    def _log(self, key: str, data: Any, step: int) -> None:
        if step not in self.logs.keys():
            self.logs[step] = dict()

        if key not in self.logs[step].keys():
            self.logs[step][key] = list()

        self.logs[step][key].append(data)

    def _prep_tensor(self, data: t.Tensor) -> t.Tensor:
        return data.detach().cpu()

    def _write_images(self, target: Path) -> None:
        for step, vals in self.logs.items():
            if "images" in vals.keys():
                for (name, img) in vals["images"]:
                    name = "step{:02d}_image_{}.png".format(step, name)
                    save_pytorch_image(target / name, img)

    def _write_plots(self, target: Path) -> None:
        for step, vals in self.logs.items():
            if "plots" in vals.keys():
                for (name, img, lognorm) in vals["plots"]:
                    name = "step{:02d}_plot_{}.png".format(step, name)
                    plot_pytorch_image(target / name, img, lognorm)

    def _collect_metrics(self):
        all_y = dict()
        all_x = dict()

        # collect metrics
        for step, vals in self.logs.items():
            y_step = dict()

            if "metrics" in vals.keys():
                for (name, v) in vals["metrics"]:
                    if name not in y_step.keys():
                        y_step[name] = list()
                    y_step[name].append(v)

            for name in y_step.keys():
                if name not in all_x.keys():
                    all_x[name] = list()
                    all_y[name] = list()

                all_x[name] += np.linspace(
                    step, step + 1 - 1e-9, num=len(y_step[name]), endpoint=True
                ).tolist()
                all_y[name] += y_step[name]

        return all_x, all_y

    def _write_metrics(self, target: Path, joint_plots: List[str]) -> None:
        all_x, all_y = self._collect_metrics()

        # plot single plots
        for name in all_x.keys():
            fig, ax = plt.subplots(1)
            ax.plot(all_x[name], all_y[name], label=name)
            ax.legend()
            fig.savefig(target / "{}.pdf".format(name))

        # plot joint plots
        if len(joint_plots) > 0:
            fig, ax = plt.subplots(1)
            for name in joint_plots:
                y = np.array(all_y[name])
                y -= y.min()
                y /= y.max()
                ax.plot(all_x[name], y, label=name)
        ax.set_yticks([])
        ax.legend()
        fig.savefig(target / "joint.pdf")

    def __init__(self, debug: bool) -> None:
        super().__init__()
        self.debug = debug
        self.logs = dict()
        self.hparams = None

    @t.no_grad()
    @no_debug
    def log_image(self, image: t.Tensor, name: str, step: int) -> None:
        image = self._prep_tensor(image)
        self._log("images", (name, image), step)

    @t.no_grad()
    @no_debug
    def log_plot(
        self, data: t.Tensor, name: str, step: int, lognorm: bool = False
    ) -> None:
        data = self._prep_tensor(data)
        self._log("plots", (name, data, lognorm), step)

    @no_debug
    def log_metric(
        self, name: str, values: Union[List[float], float], step: int
    ) -> None:
        if isinstance(values, list):
            for v in values:
                self._log("metrics", (name, v), step)
        else:
            self._log("metrics", (name, [values, values]), step)

    @no_debug
    def log_metrics(self, values: pd.DataFrame, step: int) -> None:
        for c in values.columns:
            self.log_metric(c, values[c].tolist(), step)

    @no_debug
    def log_hparams(self, hparams: pd.Series) -> None:
        self.hparams = hparams

    @property
    def metrics(self) -> List[str]:
        names = list()

        for step, vals in self.logs.items():
            y_step = dict()

            if "metrics" in vals.keys():
                for (name, v) in vals["metrics"]:
                    if name not in names:
                        names.append(name)
        return names

    @property
    def images(self) -> List[str]:
        names = list()

        for step, vals in self.logs.items():
            y_step = dict()

            if "images" in vals.keys():
                for (name, v) in vals["images"]:
                    if name not in names:
                        names.append(name)
        return names

    @property
    def plots(self) -> List[str]:
        names = list()

        for step, vals in self.logs.items():
            y_step = dict()

            if "plots" in vals.keys():
                for (name, v, _) in vals["plots"]:
                    if name not in names:
                        names.append(name)
        return names

    @no_debug
    def save(
        self, target: Path, joint_plots: List[str], clear_dir: bool = True
    ) -> None:
        if clear_dir:
            for f in target.glob("*"):
                if f.is_file():
                    f.unlink()
                else:
                    pass  # do not delete directories
            target.mkdir(parents=True, exist_ok=True)

        self._write_images(target)
        self._write_plots(target)
        self._write_metrics(target, joint_plots)

        # save hparams
        if self.hparams is not None:
            self.hparams.to_csv(target / "hparams.csv")

    def show_metrics(self, names: Union[str, List[str]]) -> None:
        show_multiple = isinstance(names, list) and len(names) > 1

        if isinstance(names, str):
            names = [names]

        all_x, all_y = self._collect_metrics()

        # plot joint plots
        fig, ax = plt.subplots(1)

        if show_multiple:
            for name in names:
                y = np.array(all_y[name])
                y -= y.min()
                y /= y.max()
                ax.plot(all_x[name], y, label=name)
            ax.set_yticks([])
            ax.legend()
        else:
            for name in names:
                y = np.array(all_y[name])
                ax.plot(all_x[name], y, label=name)
            ax.legend()

    def show_image(self, name: str) -> None:
        all_steps = list()
        all_images = list()

        show = None

        for step, vals in self.logs.items():
            if "images" in vals.keys():
                for (n, img) in vals["images"]:
                    if n == name:
                        all_steps.append(step)
                        all_images.append(img)
                        show = "image"
            if "plots" in vals.keys():
                for (n, img, lognorm) in vals["plots"]:
                    if n == name:
                        all_steps.append(step)
                        all_images.append(img)
                        show = "plot" if not lognorm else "logplot"

        N = len(all_steps)
        cols = min(5, N)
        rows = N // cols
        rows += 1 if N % cols > 0 else 0
        aspect = cols / rows
        fig, axs = plt.subplots(rows, cols, figsize=(int(10 * aspect), 10))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < N:
                    if N > cols:
                        if show == "image":
                            show_pytorch_image(all_images[idx], ax=axs[i, j])
                        else:
                            show_pytorch_plot(
                                all_images[idx],
                                lognorm=show == "logplot",
                                ax=axs[i, j],
                                fig=fig,
                            )
                        axs[i, j].set_title("step = {:d}".format(all_steps[idx]))
                    elif N > 1:
                        if show == "image":
                            show_pytorch_image(all_images[idx], ax=axs[j])
                        else:
                            show_pytorch_plot(
                                all_images[idx],
                                lognorm=show == "logplot",
                                ax=axs[j],
                                fig=fig,
                            )
                        axs[j].set_title("step = {:d}".format(all_steps[idx]))
                    else:
                        if show == "image":
                            show_pytorch_image(all_images[idx], ax=axs)
                        else:
                            show_pytorch_plot(
                                all_images[idx],
                                lognorm=show == "logplot",
                                ax=axs,
                                fig=fig,
                            )
                        axs.set_title("step = {:d}".format(all_steps[idx]))
                else:
                    if N > cols:
                        axs[i, j].axis("off")
                    elif N > 1:
                        axs[j].axis("off")
