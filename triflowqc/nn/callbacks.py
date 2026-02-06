"""
Training callbacks: EarlyStopping, ModelCheckpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    monitor : str
        Metric key in history dict to monitor (default ``'val_loss'``).
    patience : int
        Number of epochs with no improvement after which to stop.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``'min'`` or ``'max'``.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._wait: int = 0

    def on_train_begin(self, model: Any) -> None:
        self._best = None
        self._wait = 0

    def on_epoch_end(self, epoch: int, history: Dict[str, List[float]], model: Any) -> Optional[bool]:
        if self.monitor not in history or not history[self.monitor]:
            return None
        current = history[self.monitor][-1]
        if self._best is None:
            self._best = current
            self._wait = 0
            return None

        if self.mode == "min":
            improved = current < self._best - self.min_delta
        else:
            improved = current > self._best + self.min_delta

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                return False  # Signal to stop
        return None


class ModelCheckpoint:
    """
    Save the model after every epoch (or only when improved).

    Parameters
    ----------
    filepath : str
        Path template for saving.
    monitor : str
        Metric to monitor.
    save_best_only : bool
        If True, only save when the monitored metric improves.
    mode : str
        ``'min'`` or ``'max'``.
    """

    def __init__(
        self,
        filepath: str = "model_checkpoint.tfqc",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self._best: Optional[float] = None

    def on_train_begin(self, model: Any) -> None:
        self._best = None

    def on_epoch_end(self, epoch: int, history: Dict[str, List[float]], model: Any) -> None:
        if self.monitor not in history or not history[self.monitor]:
            return
        current = history[self.monitor][-1]

        should_save = False
        if not self.save_best_only:
            should_save = True
        elif self._best is None:
            should_save = True
        elif self.mode == "min" and current < self._best:
            should_save = True
        elif self.mode == "max" and current > self._best:
            should_save = True

        if should_save:
            self._best = current
            try:
                model.save(self.filepath)
            except Exception:
                pass  # Graceful failure
