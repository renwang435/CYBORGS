import copy
import pathlib
from typing import Any, Dict, List, Optional

import torch
from torch import nn

import utils.distributed as dist


class CheckpointManager(object):
    r"""
    A helper class to periodically serialize models and other checkpointable
    objects (optimizers, LR schedulers etc., which implement ``state_dict``
    method) during training, and optionally record best performing checkpoint
    based on an observed metric.

    .. note::

        For :class:`~torch.nn.parallel.DistributedDataParallel` objects,
        ``state_dict`` of internal model is serialized.

    .. note::

        The observed metric for keeping best checkpoint is assumed "higher is
        better", flip the sign if otherwise.

    Parameters
    ----------
    serialization_dir: str
        Path to a directory to save checkpoints.
    filename_prefix: str
        Prefix of the checkpoit file names while saving. Default: "checkpoint"
        Checkpoint will be saved as ``"{prefix}_{epoch}.pth"``. 
    keep_recent: int, optional (default = 100)
        Number of recent ``k`` checkpoints to keep on disk. Older checkpoints
        will be removed. Set to a very large value for keeping all checkpoints.
    checkpointables: Any
        Keyword arguments with any checkpointable objects, for example: model,
        optimizer, learning rate scheduler.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager("/tmp", model=model, optimizer=optimizer)
    >>> num_epochs = 20
    >>> for epoch in range(num_epochs):
    ...     train(model)
    ...     val_loss = validate(model)
    ...     ckpt_manager.step(- val_loss, epoch)
    """

    def __init__(
        self,
        serialization_dir: str = "/tmp",
        # filename_prefix: str = "checkpoint",
        keep_recent: int = 200,
        boot_freq: int = 500,
        keep_every: List[int] = [],
        **checkpointables: Any,
    ):
        self.serialization_dir = pathlib.Path(serialization_dir)
        self.keep_recent = keep_recent
        self.boot_freq = boot_freq
        self.keep_every = keep_every

        # Shallow copy, keeps references to tensors as original objects.
        self.checkpointables = copy.copy(checkpointables)

        # Initialize members to hold state dict of best checkpoint and its
        # performance.
        self._best_metric: float = -1e-12
        self._best_ckpt: Dict[str, Any] = {}

        # Keep epoch numbers of recently saved 'k' checkpoints.
        self._recent_epochs: List[int] = []

    def step(self, epoch: int, metric: Optional[float] = None):
        r"""
        Serialize checkpoint and update best checkpoint based on metric. Keys
        in serialized checkpoint match those in :attr:`checkpointables`.

        Parameters
        ----------
        epoch: int
            Current training epoch. Will be saved with other checkpointables.
        metric: float, optional (default = None)
            Observed metric (higher is better) for keeping track of best
            checkpoint. If this is ``None``, best chckpoint will not be
            recorded/updated.
        """

        checkpointable_state_dict: Dict[str, Any] = self._state_dict()

        # We also checkpoint current epoch.
        checkpointable_state_dict["epoch"] = epoch

        # Update the best checkpoint based on metric, if provided.
        if metric is not None and metric > self._best_metric:
            self._best_metric = metric
            self._best_ckpt = copy.copy(checkpointable_state_dict)

        # Serialize checkpoint corresponding to current epoch.
        torch.save(
            checkpointable_state_dict,
            self.serialization_dir / f"checkpoint_{epoch}.pth",
        )
        if self._best_metric != -1e-12:
            # Serialize best performing checkpoint observed so far.
            torch.save(
                self._best_ckpt, self.serialization_dir / "checkpoint_best.pth"
            )
        
        if epoch in self.keep_every:
            torch.save(
                checkpointable_state_dict,
                self.serialization_dir / f"frozen_{epoch}.pth",
            )
        
        if epoch % self.boot_freq == 0:
            # MAY NEED TO CHANGE THIS FOR DETCON
            torch.save(
                checkpointable_state_dict,
                self.serialization_dir / f"boot_{epoch}.pth",
            )

        # Remove earliest checkpoint if there are more on disk.
        self._recent_epochs.append(epoch)
        if len(self._recent_epochs) > self.keep_recent:
            self.remove_earliest_checkpoint()
        

    def _state_dict(self):
        r"""Return a dict containing state dict of all checkpointables."""

        __state_dict: Dict[str, Any] = {}
        for key in self.checkpointables:
            if isinstance(
                self.checkpointables[key], nn.parallel.DistributedDataParallel
            ):
                __state_dict[key] = self.checkpointables[key].module.state_dict()
            else:
                __state_dict[key] = self.checkpointables[key].state_dict()

        return __state_dict

    def remove_earliest_checkpoint(self):
        r"""Remove earliest serialized checkpoint from disk."""

        earliest_epoch = self._recent_epochs.pop(0)
        (self.serialization_dir / f"checkpoint_{earliest_epoch}.pth").unlink()

    def load(self, checkpoint_path: str):
        r"""
        Load a serialized checkpoint from a path. This method will try to find
        each of :attr:`checkpointables` in the file and load its state dict.
        Since our checkpointables are held as references, this method does not
        return them.

        Parameters
        ----------
        checkpoint_path: str
            Path to a checkpoint serialized by :meth:`step`.

        Returns
        -------
        int
            epoch corresponding to the loaded checkpoint. Useful for
            resuming training. This will be -1 in case of best checkpoint,
            or if info does not exist.
        """

        # Each process will log a message after loading checkpoint.
        rank = dist.get_rank()

        print(f"Rank {rank}: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = checkpoint.pop("epoch", -1)

        # Keep flags of all checkpointables to lo which ones were not loaded.
        is_loaded = {key: False for key in self.checkpointables}

        # Load each checkpointable from checkpoint.
        for key in checkpoint:
            if key in self.checkpointables:
                print(f"Rank {rank}: Loading {key} from {checkpoint_path}")

                if isinstance(
                    self.checkpointables[key], nn.parallel.DistributedDataParallel
                ):
                    self.checkpointables[key].module.load_state_dict(checkpoint[key])
                else:
                    checkpoint[key] = {x.replace('module.',"") : y  
                        for x, y in checkpoint[key].items()}
                    self.checkpointables[key].load_state_dict(checkpoint[key])

                is_loaded[key] = True
            else:
                print(f"Rank {rank}: {key} not found in `checkpointables`.")

        not_loaded: List[str] = [key for key in is_loaded if not is_loaded[key]]
        if len(not_loaded) > 0:
            print(
                f"Rank {rank}: Checkpointables not found in file: {not_loaded}"
            )
        return epoch
