import copy
import os
import pathlib
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from torch import nn


class BootstrapManager(object):
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
        min_clusters: int = 11,
        max_clusters: int = 1,
        imgs_per_batch: int = 1,
        data_dir: str = "",
        layer_name: str = "",
        root_mask_dir: str = "/tmp",
        num_proc: int = 48,
        **checkpointables: Any,
    ):
        self.serialization_dir = pathlib.Path(serialization_dir)
        self.root_mask_dir = pathlib.Path(root_mask_dir)

        # Load the image IDs
        self.data_dir = os.path.join(data_dir, 'train2017')

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.imgs_per_batch = imgs_per_batch
        self.layer_name = layer_name
        
        self.num_proc = num_proc

        # Shallow copy, keeps references to tensors as original objects.
        self.checkpointables = copy.copy(checkpointables)

    def step(self, epoch: int, new_mask_dir: str):
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

        checkpoint_path = self.serialization_dir / f"boot_{epoch}.pth"

        # extract_kmeans(self.num_proc, new_mask_dir, self.all_imgs,
        #                 checkpoint_path)

        subprocess.run([sys.executable, "subproc_extract_kmeans.py",
                        "--data_dir", self.data_dir,
                        "--min_clusters", str(self.min_clusters),
                        "--max_clusters", str(self.max_clusters),
                        "--imgs_per_batch", str(self.imgs_per_batch),
                        "--layer_name", self.layer_name,
                        "--mask_dir", new_mask_dir,
                        "--num_proc", str(self.num_proc),
                        "--ckpt", checkpoint_path,
                        ], check=True)
