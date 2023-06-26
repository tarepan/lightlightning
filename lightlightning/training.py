"""PyTorch Lightning checkpointing, logging and training"""


import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import lightning as L                                          # pyright: ignore [reportMissingTypeStubs]
from lightning.pytorch.callbacks import ModelCheckpoint        # pyright: ignore [reportMissingTypeStubs]
from lightning.pytorch.loggers import TensorBoardLogger        # pyright: ignore [reportMissingTypeStubs]
from lightning.fabric.utilities.cloud_io import get_filesystem # pyright: ignore [reportMissingTypeStubs]
from omegaconf import MISSING


@dataclass
class ConfCkptLog:
    """Configuration of checkpointing and logging, directly unpackable with `**asdict(this)`.
    """
    dir_root: str = MISSING
    name_exp: str  = "default"
    name_version: str  = "version_-1"

class CheckpointAndLogging:
    """Generate path of checkpoint & logging.
    {dir_root}/
        {name_exp}/
            {name_version}/
                checkpoints/
                    {name_ckpt} # PyTorch-Lightning Checkpoint. Resume from here.
                hparams.yaml
                events.out.tfevents.{xxxxyyyyzzzz} # TensorBoard log file.
    """

    def __init__(self, conf: ConfCkptLog) -> None:
        # Checkpointing
        ## Storing: [Trainer's `default_root_dir`](https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer.params.default_root_dir)
        self.default_root_dir: str | None = conf.dir_root
        self.ckpt_cb = ModelCheckpoint(
            dirpath=None, # Path is inferred by Trainer with `ckpt_log`
            train_time_interval=timedelta(minutes=15), # every 15 minutes
            save_last=True,
            save_top_k=0,
        )
        path_ckpt = os.path.join(conf.dir_root, conf.name_exp, conf.name_version, "checkpoints", "last.ckpt")
        ## Resuming: [Trainer.fit's `ckpt_path`](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer.fit)
        exists = get_filesystem(path_ckpt).exists(path_ckpt) # type: ignore ; because of fsspec
        self.ckpt_path: str | None = path_ckpt if exists else None

        # Logging
        ## [PL's TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html)
        self.logger: TensorBoardLogger = TensorBoardLogger(conf.dir_root, conf.name_exp, conf.name_version)


@dataclass
class ConfTrain:
    """Configuration of train.
    Args:
        gradient_clipping  - Maximum gradient L2 norm, clipped when bigger than this (None==∞)
        max_epochs         - Number of maximum training epoch
        use_amp            - Whether to use Automatic-Mixed-Precision training (default: use)
        val_interval_epoch - Interval epoch between validation
        profiler           - Profiler setting
    """
    gradient_clipping:  float | None = MISSING
    max_epochs:         int          = MISSING
    val_interval_epoch: int          = MISSING
    use_amp:            bool         = True
    profiler:           str | None   = MISSING
    ckpt_log:           ConfCkptLog  = ConfCkptLog()


def train(model: L.LightningModule, conf: ConfTrain, datamodule: L.LightningDataModule) -> None:
    """Train the PyTorch-Lightning model.
    """

    # Fast non-deterministic training
    torch.backends.cudnn.benchmark = True # pyright: ignore [reportGeneralTypeIssues, reportUnknownMemberType]

    # Harmonized setups of checkpointing/logging
    ckpt_log = CheckpointAndLogging(conf.ckpt_log)

    # Trainer for mixed precision training on fast accelerator
    trainer = L.Trainer(
        precision="16-mixed" if conf.use_amp else "32", # Better choice for recent hardware: "bf16-mixed"
        gradient_clip_val=conf.gradient_clipping,
        max_epochs=conf.max_epochs,
        check_val_every_n_epoch=conf.val_interval_epoch,
        profiler=conf.profiler,
        # checkpoint/logging
        default_root_dir=ckpt_log.default_root_dir,
        logger=ckpt_log.logger,
        callbacks=[ckpt_log.ckpt_cb],
    )

    # training
    trainer.fit(model, ckpt_path=ckpt_log.ckpt_path, datamodule=datamodule) # pyright: ignore E[reportUnknownMemberType] ; because of PL
