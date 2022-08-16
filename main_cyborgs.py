import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import utils.distributed as dist
from data.datasets import renew_dataloader
from model.builder import DetConB
from model.losses import DetConBLoss, VMFClusterLoss
from model.lr_scheduler import get_scheduler
from model.optimizers import LARS, add_weight_decay
from options import get_args
from utils.bootstrapping import BootstrapManager
from utils.checkpointing import CheckpointManager
from utils.meters import AverageMeter, ProgressMeter


def main(args):
    # Multi-GPU setup
    # This method will only work for GPU training (single or multi).
    # Get the current device as set for current distributed process.
    # Check `launch` function in `moco.utils.distributed` module.
    device = torch.cuda.current_device()
    # pdb.set_trace()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # Remove default logger, create a logger for each process which writes to a
    # separate log-file. This makes changes in global scope.
    logger.remove(0)
    if dist.get_world_size() > 1:
        logger.add(
            os.path.join(args.output_dir, f"log-rank{dist.get_rank()}.txt"),
            format="{time} {level} {message}",
        )

    # Add a logger for stdout only for the master process.
    if dist.is_master_process():
        logger.add(
            sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
        )

    logger.info(
        f"Current process: Rank {dist.get_rank()}, World size {dist.get_world_size()}"
    )

    # Setup model (ALT)
    backbone = torchvision.models.__dict__[args.arch]()
    model = DetConB(
        backbone,
        224,
        hidden_layers=["layer2", "layer3", "layer4"],
        projection_size=args.proj_dim,
        projection_hidden_size=args.proj_hidden_dim,
        num_classes=args.num_classes,
        downsample=args.downsample_masks,
        num_samples=args.num_masks,
        moving_average_decay=args.ema_decay,
        use_momentum=True,
    ).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Setup loss
    criterion = DetConBLoss(args.softmax_temp).to(device)
    vmf_criterion = VMFClusterLoss(args.min_clusters, args.max_clusters + 1).to(device)

    # Setup optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    # lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
                                    lr=args.base_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        params = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.SGD(params,
                                    # lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
                                    lr=args.base_learning_rate,
                                    momentum=args.momentum)
        optimizer = LARS(optimizer)
    else:
        raise NotImplementedError



    # Create checkpoint manager and tensorboard writer.
    # keep_every = [50, 100, 200, 500, 800, 1000]
    # keep_every = [324, 648, 1296]
    keep_every = [1, 50, 100, 200, 324, 500, 648, 800, 1000, 1296]
    checkpoint_manager = CheckpointManager(
        serialization_dir=args.output_dir,
        keep_recent=5,
        boot_freq=args.boot_freq,
        keep_every=keep_every,
        state_dict=model,
        optimizer=optimizer,
    )   
    tensorboard_writer = SummaryWriter(log_dir=args.output_dir)

    min_clusters = args.min_clusters
    max_clusters = args.max_clusters + 1

    # Create bootstrap manager
    bootstrap_manager = BootstrapManager(
        serialization_dir=args.output_dir,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        imgs_per_batch=args.imgs_per_batch,
        data_dir=args.data_dir,
        layer_name=args.layer_name,
        root_mask_dir=args.mask_dir,
        num_proc=args.num_proc,
        state_dict=model,
        optimizer=optimizer,
    ) 

    if dist.is_master_process():
        tensorboard_writer.add_text("args", f"```\n{vars(args)}\n```")

    # optionally resume from a checkpoint
    if args.resume:
        args.start_epoch = CheckpointManager(state_dict=model).load(args.resume)

    cudnn.benchmark = True

    # Distributed training
    if dist.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],
                                                            find_unused_parameters=True)
    
    train_sampler, train_loader = renew_dataloader(args, 
                                                    os.path.join(args.mask_dir, 'train2017'),
                                                    )
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model.train()

    # start_time = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        if dist.get_world_size() > 1:
            train_sampler.set_epoch(epoch)
        
        logger.info("Current learning rate is {}".format(scheduler.get_last_lr()))

        cyborgs_train(train_loader, epoch + 1, model, criterion, vmf_criterion, optimizer, scheduler, 
                        args, device, tensorboard_writer)
        
        if not args.baseline and (epoch + 1) % args.boot_freq == 0:
            if not dist.is_master_process():
                dist.synchronize()

        if dist.is_master_process():
            checkpoint_manager.step(epoch=epoch + 1)
            if not args.baseline and (epoch + 1) % args.boot_freq == 0:
                new_mask_dir = os.path.join(args.mask_dir,
                                            str(args.boot_freq) + '_' + str(args.epochs) + '_' + str(args.base_learning_rate),
                                            str(epoch + 1))
                os.makedirs(new_mask_dir, exist_ok=True)
                bootstrap_manager.step(epoch=epoch + 1, new_mask_dir=new_mask_dir)
                dist.synchronize()
        
        if not args.baseline and (epoch + 1) % args.boot_freq == 0:
            new_mask_dir = os.path.join(args.mask_dir,
                                            str(args.boot_freq) + '_' + str(args.epochs) + '_' + str(args.base_learning_rate),
                                            str(epoch + 1))
            train_sampler, train_loader = renew_dataloader(args, new_mask_dir,)



def cyborgs_train(train_loader, epoch, model, criterion, vmf_criterion, optimizer, scheduler, args, device, tensorboard_writer):
    batch_time_meter = AverageMeter("Time", ":6.3f")
    data_time_meter = AverageMeter("Data", ":6.3f")
    losses_total = AverageMeter("Loss_total", ":.4e")
    losses_mask = AverageMeter("Loss_mask", ":.4e")
    losses_vmf = AverageMeter("Loss_vmf", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        # [batch_time_meter, data_time_meter, losses_total, losses_ssl, losses_gcam_masked, top1, top5],
        [batch_time_meter, data_time_meter, losses_total, losses_mask, losses_vmf],
        prefix=f"Epoch: [{epoch}]",
    )

    start_time = time.perf_counter()

    for i, (view1, view2) in enumerate(train_loader):
        data_time = time.perf_counter() - start_time
        view1[0] = view1[0].to(device, non_blocking=True)
        view1[1] = view1[1].to(device, non_blocking=True)

        view2[0] = view2[0].to(device, non_blocking=True)
        view2[1] = view2[1].to(device, non_blocking=True)

        # Compute DetConB loss
        (online_pred1, online_pred2, 
        target_proj1, target_proj2, 
        online_ids1, online_ids2, 
        target_ids1, target_ids2,
        online_ft_map1, online_ft_map2,
        target_ft_map1, target_ft_map2) = model(view1[0], view1[1], view2[0], view2[1])

        loss_mask = criterion(pred1=online_pred1,
                            pred2=online_pred2,
                            target1=target_proj1.detach(),
                            target2=target_proj2.detach(),
                            pind1=online_ids1,
                            pind2=online_ids2,
                            tind1=target_ids1,
                            tind2=target_ids2,
                        )
        
        if args.vmf_loss_weight > 0 and epoch % args.vmf_freq == 0:
            loss_vmf = vmf_criterion(omap1=online_ft_map1,
                                        omap2=online_ft_map2,
                                        tmap1=target_ft_map1,
                                        tmap2=target_ft_map2,
                                        vmf_temp=torch.tensor(args.vmf_temp),
                                        vmf_weight=args.vmf_loss_weight)
        else:
            loss_vmf = torch.tensor(0)
        
        loss = loss_mask + loss_vmf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.update_moving_average()
        else:
            model.update_moving_average()

        # measure elapsed time
        batch_time = time.perf_counter() - start_time

        # update all progress meters
        data_time_meter.update(data_time)
        batch_time_meter.update(batch_time)
        losses_total.update(loss.item(), view1[0].size(0))
        losses_mask.update(loss_mask.item(), view1[0].size(0))
        losses_vmf.update(loss_vmf.item(), view1[0].size(0))

        # log to tensorboard
        if dist.is_master_process():
            tensorboard_writer.add_scalar("data_time", data_time, epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("batch_time", batch_time, epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("loss_total", loss.detach(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("loss_mask", loss_mask.detach(), epoch * len(train_loader) + i)
            tensorboard_writer.add_scalar("loss_vmf", loss_vmf.detach(), epoch * len(train_loader) + i)

            
        if i % args.print_freq == 0:
            progress.display(i)

        start_time = time.perf_counter()



if __name__ == '__main__':
    _A = get_args()
    os.makedirs(_A.output_dir, exist_ok=True)    

    _A.output_dir = os.path.abspath(_A.output_dir)

    if _A.num_gpus_per_machine == 0:
        raise NotImplementedError("Training on CPU is not supported.")        
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus_per_machine,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            dist_backend=_A.dist_backend,
            args=(_A, ),
        )
