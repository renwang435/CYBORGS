import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers per GPU to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for single gpu')

    # model
    parser.add_argument('--arch', type=str, default='resnet50', help="backbone architecture")
    parser.add_argument('--feature_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--num_classes', type=int, default=81, 
                        help="Number of object classes in masks (default: 80 from COCO")
    parser.add_argument('--num_masks', type=int, default=16,
                        help="Number of masks to sample for DetCon objective")
    parser.add_argument('--downsample_masks', type=int, default=32,
                        help="Factor to downsample masks by")
    parser.add_argument('--proj_hidden_dim', type=int, default=4096,
                        help="Hidden dim size for MLP in BYOL")
    parser.add_argument('--proj_dim', type=int, default=256,
                        help="Output dim size for MLP in BYOL")
    parser.add_argument('--ema_decay', type=float, default=0.996,
                        help="Exponential moving average for setting target network params")
    parser.add_argument('--softmax_temp', type=float, default=0.1,
                        help="Temperature for softmax in DetCon objective")

    # optimization
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='lars',
                        help='for optimizer choice.')
    parser.add_argument('--base_learning_rate', '--base_lr', type=float, default=0.2,
                        help='base learning when batch size = 128. final lr is determined by linear scale')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup_epoch', type=int, default=10, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list (NOT USED FOR COSINE ANNEALING)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate (NOT USED FOR COSINE ANNEALING)')
    parser.add_argument('--weight_decay', type=float, default=1.5e-6, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--start_epoch', type=int, default=0, help='used for resume')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # Bootstrapping arguments
    parser.add_argument('--pretrained', default=None, type=str,
                        help='If NOT None, then either i) torchvision or ii) path to pretrained ckpt')
    parser.add_argument('--pretrained_name', default=None, type=str,
                        help='If NOT None, then pretrained must be a path, and this is the type of pretrained model')
    parser.add_argument('--mask_dir', default='', type=str, metavar='PATH',
                    help='initial mask directory')
    parser.add_argument("--boot_freq", default=None, type=int,
                        help="Epochs before regenerating masks"
                        )
    parser.add_argument("--num_proc", default=48, type=int,
                        help="Multiprocessing to generate masks"
                        )
    parser.add_argument('--baseline', action="store_true",
                        help="Running a baseline of the hyperparameters."
                        )
    parser.add_argument('--layer_name', type=str, default="",
                        help="Layer to extract features from for unsupervised segmentation."
                        )
    parser.add_argument('--scramble_masks', action="store_true",
                        help="Randomize labels for each incoming mask."
                        )
    parser.add_argument('--bootstrap_classes', type=int, default=11, 
                        help="Number of object classes in bootstrapped masks")
    parser.add_argument('--imgs_per_batch', type=int, default=1,
                        help="Images to run KMeans together on")
    parser.add_argument("--min_clusters", type=int, default=11,
                    help="Minimum number of classes in segmentation")
    parser.add_argument("--max_clusters", type=int, default=11,
                        help="Maximum number of classes in segmentation")

    # VMF loss params
    parser.add_argument("--vmf_temp", default=0.1, type=float,
                    help="Temperature for VMF loss"
                    )
    parser.add_argument("--vmf_freq", default=5000, type=int,
                        help="Frequency to use VMF loss in"
                        )
    parser.add_argument("--vmf_loss_weight", type=float, default=0,
                        help="VMF clustering loss weight")
    

    # Distributed training arguments.
    parser.add_argument(
        "--num_machines", type=int, default=1,
        help="Number of machines used in distributed training."
    )
    parser.add_argument(
        "--num_gpus_per_machine", type=int, default=8,
        help="""Number of GPUs per machine with IDs as (0, 1, 2 ...). Set as
        zero for single-process CPU training.""",
    )
    parser.add_argument(
        "--machine_rank", type=int, default=0,
        help="""Rank of the machine, integer in [0, num_machines). Default 0
        for training with a single machine.""",
    )
    parser.add_argument("--dist_url", default="tcp://localhost:10001", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str,
                        help="distributed backend")

    # misc
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")

    args = parser.parse_args()

    return args
