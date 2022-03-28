from pathlib import Path
import numpy as np
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import datasets, transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import resnet


def get_arguments():
    parser = argparse.ArgumentParser(description="Finetune a pretrained model on cracks")

    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument("--exp-dir", default="./checkpoint/lincls/", type=Path, metavar="DIR", help="path to checkpoint directory")
    parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--lr-backbone", default=0.0, type=float, metavar="LR", help="backbone base learning rate")
    parser.add_argument("--lr-head", default=0.3, type=float, metavar="LR", help="classifier base learning rate")
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay")
    parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"), help="finetune or freeze resnet weights")
    parser.add_argument("--workers", default=8, type=int, metavar="N", help="number of data loader workers")

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        
