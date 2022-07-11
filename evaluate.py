# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import numpy as np
from visdom import Visdom

from torch import nn, optim
from torchvision import datasets, transforms
import torch

import resnet
from encoder_decoder import ResnetEncoder

from builder import build_segmentor
import mmcv
import mit
import segformer_head
import cross_entropy_loss


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8098)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')



def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model")

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument("--train-percent", default='100', type=str, choices=('100', '10', '1', 'zero1'), help="size of traing set in percent", )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, help="path to pretrained model")
    parser.add_argument("--exp-dir", default="./checkpoint/lincls/", type=Path, metavar="DIR", help="path to checkpoint directory", )
    parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency")

    # Model
    parser.add_argument("--arch", type=str, default="resnet50", choices=("resnet50", "segformer", "encoder"), help='architecture of the network to evaluate')

    # Optim
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run", )
    parser.add_argument("--batch-size", default=256, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--lr-backbone", default=0.0, type=float, metavar="LR", help="backbone base learning rate", )
    parser.add_argument("--lr-head", default=0.3, type=float, metavar="LR", help="classifier base learning rate", )
    parser.add_argument("--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay")
    parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"), help="finetune or freeze resnet weights", )

    # Running
    parser.add_argument("--workers", default=8, type=int, metavar="N", help="number of data loader workers", )

    return parser



def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    plotter = VisdomLinePlotter(env_name='env_test')
    
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    """Choice of the backbone : segformer, encoder or classic vicreg resnet"""
    if args.arch == "segformer":
        cfg = mmcv.Config.fromfile(os.path.join("vicreg", "segformer_config_b0.py"))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.decode_head.num_classes = 10
        net = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg")
        )
        backbone = net.backbone

    elif args.arch == "encoder":
        backbone = ResnetEncoder(input_nc=3, output_nc=3)

    elif args.arch == "resnet50":
        backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)

    
    state_dict = torch.load(args.pretrained, map_location="cpu")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    assert missing_keys == [] and unexpected_keys == []
    #if "model" in state_dict:
    #    state_dict = state_dict["model"]
    #    state_dict = {
    #        key.replace("module.backbone.", ""): value
    #        for (key, value) in state_dict.items()
    #    }
    #backbone.load_state_dict(state_dict, strict=False)

    """Change the size of the input according to the backbone chosen"""
    head = nn.Linear(256, 1000)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    """only use with the segformer backbone"""
    class Model(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        def forward(self, x):
            x = backbone(x)
            x = x[-1]
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            return x
        
    bbone = Model(backbone)
    model = nn.Sequential(bbone, head)
    model.cuda(gpu)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    if args.train_percent == '100':
        traindir = args.data_dir / "train"

    elif args.train_percent in {'zero1', '1', '10'}:
        traindir = args.data_dir / "".join(("train", args.train_percent))
    valdir = args.data_dir / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            output = model(images.cuda(gpu, non_blocking=True))
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        plotter.plot('loss', 'val', 'Class Loss', epoch, loss.item())

        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(
                        output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                    )
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(
                epoch=epoch,
                acc1=top1.avg,
                acc5=top5.avg,
                best_acc1=best_acc.top1,
                best_acc5=best_acc.top5,
            )
            acc = top1.avg
            plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            torch.save(state, args.exp_dir / "checkpoint.pth")



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
