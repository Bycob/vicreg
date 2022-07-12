from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import numpy as np
from visdom import Visdom

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import imgaug.augmenters as iaa
import mmcv

import augmentations as aug
from distributed import init_distributed_mode

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
import segformer_config_b5
import segformer_config_b0
import builder
import EncoderDecoder
import mit
import segformer_head
import cross_entropy_loss

from builder import build_segmentor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
                xlabel='Time',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a segformer model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/dataset", required=True,
                        help='Path to the dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="b0",
                        help='Architecture of the segformer')
    parser.add_argument("--embedding", default="256",
                        help='Size of the embedding')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    
    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    #Cutout
    parser.add_argument("--nb-iterations", type=int, default=1,
                        help='number of cutouts made in the training image')
    parser.add_argument("--cutout-size", type=float, default=0.3,
                        help='Size of the cutouts made in the training image')

    return parser



def main(args):
    plotter = VisdomLinePlotter(env_name='env_test')
    
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    transform = aug.MaskTransform()

    #dataset = datasets.ImageFolder(args.data_dir / "train", transform2)
    dataset = datasets.ImageFolder(args.data_dir, transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    cfg = mmcv.Config.fromfile(os.path.join("vicreg", "".join(("segformer_config_", args.arch, ".py"))))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.decode_head.num_classes = 10
    
    net = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg")
        )

    model = VICSegformer(args, net=net).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            img = x
            x = torch.einsum('nchw->nhwc', x)
            x = x.numpy()
            cut = iaa.Cutout(nb_iterations=args.nb_iterations, size=args.cutout_size)
            x = cut(images=x)
            x = torch.tensor(x)
            x = torch.einsum('nhwc->nchw', x)

            x = x.cuda(gpu, non_blocking=True)
            img = img.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out, loss = model.forward(x, y, img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
                plotter.plot('loss', 'val', 'Class Loss', int(current_time - start_time), loss.item())
        
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.backbone.state_dict(), args.exp_dir / "segformer.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class VICSegformer(nn.Module):
    def __init__(self, args, net):
        super().__init__()
        self.args = args
        self.backbone = VICReg(args, net)

    def forward(self, x, y, img):
        out, loss = self.backbone(x, y)

        return out, loss
        


class VICReg(nn.Module):
    def __init__(self, args, net):
        super().__init__()
        self.args = args
        self.projector = Projector(args)
        self.backbone = net.backbone
        self.net = net
        self.num_features = 4096
        
    def forward(self, x, y):
        x = self.net.extract_feat(x)[-1]
        out = x
        x = self.projector(x)
        y = self.projector(self.net.extract_feat(y)[-1])
        
        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return out, loss


class Projector(nn.Module):
    def __init__(self, args):
        super().__init__()
        mlp = 4096
        mlp_spec = f"{args.embedding}-{mlp}-{mlp}-{mlp}"
        f = list(map(int, mlp_spec.split("-")))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(f[0], f[1])
        self.bn1 = nn.BatchNorm1d(f[1])
        self.relu = nn.ReLU(True)
        self.linear2 = nn.Linear(f[1], f[2])
        self.bn2 = nn.BatchNorm1d(f[2])
        self.linear3 = nn.Linear(f[2], f[3], bias=False)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
    
        return x


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]



if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
