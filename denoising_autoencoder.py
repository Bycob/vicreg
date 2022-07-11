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
import torchvision.datasets as datasets

import resnet
import imgaug.augmenters as iaa
import imgaug
import cv2

from ResDecoder import ResnetGenerator
import augmentations as aug
from distributed import init_distributed_mode

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    def image(self, grid):
        self.viz.image(grid)



def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a denoising autoencoder", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    
    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=64,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument("--world-size", default=1, type=int,
                        help='number of distributed processes')

    #Cutout
    parser.add_argument("--nb-iterations", type=int, default=2,
                        help='number of cutouts made in the training image')
    parser.add_argument("--cutout-size", type=float, default=0.2,
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

    transforms = aug.MaskTransform() #transform without augmentations

    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
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

    model = ResnetGenerator(input_nc=3, output_nc=3).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
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
        for step, ((x, _), _) in enumerate(loader, start=epoch * len(loader)):
            img = x

            """cutout on the image, img is the original image and x is the image with cutouts"""
            x = torch.einsum('nchw->nhwc', x)
            x = x.numpy()
            cut = iaa.Cutout(nb_iterations=args.nb_iterations, size=args.cutout_size)
            x = cut(images=x)
            x = torch.tensor(x)
            x = torch.einsum('nhwc->nchw', x)

            x = x.cuda(gpu, non_blocking=True)
            img = img.cuda(gpu, non_blocking=True)
            
            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model.forward(x)
                loss = F.mse_loss(out, img)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            """print examples of the image, the image with cutouts and the reconstructed image"""
            if (step % 200) == 0:
                img2 = torch.einsum('nchw->nhwc', img.cpu())
                x2 = torch.einsum('nchw->nhwc', x.cpu())
                out2 = torch.einsum('nchw->nhwc', out.cpu())
                
                img2 = img2.detach().numpy()
                x2 = x2.detach().numpy()
                out2 = out2.detach().numpy()
                
                img2 = img2.astype(np.uint8)
                x2 = x2.astype(np.uint8)
                out2 = out2.astype(np.uint8)
                
                cells = [img2[4], x2[4], out2[4]]
                grid_image = imgaug.draw_grid(cells, cols=3)
                cv2.imwrite(str(epoch) + "_test.png", grid_image)
                plotter.image(grid_image)

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
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")


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



class Denoising_Autoencoder_Res(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.decoder = Decoder(args)

    def forward(self, x, img):
        out = self.decoder(self.backbone(x))
        loss = F.mse_loss(out, img)

        return loss, out


def Decoder(args):
    layers = []
    layers.append(nn.ConvTranspose2d(2048, 2048, kernel_size=7, stride=2, padding=0))
    layers.append(nn.SELU(True))
    layers.append(nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1))
    layers.append(nn.SELU(True))
    layers.append(nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=1))
    layers.append(nn.SELU(True))
    layers.append(nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1))
    layers.append(nn.SELU(True))
    layers.append(nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=1))
    layers.append(nn.SELU(True))
    layers.append(nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=1, output_padding=1))
    layers.append(nn.SELU(True))
    return nn.Sequential(*layers)

def exclude_bias_and_norm(p):
    return p.ndim == 1

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Autoencoder training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
    
        
