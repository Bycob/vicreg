from pathlib import Path
import numpy as np
import random
import signal
import argparse
import json
import math
import os
import sys
import time
import urllib


import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import augmentations as aug
from distributed import init_distributed_mode

import resnet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain an asymmetric architecture with VICReg", add_help=False)

    parser.add_argument("--exp-dir", type=Path, default="./exp", help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60, help='Print logs, to the stats.txt file every [log-freq-time] seconds')
    parser.add_argument("--arch-1", type=str, default="resnet50", help='Architecture of the first backbone encoder network')
    parser.add_argument("--arch-2", type=str, default="resnet50", help='Architecture of the second backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192", help='Size and number of layers of the MLP expander head')
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048, help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2, help='Base learning rate, effective learning rate after warmup is [base-lr]*[batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6, help='Weight decay')
    parser.add_argument("--sim-coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world-size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--print-freq", default=100, type=int, metavar="N", help="print frequency")
    parser.add_argument("--epochs-test", default=100, type=int, metavar="N", help="number of total epochs to run")
    
    return parser





def main(args):
    
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    
    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        
        stats_test_file = open(args.exp_dir / "stats_test.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_test_file)

        

    transforms = aug.TrainTransform()
    datadir = "/data1/jeanne/datasets/dogs_cats"
    dataset = datasets.ImageFolder(datadir, transforms)
    
    num_data = len(dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_data))
    train_idx, test_idx = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    assert args.batch_size % args.world_size == 0
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, **kwargs)

    
    
    model = Asym_VICReg(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.wd, weight_decay_filter=exclude_bias_and_norm, lars_adaptation_filter=exclude_bias_and_norm)


    
    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
            
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        
    else:
        start_epoch = 0


        

    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    
    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(start_epoch, args.epochs):
        
        train_sampler.set_epoch(epoch)
        
        for step, ((x, y), _) in enumerate(train_loader, start=epoch * len(train_loader)):
            
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, train_loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                
                stats = dict(epoch=epoch, step=step, loss=loss.item(), time=int(current_time - start_time), lr=lr)
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time


                
        if args.rank == 0:
            state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
            torch.save(state, args.exp_dir / "model.pth")


            
    if args.rank == 0:
        torch.save(model.module.backbone_1.state_dict(), args.exp_dir / "resnet50.pth")
        torch.save(model.module.backbone_2.state_dict(), args.exp_dir / "resnet50x2.pth")





        
    backbone, embedding = resnet.__dict__["resnet50"](zero_init_residual=True)
    state_dict_test = torch.load(args.exp_dir / "resnet50.pth")
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict_test, strict=False)
    assert missing_keys == [] and unexpected_keys ==[]

    head = nn.Linear(embedding, 1000)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    
    model_test = nn.Sequential(backbone, head)
    model_test.cuda(gpu)
    
    backbone.requires_grad_(False)
    head.requires_grad_(True)
    model_test = torch.nn.parallel.DistributedDataParallel(model_test, device_ids=[gpu])

    

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=head.parameters(), lr=0.02)]
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


    
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model_test.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)



        
    start_epoch_test = 0
    start_time_test = time.time()
    
    for epoch in range(start_epoch_test, args.epochs_test):

        model.eval()
        
        if args.rank == 0:
            
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            
            with torch.no_grad():
                for images, target in test_loader:
                    print(len(images))
                    output = model_test(images.cuda(gpu, non_blocking=True))
                    
                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))

                    
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            
            stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_test_file)

            
        scheduler.step()
        
        if args.rank == 0:
            state = dict(epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
            torch.save(state, args.exp_dir / "checkpoint.pth")





            
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





class Asym_VICReg(nn.Module):
    
    def __init__(self, args):
        
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        
        
        if args.arch_1 == "resnetencoder":
            self.backbone_1, self.embedding_1 = ResnetEncoder()
            
        else:
            self.backbone_1, self.embedding_1 = resnet.__dict__[args.arch_1](zero_init_residual=True)

            
        if args.arch_2 =="resnetencoder":
            self.backbone_2, self.embedding_2 = ResnetEncoder()
            
        else:
            self.backbone_2, self.embedding_2 = resnet.__dict__[args.arch_2](zero_init_residual=True)

            
        self.projector_1 = Projector(args, self.embedding_1)
        self.projector_2 = Projector(args, self.embedding_2)



        
    def forward(self, x, y):
        
        x = self.projector_1(self.backbone_1(x))
        y = self.projector_2(self.backbone_2(y))

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
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (self.args.sim_coeff * repr_loss + self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss)

        
        return loss

    

def Projector(args, embedding):
    
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i+1]))
        layers.append(nn.ReLU(True))
        
    layers.append(nn.Linear(f[-2], f[-1], bias=False))

    
    return nn.Sequential(*layers)





def exclude_bias_and_norm(p):
    return p.ndim == 1




def off_diagonal(x):
    n,m = x.shape
    assert n == m
    
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()






class LARS(optim.Optimizer):

    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001, weight_decay_filter=None, lars_adaptation_filter=None):
        
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta, weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
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

                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.0, torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one), one,)
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


    

    
    
class AverageMeter(object):
    
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
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

        
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Asym_VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)

