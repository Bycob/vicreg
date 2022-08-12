#!/usr/bin/python3

import sys
import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn

import mmcv
import mmseg
from mmseg.models import build_segmentor


def main():
    parser = argparse.ArgumentParser(
        description="Trace a segformer trained with vicreg"
    )
    parser.add_argument("files", type=str, nargs="*", help="Input files")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to INFO"
    )
    parser.add_argument("--weights", help="File containing pretrained weights")
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size to trace the model"
    )
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    device = "cpu"

    for fname in args.files:
        # create segformer
        cfg = mmcv.Config.fromfile("segformer_config_b5.py")
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.decode_head.num_classes = args.num_classes
        net = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))

        # load weights
        state_dict = torch.load(args.weights, map_location=device)
        state_dict = {
            key.replace("module.net.", ""): state_dict["model"][key]
            for key in state_dict["model"]
            if "module.net." in key
        }
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
        assert missing_keys == [] and unexpected_keys == [], (
            str(missing_keys) + "\n\n\n" + str(unexpected_keys)
        )

        # trace model
        input_nc = 3
        model_out_file = os.path.join(args.output_dir, "segformer_b5_vicreg.pt")
        pytorch2libtorch(
            net, (1, 3, args.img_size, args.img_size), output_file=model_out_file
        )


# ====


def exclude_bias_and_norm(p):
    return p.ndim == 1


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.
    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    segs = rng.randint(low=0, high=num_classes - 1, size=(N, 1, H, W)).astype(np.uint8)
    img_metas = [
        {
            "img_shape": (H, W, C),
            "ori_shape": (H, W, C),
            "pad_shape": (H, W, C),
            "filename": "<demo>.png",
            "scale_factor": 1.0,
            "flip": False,
        }
        for _ in range(N)
    ]
    mm_inputs = {
        "imgs": torch.FloatTensor(imgs).requires_grad_(True),
        "img_metas": img_metas,
        "gt_semantic_seg": torch.LongTensor(segs),
    }
    return mm_inputs


def pytorch2libtorch(
    model, input_shape, show=False, output_file="tmp.pt", verify=False
):
    """Export Pytorch model to TorchScript model and verify the outputs are
    same between Pytorch and TorchScript.
    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the
            output TorchScript model. Default: `tmp.pt`.
        verify (bool): Whether compare the outputs between
            Pytorch and TorchScript. Default: False.
    """
    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop("imgs")

    # replace the original forword with forward_dummy
    model.forward = model.forward_dummy
    model.eval()
    traced_model = torch.jit.trace(
        model,
        example_inputs=imgs,
        check_trace=verify,
    )

    if show:
        print(traced_model.graph)

    traced_model.save(output_file)
    print("Successfully exported TorchScript model: {}".format(output_file))


def stuff(args):
    pass


if __name__ == "__main__":
    main()
