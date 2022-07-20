# VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning

This repository provides a PyTorch implementation and pretrained models for VICReg, as described in the paper [VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning](https://arxiv.org/abs/2105.04906)\
Adrien Bardes, Jean Ponce and Yann LeCun\
Meta AI, Inria

--- 

<!-- 
<div align="center">
  <img width="100%" alt="VICReg illustration" src=".github/vicreg_archi_full.jpg">
</div> -->

<p align="center">
<img src=".github/vicreg_archi_full.jpg" width=100% height=100% 
class="center">
</p>


## Training

### Denoising Autoencoder

To pretrain a denoising autoencoder with ResNet, run:

```
python -m torch.distributed.launch denoising_autoencoder.py --data-dir /path/to/dataset/ --exp-dir /path/to/experiment/ --epochs 100 --batch-size 64 --base-lr 0.3 
```

### VICReg with decoder head

To pretrain VICReg with ResNet and a decoder head, run:

```
python -m torch.distributed.launch main_decoder.py --data-dir /path/to/dataset/ --exp-dir /path/to/experiment/ --epochs 100 --batch-size 64 --base-lr 0.3 --vic-coeff 10 --dec-coeff 5
```

### Segformer

To pretrain VICReg with a segformer with b5 configuration, run:
```
python -m torch.distributed.launch segformer.py --data-dir /path/to/dataset/ --exp-dir /path/to/experiment/ --epochs 100 --batch-size 64 --base-lr 0.3 --arch b5 --embedding 512
```


## Evaluation

### Linear evaluation

To evaluate a pretrained ResNet-50 backbone on linear classification on 10% of the dataset labels, run:

```
python evaluate.py --data-dir /path/to/imagenet/ --pretrained /path/to/checkpoint/resnet50.pth --exp-dir /path/to/experiment/ --arch resnet50 --train-perc 10 --epochs 20 --lr-head 0.02
```

### Semi-supervised evaluation

To evaluate a pretrained encoder-model on semi-supervised fine-tunning on 1% of dataset labels, run:

```
python evaluate.py --data-dir /path/to/dataset/ --pretrained /path/to/experiment/encoder.pth --exp-dir /path/to/checkpoint/ --arch encoder --weights finetune --train-perc 1 --epochs 20 --lr-head 0.02
```

To evaluate a pretrained segformer-model on semi-supervised fine-tuning on 0.1% of dataset labels, run:

```
python evaluate.py --data-dir /path/to/imagenet/ --pretrained /path/to/experiment/segformer.pth --exp-dir /path/to/checkpoint/ --arch segformer --weights finetune --train-perc zero1 --epochs 20 --lr-head 0.02
```


## Acknowledgement

This repository is built using the [Barlow Twins](https://github.com/facebookresearch/barlowtwins) repository.

## License

This project is released under MIT License, which allows commercial use. See [LICENSE](LICENSE) for details.

## Citation
If you find this repository useful, please consider giving a star :star: and citation:

```
@inproceedings{bardes2022vicreg,
  author  = {Adrien Bardes and Jean Ponce and Yann LeCun},
  title   = {VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning},
  booktitle = {ICLR},
  year    = {2022},
}
```
