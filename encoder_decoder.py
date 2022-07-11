import functools
import math

from torch import nn
import torch
import torch.nn.functional as F


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

def norm_layer(x):
            return Identity()


        
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        use_spectral=False,
        conv=nn.Conv2d,
    ):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv = conv
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral
    ):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zeros":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            spectral_norm(
                self.conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                use_spectral,
            ),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zeros":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            spectral_norm(
                self.conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                use_spectral,
            ),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

    

class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        use_spectral=False,
        conv=nn.Conv2d,
    ):
        """Construct a Resnet-based encoder
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetEncoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        fl = [
            # nn.ReflectionPad2d(3),
            spectral_norm(
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                use_spectral,
            ),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        model += fl

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            dsp = [
                spectral_norm(
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    use_spectral,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            model += dsp

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            resblockl = [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    conv=conv,
                )
            ]
            model += resblockl

        self.model = nn.Sequential(*model)

    def compute_feats(self, input, extract_layer_ids=[]):
        if -1 in extract_layer_ids:
            extract_layer_ids.append(len(self.encoder))
        feat = input
        feats = []
        for layer_id, layer in enumerate(self.model):

            feat = layer(feat)
            if layer_id in extract_layer_ids:
                feats.append(feat)
        return feat, feats  # return both output and intermediate features

    def forward(self, input):
        """Standard forward"""
        output, _ = self.compute_feats(input)
        return output

    def get_feats(self, input, extract_layer_ids=[]):
        _, feats = self.compute_feats(input, extract_layer_ids)
        return feats


class ResnetDecoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        use_spectral=False,
    ):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetDecoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                spectral_norm(
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    use_spectral,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        p = 0
        if padding_type == "reflect":
            model += [nn.ReflectionPad2d(5)]
        elif padding_type == "replicate":
            model += [nn.ReplicationPad2d(5)]
        elif padding_type == "zeros":
            p = 5
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=p)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        return output
