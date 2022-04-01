import functools
from torch import nn
import torch
import torch.nn.functional as F
import math


class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_spectral=False,conv=nn.Conv2d):
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
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        fl = [nn.ReflectionPad2d(3),
                 spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),use_spectral),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model += fl
        
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            dsp = [spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),use_spectral),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            model += dsp

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            resblockl = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,conv=conv)]
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
        output,_ = self.compute_feats(input)
        return output

    def get_feats(self, input, extract_layer_ids=[]):
        _,feats = self.compute_feats(input, extract_layer_ids)
        return feats



    

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral=False,conv=nn.Conv2d):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv = conv
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral):
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
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [spectral_norm(self.conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),use_spectral), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [spectral_norm(self.conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),use_spectral), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


    

    
class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros', norm_layer=nn.InstanceNorm2d, bias=True, scale_factor=1):                                      
        super(SeparableConv2d, self).__init__()                                                                                                                                                           
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=in_channel, bias=bias),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels, kernel_size=1, stride=1, bias=bias),
        )


        
    def forward(self, x):
        return self.conv(x)



class BaseGenerator_attn(nn.Module):

    # initializers
    def __init__(self,nb_mask_attn,nb_mask_input):
        super(BaseGenerator_attn, self).__init__()
        self.nb_mask_attn = nb_mask_attn
        self.nb_mask_input = nb_mask_input

        
    def compute_outputs(self ,input,attentions,images):
        outputs = []

        for i in range(self.nb_mask_attn-self.nb_mask_input):
            outputs.append(images[i]*attentions[i])

        for i in range(self.nb_mask_attn-self.nb_mask_input,self.nb_mask_attn):
            outputs.append(input * attentions[i])

        return images,attentions,outputs



    # forward method
    def forward(self, input):
        feat,_ = self.compute_feats(input)
        attentions,images = self.compute_attention_content(feat)
        _,_,outputs = self.compute_outputs(input,attentions,images)
        o = outputs[0]

        for i in range(1,self.nb_mask_attn):
            o += outputs[i]

        return o



    def get_attention_masks(self,input):
        feat,_ = self.compute_feats(input)
        attentions,images = self.compute_attention_content(feat)
        return self.compute_outputs(input,attentions,images)



    def get_feats(self,input,extract_layer_ids):
        _,feats = self.compute_feats(input, extract_layer_ids)
        return feats





def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
          net (network)      -- the network to be initialized
          init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
          gain (float)       -- scaling factor for normal, xavier and orthogonal.
    Return an initialized network.
    """

    init_weights(net, init_type, init_gain=init_gain)
    return net





def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)



        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>




def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module



def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()                                                                                                                                                                                
