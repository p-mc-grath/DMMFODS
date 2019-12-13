import numpy as np 
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

import logging
from collections import OrderedDict

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.runner import load_state_dict
import mmcv.runner

from ..registry import BACKBONES
from ..utils import ConvModule
from .resnet import BasicBlock
from .resnext import Bottleneck
from .resnext import ResNeXt

# Changes to the original functions are clearly marked with CHANGE TO ORIGINAL comment
@BACKBONES.register_module
class ResNeXt_with_lidar_branch(ResNeXt):
    """ResNeXt Lidar backbone.
    Args:
        see ResNeXt
        lidar_in_channels: number of input channels to the lidar branch
        fusion_block: resnet block after which lidar data is incorporated
            into the main branch
        dense_block_indices: np.array containing the numbers of layers 
            being fed the downsized original input
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    ## Build ResNeXt original 
    # Only change is double input size into fusion layer
    def __init__(self,
                 lidar_in_channels=4*3,
                 fusion_block=0,
                 dense_block_indices=None,
                 **kwargs):

        # ORIGINAL
        super().__init__(**kwargs)

        # CHANGE TO ORIGINAL: additional object variables
        self.lidar_in_channels = lidar_in_channels      
        self.fusion_block = fusion_block
        self.dense_block_indices = dense_block_indices
        self.lidar_kernel = 3
        self.fusion_kernel = 3
        
        # ADDED: Lidar branch
        inplanes = self.lidar_in_channels       
        outplanes = 64
        self.lidar_layers = []
        
        counter = 0
        while(outplanes <= 64 * 2**self.fusion_block):
            lidar_layer = ConvModule(
                inplanes,
                outplanes,
                self.lidar_kernel,
                stride=1,
                padding=(self.lidar_kernel-1)/2)        # same
            layer_name = 'lidar_layer{}'.format(counter + 1)
            self.add_module(layer_name, lidar_layer)
            self.lidar_layers.append(layer_name)

            counter += 1
            inplanes = outplanes 
            outplanes *= 2**counter

        # ADDED: fusion layer
        if not self.fusion_block in self.dense_block_indices:
            outplanes = 64 * 2**self.fusion_block
            inplanes = outplanes * 2        # want lidar and rgb to have same amount of channels
            fusion_layer = ConvModule(
                    inplanes,
                    outplanes,
                    self.fusion_kernel,
                    stride=1,
                    padding=(self.fusion_kernel-1)/2    # same
                    )
            layer_name = 'fusion_layer'
            self.add_module(layer_name, fusion_layer)
            self.fusion_layer = layer_name

        # ADDED: dense layers
        self.dense_layers = []
        for i in self.dense_block_indices:
            outplanes = 64 * 2**i
            inplanes = self.lidar_in_channels*2 + outplanes
            # too messy if combining fusion layer into dense layer; now extra
            #if i == self.fusion_block:
            #    inplanes += self.lidar_in_channels*2
            dense_layer = ConvModule(
                inplanes,
                outplanes,
                self.fusion_kernel,
                stride=1,
                padding=(self.fusion_kernel-1)/2    # same
                )
            layer_name = 'dense_layer{}'.format(i)
            self.add_module(layer_name, dense_layer)
            self.dense_layers.append(layer_name)      

        self._freeze_stages()

    # (1) load model into checkpoint
    # (2) extract state_dict
    # (3) change state dict
    # (4) put into model and save
    def create_pretrained_base_model(self, filename, map_location, strict=False, logger=None):
        if not filename.startswith('torchvision//'):
            raise TypeError('the model name passed to weight init is not supported')

        # (1)download torchvision model
        model_urls = mmcv.runner.get_torchvision_models()
        model_name = filename[14:]
        checkpoint = mmcv.runner.load_url_dist(model_urls[model_name])
        
        # (2) get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict_checkpoint = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict_checkpoint = checkpoint['state_dict']
        else:
            raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
        # strip prefix of state_dict
        if list(state_dict_checkpoint.keys())[0].startswith('module.'):
            state_dict_checkpoint = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        # (3) TODO fusion layer add additional weights and biases 
        # first init everything
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        # then update layers given by downloaded model
        state_dict_model = self.state_dict()
        state_dict_model.update(state_dict_checkpoint) 

        # (4) load state_dict into model
        load_state_dict(self.module, state_dict_model, strict=False, logger=logger)

        # save model 
        checkpoint = torch.save(filename, map_location)

        return filename, logger
        
    # Use Original mainly
    def init_weights(self, checkpoint_source=None, pretrained=None, map_location=None):     
        ### Basically ORIGINAL resnet weight_init 
        ### added: allowing loading checkpoint from file
        logger = logging.getLogger()

        # if first time init; need to create parameters from some pretrained and what is added
        if checkpoint_source == 'TORCHVISION':
            pretrained, logger = self.create_pretrained_base_model(pretrained, map_location, logger=logger)

        # load checkpoint from gdrive
        elif checkpoint_source == 'GDRIVE':
            load_checkpoint(self, pretrained, map_location=map_location, strict=False, logger=logger)

        # initialize weights from scratch
        # probably irrelevant
        elif checkpoint_source is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('checkpoint_source must be GDRIVE, TORCHVISION or None')
        

    # apply maxpooling s.t. input size same as current iteration of processed data
    def denselike_concat(self, x, rgb_data, lidar_data):
        
        # get size
        width_x = np.shape(x)[0]
        width_rgb_data = np.shape(rgb_data)[0]
        width_lidar_data = np.shape(lidar_data)[0]

        # Pad data such that repeated maxpooling results in desired size
        size_difference = width_rgb_data % width_x
        padding = width_x - size_difference
        p = nn.ConstantPad2d(padding, 0)
        rgb_data = p(rgb_data)

        size_difference = width_lidar_data % width_x
        padding = width_x - size_difference
        p = nn.ConstantPad2d(padding, 0)
        lidar_data = p(lidar_data)

        # maxpool until size is same
        m = nn.MaxPool2d(2,2)
        while(width_x<width_lidar_data):
            lidar_data = m(lidar_data)
        while(width_x<width_rgb_data):
            rgb_data = m(rgb_data)
        
        x = torch.cat((x, rgb_data, lidar_data), 0)

        return x, rgb_data, lidar_data
    
    # Very simple forward pass
    def _lidar_forward(self, x):

        outs = []
        for i, layer_name in enumerate(self.lidar_layers):
            
            lidar_layer = getattr(self, layer_name)
            x = lidar_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return x, tuple(outs)

    # VERY CLOSE TO ORIGINAL
    # added lidar data forward pass
    # added concat; data fusion
    def forward(self, rgb_data, lidar_data):

        # CHANGE TO ORIGINAL: FORWARD PASS LIDAR
        lidar_data_pre_proc, lidar_outs = self._lidar_forward(lidar_data)

        x = self.conv1(rgb_data)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            
            # CHANGE TO ORIGINAL: 
            # concatenate processed lidar data and resnet's mainstream i.e. processed rgb data
            # pass through fusion_layer
            if i == self.fusion_block:
                x = torch.cat((lidar_data_pre_proc, x), 0)
                fusion_layer = getattr(self, self.fusion_layer)
                x = fusion_layer(x)
            
            # CHANGE TO ORGINAL: Added denseblock-like structure
            # directly concat and feed orignial input 
            if i in self.dense_block_indices:
                x, rgb_data, lidar_data = self.denselike_concat(x, rgb_data, lidar_data)
                dense_layer_name = 'dense_layer{}'.format(i)
                dense_layer = getattr(self, dense_layer_name)
                x = dense_layer(x)

            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return lidar_outs + tuple(outs)
