import numpy as np
from torchvision.models.densenet import model_urls, _DenseLayer, _DenseBlock, _Transition
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import re
import torch.distributed as dist
import os
from collections import OrderedDict
from collections import deque 

# Structure basically same as 
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

class Dense_U_Net_lidar(nn.Module):
    '''
    * adds blocks to process lidar data seprately
    * adds concat layer to join rgb and lidar features
    * replaces classifier with decoder and results in heatmap for each class

    If  - growth_rate       as in original and pretrained=True
        - block_config
        - num_init_features
    then pytorch pretrained densenet weights are used (therefore the encoder is called self.features )   

    Args:   concat_before_block -> concat of lidar and rgb directly before block after transition layer
            assert: concat_before_block>=2
    
    for forward pass if concat_before_block_num=2: (W_lidar, H_lidar) == (W_rgb/4, H_rgb/4) 
    
    potential mods: (1) use max pool in transition layers instead of avg pool
                        with return_indices=True 
                        use maxunpool2d with corresponding indices in upsampling
                        !weights still same because no weight op!
                    (2) Processing block in the decoder
                    (3) Use pre block features in decoder
                    (4) add possiblity for multiple lidar blocks
    '''

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,          # orig
                num_lidar_in_channels=1, concat_before_block_num=2, num_layers_before_blocks=4,     # new                                                     # new
                bn_size=4, drop_rate=0, num_classes=3, memory_efficient=True):                      # orig     

        # block config should also only contain even numbers

        super().__init__()

        self.concat_before_block_num = concat_before_block_num
        self.block_config = block_config
        self.num_layers_before_blocks = num_layers_before_blocks
        self.best_checkpoint_path = '/content/notebooks/DeepCV_Packages/best_checkpoint.pth.tar'
        self.epoch = None
        self.loss = None
        self.optimizer = None

        # First convolution rgb
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # amount and size of denseblocks dictate num_features after init
        num_features = num_init_features
        feature_size_stack = deque()
        for i, num_layers in enumerate(block_config):

            # concat layer; 1x1 conv
            if i == concat_before_block_num-1:
                self.concat_module = nn.Sequential(OrderedDict([
                ('norm', nn.BatchNorm2d(num_features*2)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(num_features*2, num_features, kernel_size=1, 
                                stride=1, padding=0, bias=False))
            ]))

            # first convolution and denseblock lidar
            if i == concat_before_block_num-2:
                self.lidar_features = nn.Sequential(OrderedDict([
                    ('norm0', nn.BatchNorm2d(num_lidar_in_channels)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('conv0', nn.Conv2d(num_lidar_in_channels, num_init_features//2, kernel_size=7, 
                                stride=1, padding=3, bias=False)),
                    ('norm1', nn.BatchNorm2d(num_init_features//2)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('conv1', nn.Conv2d(num_init_features//2, num_features, kernel_size=3, 
                                stride=1, padding=1, bias=False))
                ]))
                block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient
                )
                self.lidar_features.add_module('lidar_denseblock%d' % (i + 1), block)
                trans = _Transition(num_input_features=num_features + num_layers * growth_rate,
                                    num_output_features=(num_features + num_layers * growth_rate) // 2)
                self.lidar_features.add_module('lidar_transition%d' % (i + 1), trans)

            # denseblocks rgb
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            feature_size_stack.append(num_features)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # U net like decoder blocks up to WxH of first block; ugly: should have own class for whole sequence 
        self.decoder = nn.Sequential()
        num_in_features = feature_size_stack.pop()
        for i in range(len(block_config)-1):                            
            num_features = feature_size_stack.pop()
            transp_conv_seq = nn.Sequential(OrderedDict([                                           # denselayer like struct; reduce channels with 1x1 convs
                            ('norm1', nn.BatchNorm2d(num_in_features)),
                            ('relu1', nn.ReLU(inplace=True)),
                            ('conv_reduce', nn.Conv2d(num_in_features, num_features, kernel_size=1, stride=1,
                                padding=0, bias=False)),
                            ('norm2', nn.BatchNorm2d(num_features)),
                            ('relu2', nn.ReLU(inplace=True))
            ]))
            self.decoder.add_module('Transposed_Convolution_Sequence_%d' %(i+1), transp_conv_seq)
            self.decoder.add_module('Transposed_Convolution_%d' %(i+1), nn.ConvTranspose2d(num_features, num_features, kernel_size=3))
            num_in_features = num_features*2
        
        # stepwise reducing channels to 3d map
        # TODO rescale to original input size
        self.dec_out_to_heat_maps = nn.Sequential(OrderedDict([                   
                            ('norm1', nn.BatchNorm2d(num_features)),
                            ('relu1', nn.ReLU(inplace=True)),
                            ('conv1', nn.Conv2d(num_features, 64, kernel_size=3, stride=1,
                                padding=1, bias=False)),
                            ('norm2', nn.BatchNorm2d(64)),
                            ('relu2', nn.ReLU(inplace=True)),
                            ('conv2', nn.Conv2d(64, num_classes, kernel_size=3, stride=1,
                                padding=1, bias=False)),
                            ('out_sigmoid', nn.Sigmoid())
            ]))

        ### Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.num_params = sum(p.numel() for p in self.parameters())

    # unnecessarily complex because init respects original densenet implementation; see __init__
    def forward(self, rgb_data, lidar_data):
        
        features_from_enc_stack = deque()
        HxW_shape_stack = deque()
        
        # encoding
        lidar_features = self.lidar_features(lidar_data)
        features = rgb_data
        for i, enc_module in enumerate(self.features): 
            # enc
            features = enc_module(features)  
            
            # concat lidar and rgb
            if i == self.num_layers_before_blocks + 2**(self.concat_before_block_num-2)+1: 
                assert features.size() == lidar_features.size(), 'lidar and rgb data dim mismatch'
                features = torch.cat((features, lidar_features), 1)              
                features = self.concat_module(features)

            # save features for decoder in stack
            if isinstance(enc_module, _DenseBlock):
                features_from_enc_stack.append(features)                            
                HxW_shape_stack.append(features.size())                                 
        
        # decoding; ugly quick and dirty implementation 
        for i, dec_module in enumerate(self.decoder):
            if not isinstance(dec_module, nn.ConvTranspose2d):                                      # sequence without TransposedConv
                if i > 0:                                                                           # concat upsampled data and data from encoder
                    features = torch.cat((features, features_from_enc_stack.pop()), 1)  
                features = dec_module(features)
            else:                                                                                   # TransposedConv
                features = dec_module(features, output_size=HxW_shape_stack.pop())       

        features = self.dec_out_to_heat_maps(features)

        return features

# load pretrained state dict
def _load_state_dict(model, model_url, state_dict_checkpoint, progress):
    
    # load from torchvision server
    if state_dict_checkpoint is None:
        
        state_dict_checkpoint = load_state_dict_from_url(model_url, progress=progress)
        
        # copied from mmdetection
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict_checkpoint.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict_checkpoint[new_key] = state_dict_checkpoint[key]
                del state_dict_checkpoint[key]

        # remove classifier from pretrained model TODO necessary? add rmv norm5?
        del state_dict_checkpoint['classifier.weight']
        del state_dict_checkpoint['classifier.bias']

    # update state dict of model with retrieved pretrained values & load state_dict into model
    state_dict_model = model.state_dict()
    state_dict_model.update(state_dict_checkpoint) 
    model.load_state_dict(state_dict_model, strict=False)

def _load_checkpoint(model, model_url, gdrive_checkpoint, progress):

    # load checkpoint from gdrive
    if gdrive_checkpoint:
        checkpoint = torch.load(model.best_checkpoint_path)
        _load_state_dict(model, model_url, checkpoint['model_state_dict'], progress)
        if model.optimizer is not None:
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.epoch = checkpoint['epoch']
        model.loss = checkpoint['loss']
    else:
        # get pretrained state_dict from pytorch  
        _load_state_dict(model, model_url, None, progress)
        
    # set model to training 
    model.train()

def _dense_u_net_lidar(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              gdrive_checkpoint, **kwargs):
    model = Dense_U_Net_lidar(growth_rate, block_config, num_init_features, **kwargs)
    
    if pretrained:
        _load_checkpoint(model, model_urls[arch], gdrive_checkpoint, progress)
        
    return model


def densenet121_u_lidar(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     gdrive_checkpoint, **kwargs)


def densenet161__u_lidar(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     gdrive_checkpoint, **kwargs)


def densenet169_u_lidar(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     gdrive_checkpoint, **kwargs)


def densenet201_u_lidar(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     gdrive_checkpoint, **kwargs)