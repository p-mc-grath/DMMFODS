import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torchvision.models.densenet import model_urls, _DenseLayer, _DenseBlock, _Transition
from torchvision.models.utils import load_state_dict_from_url
from collections import OrderedDict
from collections import deque 
from ...utils.Dense_U_Net_lidar_helper import get_config 

# Structure of code essentially: 
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

class Dense_U_Net_lidar(nn.Module):
    '''
    U-Net like structure | Encoder = DenseNet original | optional secondary stream in     encoder processing lidar data

    Keeping the structure and variable namung of original densenet | allowing to use pretrained weigthts from torchvision.models
    1. Added optional lidar stream mirroring the densenet rgb stream
    2. Added optional Concat layer to bring together rgb and lidar
    3. replacing the classifier with UNet like Decoder | feeding in output of blocks | if split streams using rgb stream
    4. Output: Heat Maps for each class

    Args:   
        check original densenet implementation
        concat_before_block: concat of lidar and rgb directly before block after transition layer
    '''
    def __init__(self, config):
    
        super().__init__()

        self.config = config

        # original densenet attributes
        self.growth_rate = config.model.growth_rate 
        self.block_config = config.model.block_config
        self.num_init_features = config.model.num_init_features          
        self.bn_size = config.model.bn_size
        self.drop_rate = config.model.drop_rate
        self.memory_efficient = config.model.memory_efficient  
        self.num_classes = config.model.num_classes        

        # param assignment
        self.concat_before_block_num = config.model.concat_before_block_num
        self.num_layers_before_blocks = config.model.num_layers_before_blocks
        self.num_lidar_in_channels = config.model.num_lidar_in_channels
        self.num_rgb_in_channels = config.model.num_rgb_in_channels        
        self.network_input_channels = self.num_rgb_in_channels                                               # Allowing for rgb input or torch.cat((rgb,lidar),1) | added
        if self.concat_before_block_num == 1 and self.num_lidar_in_channels == 0:
            self.fusion = 'no'
        elif self.concat_before_block_num == 1 and self.num_lidar_in_channels > 0:
            self.fusion = 'early'
            self.network_input_channels += self.num_lidar_in_channels
        elif self.concat_before_block_num > 1 and self.concat_before_block_num <= len(self.block_config):
            self.fusion = 'mid'
        else:
            raise AttributeError
        
        ### core structure
        
        ## Encoder | same as densenet without norm5 and classifier 

        # First convolution | original densenet 
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.network_input_channels, self.num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(self.num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each denseblock | original densenet + stack comprising layer sizes for the decoder
        feature_size_stack = deque()
        feature_size_stack.append(self.num_init_features + 2*self.growth_rate) 
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                drop_rate=self.drop_rate,
                memory_efficient=self.memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate
            feature_size_stack.append(num_features)
            if i != len(self.block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        ## Decoder
        # U net like | ugly: should have own class for whole sequence 
        self.decoder = nn.Sequential()
        num_in_features = feature_size_stack.pop()
        for i in range(len(self.block_config)):                            
            num_features = feature_size_stack.pop()
            transp_conv_seq = nn.Sequential(OrderedDict([                                                   # denselayer like struct; reduce channels with 1x1 convs
                ('norm0', nn.BatchNorm2d(num_in_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv_reduce', nn.Conv2d(num_in_features, num_features, 
                    kernel_size=1, stride=1, padding=0, bias=False)),
                ('norm1', nn.BatchNorm2d(num_features)),
                ('relu1', nn.ReLU(inplace=True))
            ]))
            self.decoder.add_module('Transposed_Convolution_Sequence_%d' %(i+1), transp_conv_seq)
            self.decoder.add_module('Transposed_Convolution_%d' %(i+1), nn.ConvTranspose2d(num_features, 
                num_features, 3, stride=2, padding=1))
            num_in_features = num_features*2
        self.decoder.add_module('Upsampling', nn.Upsample(scale_factor=2))
        
        # final refinement: concat orig rgb & lidar before passing
        self.dec_out_to_heat_maps = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(num_features+self.num_rgb_in_channels+self.num_lidar_in_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('refine0', nn.Conv2d(num_features+self.num_rgb_in_channels+self.num_lidar_in_channels, 
                num_features//2, 3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_features//2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('refine1', nn.Conv2d(num_features//2, self.num_classes, 
                5, stride=1, padding=2, bias=False))
            ]))

        ### additional structure depending on fusion mechanism
        if self.fusion == 'no':
            # i.e. one stream only

            pass

        elif self.fusion == 'early':
            # i.e. concat rgb and lidar before network

            pass

        elif self.fusion == 'mid':
            # add all the same processing for the lidar data as for rgb data
            # add concat layer

            # First convolution | original densenet | for lidar block  
            self.lidar_features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(self.num_lidar_in_channels, self.num_init_features, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(self.num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

            # Each denseblock | original densenet + break before concat layer
            num_features = self.num_init_features
            for i, num_layers in enumerate(self.block_config):
                if i == self.concat_before_block_num-1:
                    break
                block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                drop_rate=self.drop_rate,
                memory_efficient=self.memory_efficient
                )
                self.lidar_features.add_module('denseblock%d' % (i + 1), block)
                num_features = num_features + num_layers * self.growth_rate
                if i != len(self.block_config) - 1:
                    trans = _Transition(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                    self.lidar_features.add_module('transition%d' % (i + 1), trans)
                    num_features = num_features // 2
            
            # concat layer | rgb + lidar | 1x1 conv
            num_features = self.features[self.concat_before_block_num*2-1 + self.num_layers_before_blocks-1].denselayer1.norm1.num_features
            self.concat_module = nn.Sequential(OrderedDict([
                ('norm', nn.BatchNorm2d(num_features*2)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(num_features*2, num_features, kernel_size=1, 
                                stride=1, padding=0, bias=False))
            ]))

        else:
            raise AttributeError

        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # get number of parameters of model
        self.num_params = sum(p.numel() for p in self.parameters())

    '''
    unnecessarily complex because init respects original densenet implementation; see __init__

    Arguments:
        rgb_data: batch, channels, W, H (W and H as lidar_data!)
        lidar_data: batch, channels, W, H (W and H as rgb_data!)
    '''
    def forward(self, rgb_data, lidar_data):
        
        # stack of encoding features used in decoder
        HxW_shape_stack = deque()
        features_from_enc_stack = deque()

        # assigning name
        if self.fusion == 'no':                                                                     # allowing net to work with lidar only
            features_from_enc_stack.append(rgb_data)
            features = rgb_data
        elif self.fusion == 'early':
            features_from_enc_stack.append(torch.cat((rgb_data, lidar_data), 1))
            features = torch.cat((rgb_data, lidar_data), 1)
        elif self.fusion == 'mid':
            features_from_enc_stack.append(torch.cat((rgb_data, lidar_data), 1))
            features = rgb_data
            lidar_features = self.lidar_features(lidar_data)
        else:
            raise AttributeError

        # encoding
        for i, enc_module in enumerate(self.features): 
            features = enc_module(features)                                                         # encode

            # concat lidar and rgb after transition
            if self.fusion == 'mid' and i == self.num_layers_before_blocks-1 + 2*(self.concat_before_block_num-1): 
                assert features.size() == lidar_features.size(), str(features.size()) + ' ' + str(lidar_features.size())
                features = torch.cat((features, lidar_features), 1)              
                features = self.concat_module(features)

            # save features for decoder in stack
            if i == self.num_layers_before_blocks-2:                                                # get size before maxpool before first block
                HxW_shape_stack.append(features.size())
            if isinstance(enc_module, _DenseBlock) and i<len(self.features)-1:                      # only blocks but skip last
                features_from_enc_stack.append(features)                            
                HxW_shape_stack.append(features.size())                             
                               
        # decoding | ugly quick and dirty implementation 
        for i, dec_module in enumerate(self.decoder):
            if not isinstance(dec_module, nn.ConvTranspose2d):                                      
                if i > 0 and not isinstance(dec_module, nn.Upsample):                                                                           # concat upsampled data and data from encoder
                    features = torch.cat((features, features_from_enc_stack.pop()), 1)              # concat
                features = dec_module(features)                                                     # decode
            else:                                                                                   
                features = dec_module(features, output_size=HxW_shape_stack.pop())                  # decode                                  # decode
        
        # scale to heat maps
        features = torch.cat((features, features_from_enc_stack.pop()), 1)
        features = self.dec_out_to_heat_maps(features)

        return features

def _load_state_dict(model, config, model_url, progress):
    '''
    load pretrained densenet state dict from torchvision into model
    
    copy from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    !!added sencond part before last line; that's why cannot simply import function 
    '''
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict_torchvision = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict_torchvision.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict_torchvision[new_key] = state_dict_torchvision[key]
            del state_dict_torchvision[key]

    ### ADDED pytorch version such that it fits the Dense_U_Net_lidar

    # remove state dict keys that are unnecessary/ different in this implementation
    if model.fusion == 'early' or model.num_rgb_in_channels != 3:
        del state_dict_torchvision['features.conv0.weight']
    del state_dict_torchvision['features.norm5.weight']
    del state_dict_torchvision['classifier.weight']
    del state_dict_torchvision['classifier.bias']

    # update state dict of model with retrieved pretrained values & load state_dict into model
    state_dict_model = model.state_dict()
    state_dict_model.update(state_dict_torchvision) 
    model.load_state_dict(state_dict_model, strict=False)

def _dense_u_net_lidar(arch, growth_rate, block_config, num_init_features, pretrained, progress,
            config):
    if config is None:
        config = get_config(os.path.join('content', 'mnt', 'My Drive', 'Colab Notebooks', 'DeepCV_Packages'))
    
    # for compatibility with densenet original functions
    config.model.growth_rate = growth_rate
    config.model.block_config = block_config
    config.model.num_init_features = num_init_features

    model = Dense_U_Net_lidar(config)
    
    if pretrained:
        _load_state_dict(model, config, model_urls[arch], progress)
        
    return model


def densenet121_u_lidar(pretrained=False, progress=True, config=None):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                    config)


def densenet161_u_lidar(pretrained=False, progress=True, config=None):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                    config)


def densenet169_u_lidar(pretrained=False, progress=True, config=None):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                    config)


def densenet201_u_lidar(pretrained=False, progress=True, config=None):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _dense_u_net_lidar('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                    config)
