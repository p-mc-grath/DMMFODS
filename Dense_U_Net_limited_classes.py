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

def _create_ground_truth_bb_pedestrian(width, height):
    
    unlikely = 0.3
    uncertain = 0.5
    half_certain = 0.75
    
    height_fraction = height//5
    width_fraction = width//4

    ground_truth_box = np.ones((height, width))
    ground_truth_box[0:height_fraction,:width_fraction] = unlikely
    ground_truth_box[0:height_fraction,width_fraction*3:] = unlikely
    ground_truth_box[height_fraction*3:,:width_fraction] = uncertain
    ground_truth_box[height_fraction*3:,width_fraction*3:] = uncertain
    ground_truth_box[height_fraction*3:,width_fraction:width_fraction*3] = half_certain

    return ground_truth_box

def _create_ground_truth_bb_cyclist(width, height):
    return np.ones((height, width))

def _create_ground_truth_bb_car(width, height):
    return np.ones((height, width))

def _create_ground_truth_bb(object_class, width, height):
    if object_class == 'pedestrian':
        ground_truth_box = _create_ground_truth_bb_pedestrian(width, height)
    elif object_class == 'cyclist':
        ground_truth_box = _create_ground_truth_bb_cyclist(width, height)
    elif object_class == 'car':
        ground_truth_box = _create_ground_truth_bb_car(width, height)
    else:
        raise TypeError('the ground truth label class does not exist')
    return ground_truth_box

# TODO check dimensions
def create_ground_truth_maps(ground_truth, width_img, height_img):
    maps = np.zeros((height_img, width_img, 3))
    
    for _, elem in enumerate(ground_truth):
        
        object_class = elem['class']
        width_bb = elem['width']
        height_bb = elem['height']
        x = elem['x']
        y = elem['y'] 
        if object_class == 'pedestrian':
            object_class_idx = 0
        elif object_class == 'cyclist':
            object_class_idx = 1
        elif object_class == 'car':
            object_class_idx = 2
        else:
            raise TypeError('the ground truth label class does not exist')

        maps[y:y+height_bb, x:x+width_bb, object_class_idx] = _create_ground_truth_bb(object_class, width_bb, height_bb)
        
    return maps       

# Structure basically same as 
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

class DenseNet_with_lidar_branch(nn.Module):
    '''
    - adds blocks to process lidar data seprately
    - adds concat layer to join rgb and lidar features
    - respects struct of orig s.t. can make full use of pretrained pytorch densenet models
    therefore the encoder is called self.features 

    arg: concat_before_block -> concat of lidar and rgb directly before block after transition layer
    arg: num_lidar_blocks -> determine block config for lidar data: same as for rgb 
                          from concat_before_block_num backward taking blocks from config

    version such that trained weights of original densenet can be used without effort
    '''
    # TODO: __init__:   see how many lidar channels
    # TODO: __init__:   see lidar WxH; pad accordingly; adjust where fed into network and how processed before block
    # TODO: __init__:   rescale decoder out to original input size
    # TODO: forward:    check concat dimensions 
    # TODO: forward:    WxH_size_stack cut channels 
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,          # orig
                num_lidar_in_channels=12, concat_before_block_num=2, num_layers_before_blocks=4,    # new                                                     # new
                bn_size=4, drop_rate=0, num_classes=3, memory_efficient=True):                      # orig     
        assert concat_before_block_num >= 2, 'concatinating before first block is not allowed'
        assert growth_rate % 2 == 0, 'growth_rate needs to be an even number'
        assert num_init_features % 2 == 0, 'num_init_features needs to be an even number'
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
                    ('conv0', nn.Conv2d(num_lidar_in_channels, num_features, kernel_size=5, 
                                stride=1, padding=2, bias=False))
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

        # decoder; ugly: should have own class for whole sequence -> TODO clean up
        self.decoder = nn.Sequential()
        num_in_features = feature_size_stack.pop()
        for i in range(len(block_config)-1):                            
            num_features = feature_size_stack.pop()
            transp_conv_seq = nn.Sequential(OrderedDict([                   # denselayer like struct; reduce channels with 1x1 convs
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
        WxH_size_stack = deque()
        
        # encoding
        lidar_features = self.lidar_features(lidar_data)
        features = rgb_data
        for i, enc_module in enumerate(self.features): 
            # enc
            features = enc_module(features)  
            
            # concat lidar and rgb
            if i == self.num_layers_before_blocks + 2**(self.concat_before_block_num-2)+1: 
                features = torch.cat((features, lidar_features), 0)                 # TODO check dim
                features = self.concat_module(features)

            # save features for decoder in stack
            if isinstance(enc_module, _DenseBlock):
                features_from_enc_stack.append(features)                            
                WxH_size_stack.append(features.size())                                  # TODO cut channels out
        
        # decoding; ugly quick and dirty implementation -> TODO clean up
        for i, dec_module in enumerate(self.decoder):
            if not isinstance(dec_module, nn.ConvTranspose2d):                          # sequence without TransposedConv
                if i > 0:                                                               # concat upsampled data and data from encoder
                    features = torch.cat((features, features_from_enc_stack.pop()), 0)  # TODO check dim
                features = dec_module(features)
            else:                                                                       # TransposedConv
                features = dec_module(features, output_size=WxH_size_stack.pop())       

        features = self.dec_out_to_heat_maps(features)

        return features

# load pretrained state dict
def _load_state_dict(model, model_url, state_dict_checkpoint, progress):
    
    # load from torchvision server
    if state_dict_checkpoint is None:
        
        # copied from https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
        state_dict_checkpoint = load_state_dict_from_url(model_url, progress=progress)
        
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

def _densenet_with_lidar_branch(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              gdrive_checkpoint, **kwargs):
    model = DenseNet_with_lidar_branch(growth_rate, block_config, num_init_features, **kwargs)
    
    if pretrained:
        _load_checkpoint(model, model_urls[arch], gdrive_checkpoint, progress)
        
    return model


def densenet121_with_lidar_branch(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet_with_lidar_branch('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     gdrive_checkpoint, **kwargs)


def densenet161_with_lidar_branch(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet_with_lidar_branch('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     gdrive_checkpoint, **kwargs)


def densenet169_with_lidar_branch(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet_with_lidar_branch('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     gdrive_checkpoint, **kwargs)


def densenet201_with_lidar_branch(pretrained=False, gdrive_checkpoint=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet_with_lidar_branch('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     gdrive_checkpoint, **kwargs)