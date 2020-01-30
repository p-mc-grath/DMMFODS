# Deep Computer Vision lab course

Task:       Multi-modal (rgb, lidar) object detection  
Dataset:    Waymo Open Dataset   
Framework:  PyTorch

Approach:  
* U-Net like encoder-decoder structure  
* DenseNet used as encoder  
  * lidar data processed in separate first block  
  * concatenation layer added before second block  
* Ground truth bounding boxes processed to segmentation masks
* Loss: each pixel can independently belong to each of the classes (vehicle, pedestrian, cyclist)

Following the structure suggested by:   
https://github.com/moemen95/Pytorch-Project-Template  
Adaptation: Config EasyDict can be created from /utils/Dense_U_Net_lidar_helper  

Based on the torchvision densenet implementation:  
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py  
The original structure is fully respected, allowing to employ the pretrained weights from torchvision  

DenseNet paper:  
https://arxiv.org/abs/1608.06993

U-Net paper:  
https://arxiv.org/abs/1505.04597

