# A Deep Multi-Modal Fusion Architecture for Heat Map-Based Object Detection using Segmentation

Task:       Multi-modal (rgb, lidar) object detection  
Dataset:    Waymo Open Dataset   
Framework:  PyTorch

Approach:  
* U-Net like encoder-decoder structure  
* DenseNet used as encoder | optional parallel encoder (stream_2) mirroring the structure of the standard encoder (stream_1), allowing to process different modalities separately up to the selected level of fusion
  * no-fusion: stream_1 processed only 
  * early-fusion: stream_1 only | input size = stream_1_in_channels + stream_2_in_channels
  * mid-fusion: parallel densenet-like | stream_1 and stream_2 
    * concatenation layer added before a block of choice 
* Ground truth bounding boxes processed to segmentation masks | class-wise heat maps
* Loss: each pixel can independently belong to each of the classes (vehicle, pedestrian, cyclist)
* Lidar data projected into image like tensor with zero values where no data  

Tutorial:
1. download Colab_Setup.ipnb into your googledrive
2. open Colab_Setup and adjust paths in first cell to your liking
3. run the INSTALLATION and DATA sections
4. In the first cell of TRAINING, give the path to the directory containing the directory containing the deepcvlab repo to the get_config call (I know...)

Note:
This work is fully compatible with Colab given your GDrive is big enough. Due to being contraint to Colab, a lot of subdirectories are being used. For more information see Colab_Setup.ipnb

Following the structure suggested by:   
https://github.com/moemen95/Pytorch-Project-Template  
Adaptation: config EasyDict can be created from /utils/Dense_U_Net_lidar_helper  

Based on the torchvision densenet implementation:  
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py  
The original structure is fully respected, allowing to employ the pretrained weights from torchvision  

Waymo dataset paper:  
https://arxiv.org/abs/1912.04838

U-Net paper:  
https://arxiv.org/abs/1505.04597

DenseNet paper:  
https://arxiv.org/abs/1608.06993


