import os
import numpy as np
import torch
import tensorflow as tf
import json
import warnings

from easydict import EasyDict as edict
from six.moves import cPickle as pickle
from os import listdir
from os.path import join, isfile
from pathlib import Path

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def create_config():
    '''
    create according to
    https://github.com/moemen95/Pytorch-Project-Template/tree/4d2f7bea9819fe2e5e25153c5cc87c8b5f35f4b8
    put into python for convenience with directories
    '''

    # root dir
    config = {
        'dir': {
            'root': '/content/drive/My Drive/Colab Notebooks/DeepCV_Packages/DeepCVLab/deepcvlab'
        }
    }

    # all script names
    config['scripts'] = {
        'model': 'Dense_U_Net_lidar.py',
        'utils': 'Dense_U_Net_lidar_helper.py',
        'agent': 'Dense_U_Net_lidar_Agent.py',
        'dataset': 'WaymoData.py',
        'setup': 'Setup.ipynb'
    }
    
    # model params
    config['model'] = {
        'growth_rate': 32,
        'block_config': (6, 12, 24, 16),
        'num_init_features': 64,
        'num_lidar_in_channels': 1,
        'concat_before_block_num': 2,
        'num_layers_before_blocks': 4,
        'bn_size': 4,
        'drop_rate': 0,
        'num_classes': 3,
        'memory_efficient': True
    }

    # loader params
    config['loader'] = {
        'mode': 'train',
        'batch_size': 32,
        'pin_memory': True,                                                 # TODO check what else has to be done
        'num_workers': 4,
        'async_loading': True                                               # should be same as pin_memory
    }

    # optimizer params; currently torch.optim.Adam default
    config['optimizer'] = {
        'type': 'Adam',
        'learning_rate': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-08,
        'amsgrad': False,
        'weight_decay': {
            'value': 0,
            'every_n_epochs': 30,
            'gamma': 0.1
        }
    }

    # waymo dataset info
    config['dataset'] = {
        'label': {
            '1': 'TYPE_VEHICLE',
            '2': 'TYPE_PEDESTRIAN',
            '4': 'TYPE_CYCLIST'
        },
        'images': {
            'original.size': (3, 1920, 1280),
            'size': (3, 192, 128)
        },
        'datatypes': ['images', 'lidar', 'labels', 'heat_maps']
    }

    # agent params
    config['agent'] = {
        'seed': 123,                                                        # fixed random seed ensures reproducibility
        'max_epoch': 100,
        'iou_threshold': 0.7,
        'checkpoint': {
            'epoch': 'epoch',
            'iteration': 'iteration',
            'best_val_acc': 'best_val_acc',
            'state_dict': 'state_dict',
            'optimizer': 'optimizer'
        },
        'best_checkpoint_name': 'best_checkpoint.pth.tar'
    }

    # create subdirs according to pytorch project template: https://github.com/moemen95/Pytorch-Project-Template/tree/4d2f7bea9819fe2e5e25153c5cc87c8b5f35f4b8
    for subdir in ['agents', 'graphs', 'utils', 'datasets', 'pretrained_weights', 'configs']:
        config['dir'][subdir] = join(config['dir']['root'], subdir)
    config['dir']['graphs'] = {'models': join(config['dir']['graphs'], 'models')}
    
    config['dir']['data'] = {'root': join(config['dir']['root'], 'data')}                       # because config.dir.data. train,test,val also exist
    # directories according to distribute_data_into_train_val_test function in this script
    for mode in ['train', 'val', 'test']:
        config['dir']['data'][mode] = {}
        for datatype in config['dataset']['datatypes']:
                config['dir']['data'][mode][datatype] = join(config['dir']['data']['root'], mode, datatype)

    config['dir']['summary'] = join(config['dir']['root'], 'training_summary') 

    return config

def get_config(root=''):
    '''
    Using json because human readable
    '''
    json_file = join(root, 'config', 'Dense_U_Net_lidar_config.json')
    
    if isfile(json_file):
        with open(json_file, 'r') as config_file:
            config = json.load(config_file)                                                           # values -> attrs
    else:
        config = create_config()

    return edict(config)

############################################################################
# Ground Truth functions
############################################################################
def _create_ground_truth_bb_pedestrian(ground_truth_box):
    '''
    Very rough, generic approximation of human silhouette in bounding box
    '''
    unlikely = 0.3
    uncertain = 0.5
    half_certain = 0.75
    
    height, width = ground_truth_box.shape
    height_fraction = height//5
    width_fraction = width//4

    ground_truth_box[0:height_fraction,:width_fraction] = unlikely
    ground_truth_box[0:height_fraction,width_fraction*3:] = unlikely
    ground_truth_box[height_fraction*3:,:width_fraction] = uncertain
    ground_truth_box[height_fraction*3:,width_fraction*3:] = uncertain
    ground_truth_box[height_fraction*3:,width_fraction:width_fraction*3] = half_certain

    return ground_truth_box

def _create_ground_truth_bb_cyclist(ground_truth_box):
    return ground_truth_box

def _create_ground_truth_bb_vehicle(ground_truth_box):
    return ground_truth_box

def _create_ground_truth_bb(object_class, width, height):     
    
    ground_truth_box = np.ones((height, width))

    # object_class number association corresponding to waymo label.proto
    if object_class == 2:                                                                       # TYPE_PEDESTRIAN
        ground_truth_box = _create_ground_truth_bb_pedestrian(ground_truth_box)
    elif object_class == 4:                                                                     # TYPE_CYCLIST
        ground_truth_box = _create_ground_truth_bb_cyclist(ground_truth_box)
    elif object_class == 1:                                                                     # TYPE_VEHICLE
        ground_truth_box = _create_ground_truth_bb_vehicle(ground_truth_box)
    else:
        raise TypeError('the ground truth label class does not exist')
    
    return ground_truth_box

def create_ground_truth_maps(ground_truth, width_img=1920, height_img=1280):
    '''
    Arguments:
        ground_truth: expected to be dict containing dicts
                dicts with following fields: type, x, y, width, height 
                x, y coords of upper left corner
        width_img:  of original!! image
        height_img: of original!! image
    return:
        to work with pytorch's batching these ground truth maps MUST NOT have
        a BATCH DIMENSION when saved
    '''
    maps = np.zeros((3, height_img, width_img))
    
    for elem in ground_truth.values():
        
        object_class = elem['type']
        if object_class == 1 or object_class == 2 or object_class == 4:                         # VEHICLE, PEDESTRIAN, CYCLIST
            width_bb = elem['width']
            height_bb = elem['height']
            x = elem['x']
            y = elem['y'] 

            obj_idx = (object_class==1)*0+(object_class==2)*1+(object_class==4)*2               # remapping obj identifying indeces

            maps[obj_idx, y:y+height_bb, x:x+width_bb] = _create_ground_truth_bb(object_class, width_bb, height_bb)
        
    return torch.Tensor(maps)     

def compute_IoU_whole_img_per_class(ground_truth_map, estimated_heat_map, threshold):
    '''
    Custom Intersection over Union function 
    Due output format it is not possible to compute IoU per bounding box
    :return:
        IoU_per_class: special case: union == 0 -> iou=0
    '''
    # make maps boolean
    est_bool = estimated_heat_map >= threshold
    gt_bool = ground_truth_map >= threshold                                             # TODO alternative: == 1??

    # boolean magic
    intersection = torch.sum(est_bool & gt_bool, axis=(1,2))
    union = torch.sum(est_bool | gt_bool, axis=(1,2))
    
    # in case union is 0 -> division of tensors returns nan -> set iou=0
    iou_per_class = intersection/union
    iou_per_class[torch.isnan(iou_per_class)] = 0

    return iou_per_class

def compute_IoU_whole_img_batch(ground_truth_map_batch, estimated_heat_map_batch, threshold=0.7):
    '''
    Arguments:
        threshold: int
        batches: of form: instance in batch, class, y, x
    '''
    # alocate space
    iou_per_instance_per_class = torch.zeros(ground_truth_map_batch.shape[0], ground_truth_map_batch.shape[1])

    # IoU per isntance
    for i, (gt_map, h_map) in enumerate(zip(ground_truth_map_batch, estimated_heat_map_batch)):
        iou_per_instance_per_class[i, :] = compute_IoU_whole_img_per_class(gt_map, h_map, threshold)

    return torch.mean(iou_per_instance_per_class, axis=0)

############################################################################
# converting waymo tfrecord files to pytorch and helpers
############################################################################

# pickle dictionary
def save_dict(dictionary, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

# unpickle dictionary
def load_dict(filename):
    with open(filename, 'rb') as handle:
        retrieved_dict = pickle.load(handle)
    return retrieved_dict 

def convert_label_dicts_to_heat_maps(dir):
    '''
    unused
    Overwrites dict files with corresponding tensors
    '''
    files = listdir(dir) 
    for file in files:
        dict = load_dict(join(dir, file))
        tensor = create_ground_truth_maps(dict)
        torch.save(tensor, join(dir, file))

def maxpool_tensor(img_tensor):
    '''
    save storage space by downsizing images
    '''
    max_pool = torch.nn.MaxPool2d(10, stride=10)                                               
    return max_pool(img_tensor)

def pool_lidar_tensor(lidar_tensor):
    '''
    THOUGHTS
    (1) if maxpool need to somehow invert lidar points
    -> we are more interested in the close points!!
    (2) Inversion on log scale because the closer the more important small diffs
    (3) pooling: receptive field > stride -> getting rid of zero values 
    
    SOLUTION
    linear bins but more bins for close range
    inversion such that
    0 -> 255
    25 -> 100
    75 -> 0
    '''
    # make sure all vecs are in [0,75] as specified by waymo
    lidar_max_range=75.0
    lidar_tensor[lidar_tensor==-1.0] = lidar_max_range+1

    # 155 bins for meters [0,25]; 25m is cutoff/ truncation value for 4 short-range lidar sensors 
    lidar_tensor[lidar_tensor<=25] = lidar_tensor[lidar_tensor<=25]*-4+255

    # 100 bins for meters (25,75]; 75m is cutoff/ truncation value for single mid-range lidar sensor
    lidar_tensor[lidar_tensor>25] = lidar_tensor[lidar_tensor>25]*-2+150
    
    # apply max pooling
    pool = torch.nn.MaxPool2d(15, stride=10)
    lidar_tensor = pool(lidar_tensor)

    # check if any "0" values passed
    if torch.any(lidar_tensor<0):
        warnings.warn(str(torch.sum(lidar_tensor<0)) + ' init values slipped during lidar pooling')
        lidar_tensor[lidar_tensor<0] = 0

    return lidar_tensor

def lidar_array_to_image_like_tensor(lidar_array, shape=(1,1280,1920)):
    '''
    read out lidar array into image with one channel = distance values
    '''
    range_tensor = torch.ones(shape)*-1.0
    for [x, y, d] in lidar_array:
        range_tensor[0,int(y),int(x)] = d.item()                                              # tensor does not accept np.float32
    
    return range_tensor

def extract_lidar_array_from_point_cloud(points, cp_points): 
    '''
    selected collection of lines from Waymo Open Dataset Tutorial.ipynb
    extracting lidar data from point cloud
    return:
        lidar array consisting of x, y, distance_value corresponding to
        the respective image
    '''
    
    points_all = np.concatenate(points, axis=0)                                                 # 3d points in vehicle frame.
    cp_points_all = np.concatenate(cp_points, axis=0)                                           # camera projection corresponding to each point.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)                             # The distance between lidar points and vehicle frame origin.
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
    mask = tf.equal(cp_points_all_tensor[..., 0], 1)                                            # FRONT -> 1; ORIGINAL: images = sorted(frame.images, key=lambda i:i.name); mask = tf.equal(cp_points_all_tensor[..., 0], images[idx_img].name)
    cp_points_all_tensor = tf.cast(tf.gather_nd(                                                # extract data corresponding to the SPECIFIC image
                cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
    lidar_array = tf.concat(                                                                    # projected_points_all_from_raw_data [0] = x, [1] = y from cp_points_all_tensor[..., 1:3]
            [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()               # projected_points_all_from_raw_data [2] = range from points_all_tensor

    return lidar_array

def distribute_data_into_train_val_test(split, config=None):
    '''
    reason: colab might disconnect during training; better have hard separation of data subsets!

    move image, lidar and label data from their respective subdirectory
    to train, val, test subdirectories preserving their respective subdirectories

    sampling is randomized; assuming same ORDER and NUMBER OF FILES of files in all subdirectories

    Arguments:
        config: edict
        split: list: [train_percentage, val_percentage, test_percentage]
    '''
    # get config
    if config is None:
        config = get_config()

    # same indices for all subdirs
    num_samples = len(listdir(os.path.join(config.dir.data.root, 'images')))
    indices = list(range(0, num_samples))
    np.random.shuffle(indices)
    
    # make splits to split_indices for indices list
    split = np.array(split)*num_samples
    split_indices = [0, int(split[0]), int(split[0]+split[1]), num_samples] 
        
    for data_type in config.dataset.datatypes: 
        old_path = os.path.join(config.dir.data.root, data_type)
        filenames = listdir(old_path)

        for set_idx, sub_dir in enumerate(['train', 'val', 'test']):
            new_path = os.path.join(config.dir.data.root, sub_dir, data_type)
            Path(new_path).mkdir(parents=True, exist_ok=True)

            for file_idx in indices[split_indices[set_idx]:split_indices[set_idx+1]]:
                filename = filenames[file_idx]
                os.rename(os.path.join(old_path, filename), os.path.join(new_path, filename))

        Path(old_path).rmdir()
        
def waymo_to_pytorch_offline(config=None, idx_dataset_batch=-1):
    '''
    Converts tfrecords from waymo open data set to
    (1) Images -> torch Tensor
    (2) Lidar Range Image -> torch Tensor
    (3) Labels -> dictionary of dictionaries
        dictionaries with following keys:
        'type':     1=TYPE_VEHICLE; 2=TYPE_PEDESTRIAN; 4=TYPE_CYCLIST
        'x':        upper left corner x                     !!labeling not as in original!!
        'y':        upper left corner y                     !!labeling not as in original!!
        'width':    width of corresponding bounding box     !!labeling not as in original!!
        'height':   height of corresponding bbounding box   !!labeling not as in original!!
    (4) heat_maps from labels -> torch Tensor; image like
    '''      
    # allows __iter__() for tf record dataset
    tf.compat.v1.enable_eager_execution()

    # get config
    if config is None:
        config = get_config()

    # dir names
    data_root = config.dir.data.root
    save_path_labels = os.path.join(data_root, 'labels')
    save_path_images = os.path.join(data_root, 'images')
    save_path_lidar = os.path.join(data_root, 'lidar')
    save_path_heat_maps = os.path.join(data_root, 'heat_maps')

    # create save dirs if not exist
    Path(save_path_labels).mkdir(exist_ok=True)
    Path(save_path_images).mkdir(exist_ok=True)
    Path(save_path_lidar).mkdir(exist_ok=True)
    Path(save_path_heat_maps).mkdir(exist_ok=True)
    
    # read all entries in data root directory
    raw_dir_entries = os.listdir(data_root)
    for idx_entry, entry in enumerate(raw_dir_entries):

        # skip all non tfrecord files
        if not entry.endswith('.tfrecord'):  
            continue

        dataset = tf.data.TFRecordDataset(os.path.join(data_root, entry), compression_type='')           # read tfrecord

        # for all datasets stored in tfrecord
        for idx_data, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))                                  # pass data from tfrecord to frame

            # for all images of current frame
            for idx_img, image in enumerate(frame.images):                                  # can probably reduce this + next if to: image = frame.images[0]
                
                # Only consider FRONT images
                if image.name != 1:                                                         # if not CameraName == FRONT: skip; best lidar, rgb match
                    continue
                
                ### retrieve, convert and save rgb data 
                np_img = np.moveaxis(tf.image.decode_jpeg(image.image).numpy(), -1, 0)      # frame -> np array with tensor like dims: channels,y,x      
                downsized_img_tensor = maxpool_tensor(torch.Tensor(np_img)) 
                img_filename = 'img_%d_%d_%d_%d' %(idx_dataset_batch, idx_entry, idx_data, idx_img)
                torch.save(downsized_img_tensor, os.path.join(save_path_images, img_filename))                     
                
                ### retrieve, convert and save lidar data
                (range_images, camera_projections,
                    range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
                                        frame)
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                                        frame,
                                        range_images,
                                        camera_projections,
                                        range_image_top_pose)
                lidar_array = extract_lidar_array_from_point_cloud(points, cp_points)       # lidar corresponds to image due to the mask in this function
                range_tensor = lidar_array_to_image_like_tensor(lidar_array)
                downsized_range_tensor = pool_lidar_tensor(range_tensor) 
                lidar_filename = 'lidar_' + img_filename
                torch.save(downsized_range_tensor, os.path.join(save_path_lidar, lidar_filename))

                ### retrieve, convert and save labels 
                label_dict = {}                                                             # dict of dicts
                labels_filename = 'labels_' + img_filename
                for camera_labels in frame.camera_labels:
                    # ignore labels corresponding to other images
                    if camera_labels.name != image.name:
                        continue
                    # for all labels
                    for idx_label, label in enumerate(camera_labels.labels):
                        label_dict[str(idx_label)] = {                                      
                            'type':label.type,                                              # weird waymo labeling
                            'x':int(label.box.center_x - 0.5*label.box.length),
                            'y':int(label.box.center_y - 0.5*label.box.width),
                            'height':int(label.box.width),
                            'width':int(label.box.length)
                        }
                save_dict(label_dict, os.path.join(save_path_labels, labels_filename))

                ### create ground truth maps from labels and save
                heat_map_filename = 'heat_map_' + img_filename
                heat_map = create_ground_truth_maps(label_dict)
                downsized_heat_map = maxpool_tensor(heat_map)
                torch.save(downsized_heat_map, os.path.join(save_path_heat_maps, heat_map_filename))
                
                want_small_dataset_for_testing = False
                if idx_data == 9 and want_small_dataset_for_testing:
                    return 1 
