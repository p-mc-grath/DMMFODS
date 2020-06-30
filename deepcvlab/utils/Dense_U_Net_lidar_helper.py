import os
import numpy as np
import torch
import tensorflow as tf
import json
import warnings

from datetime import datetime
from easydict import EasyDict as edict
from six.moves import cPickle as pickle
from os import listdir
from os.path import join, isfile, isdir
from pathlib import Path

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def load_json_file(filepath):
    '''
    simply loads json file

    Arguments:
        filepath: full path path incl. filename and extension
    '''

    if isfile(filepath):
        with open(filepath, 'r') as jf:
                json_file = json.load(jf)
    else:
        raise FileNotFoundError

    return json_file

def save_json_file(filepath, save_file, indent=None):
    '''
    simply saves json file

    Arguments:
        filepath: full path path incl. filename and extension
        save_file: file to save
        indent: allows pretty json file for human readability | None most dense representation
    '''

    with open(filepath, 'w') as jf:
        json.dump(save_file, jf, indent=indent)

    print('Successfully saved ' + filepath)
    return 1

############################################################################
# Config functions
############################################################################

def load_config(loading_dir, file_name):
    '''
    tries to load config from json file
    '''

    json_file = join(loading_dir, file_name)
     
    if isfile(json_file):
        # load
        config = load_json_file(json_file)

        return config
    else:
        return None

def save_config(config, file_name='config.json'):
    '''
    saves to json formatted with indent = 4 | human readable
    '''
    
    # save to json file | pretty with indent
    Path(config.dir.configs).mkdir(exist_ok=True)
    save_json_file(os.path.join(config.dir.configs, file_name), config, indent=4)

def create_config(host_dir):
    '''
    create according to
    https://github.com/moemen95/Pytorch-Project-Template/tree/4d2f7bea9819fe2e5e25153c5cc87c8b5f35f4b8
    put into python for convenience with directories
    '''

    if not host_dir:
        host_dir = '/content/drive/My Drive/Colab Notebooks/DeepCV_Packages'

    # overall root dir
    config = {
        'dir': {
            'hosting': host_dir
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
        'stream_1_in_channels': 3,                                          # rgb if rgb and lidar | rgb if rgb only | lidar if lidar only
        'stream_2_in_channels': 1,                                          # lidar if rgb and lidar | 0 if rgb only | 0 if lidar only
        'concat_before_block_num': 2,
        'num_layers_before_blocks': 4,
        'bn_size': 4,
        'drop_rate': 0,
        'num_classes': 3,
        'memory_efficient': False
    }

    config['loss'] = { 
        'alpha': 1,                                                         # default value focal loss
        'gamma': 2,                                                         # default value focal loss
        'logits': True,
        'reduce': False,
        'skip_v_every_n_its': None,
        'skip_p_every_n_its': None,
        'skip_b_every_n_its': None
    }

    # loader params
    config['loader'] = {
        'mode': 'train',
        'batch_size': None,
        'pin_memory': True,                                                 
        'num_workers': 4,
        'async_loading': True,                                              
        'drop_last': False                                              # needs to be False if batch_size None
    }

    # optimizer params; currently torch.optim.Adam default
    config['optimizer'] = {
        'type': 'Adam',
        'learning_rate': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-08,
        'amsgrad': False,
        'weight_decay': 0,
        'lr_scheduler': {
            'want': False,
            'every_n_epochs': 30,
            'gamma': 0.1
        }
    }

    # waymo dataset info
    config['dataset'] = {
        'batch_size': 32,                                                   # batch size of serialized files | >=1
        'label': {
            '1': 'TYPE_VEHICLE',
            '2': 'TYPE_PEDESTRIAN',
            '4': 'TYPE_CYCLIST'
        },
        'images': {
            'original.size': (3, 1920, 1280),
            'size': (3, 192, 128)
        },
        'datatypes': ['images', 'lidar', 'labels', 'heat_maps'],
        'file_list_name': 'file_list.json'
    }

    # agent params
    config['agent'] = {
        'seed': 123,                                                        # fixed random seed ensures reproducibility
        'max_epoch': 100,
        'iou_threshold': 0.7,
        'checkpoint': {                                                     # naming in checkpoint dict
            'epoch': 'epoch',
            'train_iteration': 'train_iteration',
            'val_iteration': 'val_iteration',
            'best_val_iou': 'best_val_iou',
            'state_dict': 'state_dict',
            'optimizer': 'optimizer'
        },
        'best_checkpoint_name': 'best_checkpoint.pth.tar'
    }

    # create subdirs according to pytorch project template: https://github.com/moemen95/Pytorch-Project-Template/tree/4d2f7bea9819fe2e5e25153c5cc87c8b5f35f4b8
    config['dir']['root'] = join(config['dir']['hosting'], 'DeepCVLab', 'deepcvlab')
    for subdir in ['agents', 'graphs', 'utils', 'datasets', 'configs', 'experiments']:
        config['dir'][subdir] = join(config['dir']['root'], subdir)
    config['dir']['graphs'] = {'models': join(config['dir']['graphs'], 'models')}
    
    config['dir']['data'] = {
        'root': join(config['dir']['hosting'], 'data'),
        'file_lists': join(config['dir']['root'], 'data')
        }                      

    # Current run: tensorBoard summary writers dir and checkpoint dir
    current_run = datetime.now().strftime('%Y-%m-%d-%H-%M')
    config['dir']['current_run'] = {
        'summary':   join(config['dir']['experiments'], current_run, 'summary'),
        'checkpoints': join(config['dir']['experiments'], current_run, 'checkpoints')
    }

    return config

def get_config(host_dir='', file_name='config.json'):
    '''
    load from json file or create config
    '''

    config = load_config(join(host_dir, 'DeepCVLab', 'deepcvlab', 'configs'), file_name)
    
    if config is None:
        config = create_config(host_dir)

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

############################################################################
# Metric functions
############################################################################

def compute_IoU_whole_img_per_class(ground_truth_map, estimated_heat_map, threshold):
    '''
    Custom Intersection over Union function 
    Due output format it is not possible to compute IoU per bounding box

    Arguments:
        gt_map: 
        estimated_heat_map: 
            -> struct of these must be exactly the same: class, y, x
        threshold: must be in [0,1] values above the threshold are set to 1, below to zero

    return:
        IoU_per_class: special case: union == 0 -> iou=nan!!!
    '''

    # make maps boolean
    est_bool = estimated_heat_map >= threshold
    gt_bool = ground_truth_map >= threshold                                             

    # boolean magic: intersection of a bb is equivalent with the and operation; union with the or operation
    # note: IoU is only computed over detected BBs -> these values are True in the boolean maps 
    #       this is why and and or work
    # casting vec to float; might be long -> division by zero results in a runtime error vs. float: 0/0 = nan
    intersection = torch.sum(est_bool & gt_bool, axis=(1,2)).float()
    union = torch.sum(est_bool | gt_bool, axis=(1,2)).float()
    
    # if there is no BB in gt e.g. most images do not contain pedestrians or cyclists
    # -> union might be 0, leading to the following:
    # 0/0 = nan; 1/0 = inf -> not possible as i <= u
    # As these values would not occur with normal IoU computation, they are left to be dealt with later
    iou_per_class = intersection/union

    return iou_per_class

def compute_IoU_whole_img_batch(ground_truth_map_batch, estimated_heat_map_batch, threshold=0.7):
    '''
    Aggregates whole image per sample and class

    Arguments:
        threshold: value [0,1] representing cutoff; values are set to if above = 1 ; below = 0
        batches: of form: instance in batch, class, y, x
    
    return:
        whole image IoU per sample and class
        when IoU = 0/0 returns nan
    '''

    # alocate space of size: samples x classes
    iou_per_instance_per_class = torch.zeros(ground_truth_map_batch.shape[0], ground_truth_map_batch.shape[1])

    # IoU per instance:  compute whole image IoU for each instance separately
    for i, (gt_map, h_map) in enumerate(zip(ground_truth_map_batch, estimated_heat_map_batch)):
        iou_per_instance_per_class[i, :] = compute_IoU_whole_img_per_class(gt_map, h_map, threshold)

    # average IoU over all samples -> return class-wise IoU
    # NaN values do not carry any info so they are ignored
    return iou_per_instance_per_class

def compute_accuracy(ground_truth, prediction, threshold=0.7):
    '''
    applies threshold to both ground truth and prediction
    computes accuracy score accordingly: (TP+TN)/(All)
    
    Arguments:
        ground_truth: ground truth map of one sample/ batch of maps: classes, y, x
        prediction: heatmap of one sample/ batch of maps: classes, y, x
        threshold: used to threshold both prediction and gt
    
    
    return:
        class-wise accuracy
    '''

    # allowing single sample as well as batches to be passed to this function
    if len(ground_truth.shape) == 3:
        axes = (1,2)
        num_classes = ground_truth.shape[0]
    elif len(ground_truth.shape) == 4:
        axes = (0,2,3)
        num_classes = ground_truth.shape[1]
    else:
        raise ValueError('Number of dimensions must be either 3 or 4, you gave ' + str(len(ground_truth.shape)))
    
    # threshold the maps into binary representations
    bin_pred = prediction >= threshold
    bin_gt = ground_truth >= threshold   

    # class-wise accuracy computation
    # number of values that are equal in both tensors divided by the number of elements per class
    acc = torch.sum(bin_pred == bin_gt, axis=axes)/(ground_truth.numel()/ num_classes)
    
    return acc

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
        label_dict = load_dict(join(dir, file))
        tensor = create_ground_truth_maps(label_dict)
        torch.save(tensor, join(dir, file))

def avgpool_tensor(img_tensor):
    '''
    save storage space by downsizing images
    '''

    avg_pool = torch.nn.AvgPool2d(10, stride=10)                                               
    return avg_pool(img_tensor)
    
def maxpool_tensor(img_tensor):
    '''
    save storage space by downsizing images
    '''

    max_pool = torch.nn.MaxPool2d(10, stride=10)                                               
    return max_pool(img_tensor)

def pool_lidar_tensor(lidar_tensor):
    '''
    Waymo data set cut-off ranges:
    (1) 25m is the truncation value for 4 short-range lidar sensors 
    (2) 75m is the truncation value for the single mid-range lidar sensor

    THOUGHTS
    (1) if maxpool need to somehow invert lidar points
    -> we are more interested in the close points!!
    (2) Inversion on log scale because the closer the more important small diffs
    (3) pooling: receptive field > stride -> getting rid of zero values 
    
    SOLUTION
    linear bins but more bins for close range
    inversion such that
    empty values -> 255 | TODO change?
    0 -> 255
    25 -> 100
    75 -> 0
    '''

    # make sure all vecs are in [0,75] as specified by waymo
    lidar_max_range=75.0
    lidar_tensor[lidar_tensor>lidar_max_range] = lidar_max_range                # clipping not consequent; some vals 76.x
    lidar_tensor[lidar_tensor==-1.0] = lidar_max_range+1                        # results in negative values after inversion

    # 155 bins for meters [0,25]; converted to [100,255]
    lidar_tensor[lidar_tensor<=25] = lidar_tensor[lidar_tensor<=25]*-6.2+255

    # 100 bins for meters (25,75]; converted [0,100)
    lidar_tensor[(lidar_tensor>25)&(lidar_tensor<=lidar_max_range+1)] = lidar_tensor[
        (lidar_tensor>25)&(lidar_tensor<=lidar_max_range+1)]*-2+150
    
    # apply max pooling
    max_pool = torch.nn.MaxPool2d((20,10), stride=(10,10))
    lidar_tensor = max_pool(lidar_tensor)
    
    # pad such that shape = (1,128,192)| necessary due to maxpool filter size (20,10)
    lidar_tensor = torch.nn.functional.pad(lidar_tensor.unsqueeze(0), 
                                            pad=(0,0,0,1), mode='replicate').squeeze(0)

    # if any init values passed, set to 0
    lidar_tensor[lidar_tensor<0] = 0


    return lidar_tensor

def lidar_array_to_image_like_tensor(lidar_array, shape=(1,1280,1920), kernel_size=5):
    '''
    read out lidar array into image with one channel = distance values
    allow splatting values to surrounding pixels
    '''

    shift = (kernel_size-1)//2

    range_tensor = torch.ones(shape)*-1.0
    for [x, y, d] in lidar_array:

        min_y = int(y-shift)
        if min_y<0: min_y=0
        max_y = int(y+shift+1)
        if max_y>shape[1]-1: max_y=shape[1]-1
        min_x = int(x-shift)
        if min_x<0: min_x=0
        max_x = int(x+shift+1)
        if max_x>shape[2]-1: max_x=shape[2]-1

        range_tensor[0,min_y:max_y,min_x:max_x] = d.item()                                              # tensor does not accept np.float32
    
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

            
def waymo_to_pytorch_offline(data_root='', idx_dataset_batch=-1):
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

    colab: there are a lot of unnessecary subdirs because of googlefilestream limitations
    data is not stored in batches because the data comes in sequences of 20s. 
    '''     

    # allows __iter__() for tf record dataset
    tf.compat.v1.enable_eager_execution()

    # get config
    if not data_root:
        config = get_config()
        data_root = config.dir.data.root
    
    # read all entries in data root directory
    tf_dirs = [tfd for tfd in os.listdir(data_root) if tfd.startswith('tf_')]
    for idx_tf_dir, tf_dir in enumerate(tf_dirs):

        # skip all non tfrecord files
        tf_data_path = os.path.join(data_root, tf_dir)
        for file in os.listdir(tf_data_path):
            
            if not file.endswith('.tfrecord'):  
                continue
        
            # dir names
            save_path_labels = os.path.join(tf_data_path, 'labels')
            save_path_images = os.path.join(tf_data_path, 'images')
            save_path_lidar = os.path.join(tf_data_path, 'lidar')
            save_path_heat_maps = os.path.join(tf_data_path, 'heat_maps')

            # create save dirs if not exist
            Path(save_path_labels).mkdir(exist_ok=True)
            Path(save_path_images).mkdir(exist_ok=True)
            Path(save_path_lidar).mkdir(exist_ok=True)
            Path(save_path_heat_maps).mkdir(exist_ok=True)

            dataset = tf.data.TFRecordDataset(os.path.join(data_root, tf_dir, file), compression_type='')           # read tfrecord

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
                    downsized_img_tensor = avgpool_tensor(torch.Tensor(np_img)) 
                    img_filename = 'img_%d_%d_%d_%d' %(idx_dataset_batch, idx_tf_dir, idx_data, idx_img)
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
        print(idx_data+1, ' IMAGES PROCESSED')

def save_data_in_batch(config, buckets, mode):
    '''
    Crawls the given directories
    Loads separate samples in randomized order
    Creates batches of the given size
    Saves the batches

    Arguments:
        config: as specified in utils
        buckets: directory names in config.dir.data.root you want to include 
                    as list of strings or one string
        mode: one of train, val, test
    '''

    final_dirs = ['train','val','test']

    if not mode in final_dirs:
        raise ValueError('mode must be one of train, val, test. You gave ' + mode)
    
    if type(buckets) is str:
        buckets = [buckets]

    # load all file paths
    files = []
    for bucket in buckets:
        tf_data_dirs = [tfd for tfd in listdir(join(config.dir.data.root, bucket)) if tfd.startswith('tf_')]
        for tf_data_dir in tf_data_dirs:
            current_dir_no_root = join(bucket, tf_data_dir, 'images')                           
            current_dir = join(config.dir.data.root, current_dir_no_root)
            if isdir(current_dir):
                files += [join(current_dir_no_root, f) for f in listdir(current_dir)]
    
    # create random sampling list and alocate space
    indeces = list(range(len(files)))
    np.random.shuffle(indeces)
    vec = torch.empty((config.dataset.batch_size,7,128,192))

    # create subdirs
    Path(join(config.dir.data.root, mode)).mkdir(exist_ok=True)
    for i in range(len(indeces)//config.dataset.batch_size):
        if i%99 == 0:
            save_dir = join(config.dir.data.root, mode, 'subset' + str(i//99))
            Path(save_dir).mkdir(exist_ok = True)

        # load minibatch into vec and save
        for j in range(config.dataset.batch_size):
            idx = indeces[i*config.dataset.batch_size + j]

            # create lidar and heatmap paths
            path, image = files[idx].split('images/img_')
            lidar_file_path = join(config.dir.data.root, path, 'lidar/lidar_img_' + image)
            heat_map_file_path = join(config.dir.data.root, path, 'heat_maps/heat_map_img_' + image)

            vec[j,:3,:,:] = torch.load(join(config.dir.data.root, files[idx]))
            vec[j,3,:,:] = torch.load(lidar_file_path)
            vec[j,4:,:,:] = torch.load(heat_map_file_path)
        save_path = join(save_dir, str(i%99))
        torch.save(vec, save_path)
    print(i, 'batches serialized')