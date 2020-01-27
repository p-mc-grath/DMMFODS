import os
import numpy as np
import torch
import tensorflow as tf

from six.moves import cPickle as pickle
from os.path import join
from google.colab import drive
from pathlib import Path

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

############################################################################
# Ground Truth functions
############################################################################
def _create_ground_truth_bb_pedestrian(ground_truth_box):
    
    unlikely = 0.3
    uncertain = 0.5
    half_certain = 0.75
    
    height, width = ground_truth_box.shape[2:]
    height_fraction = height//5
    width_fraction = width//4

    ground_truth_box[0, 0, 0:height_fraction,:width_fraction] = unlikely
    ground_truth_box[0, 0, 0:height_fraction,width_fraction*3:] = unlikely
    ground_truth_box[0, 0, height_fraction*3:,:width_fraction] = uncertain
    ground_truth_box[0, 0, height_fraction*3:,width_fraction*3:] = uncertain
    ground_truth_box[0, 0, height_fraction*3:,width_fraction:width_fraction*3] = half_certain

    return ground_truth_box

def _create_ground_truth_bb_cyclist(ground_truth_box):
    return ground_truth_box

def _create_ground_truth_bb_vehicle(ground_truth_box):
    return ground_truth_box

def _create_ground_truth_bb(object_class, width, height):     
    
    ground_truth_box = np.ones((1, 1, height, width))

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

def create_ground_truth_maps(ground_truth, width_img, height_img):
    '''
    ground_truth: expected to be iterable containing dicts
                dicts with following fields: type, x, y, width, height 
                x, y coords of upper left corner
    width_img:  of original!! image
    height_img: of original!! image
    '''
    maps = np.zeros((1, 3, height_img, width_img))
    
    for elem in ground_truth.values():
        
        object_class = elem['type']
        if object_class == 1 or object_class == 2 or object_class == 4:                         # VEHICLE, PEDESTRIAN, CYCLIST
            width_bb = elem['width']
            height_bb = elem['height']
            x = elem['x']
            y = elem['y'] 

            obj_idx = (object_class==1)*0+(object_class==2)*1+(object_class==4)*2

            maps[0, obj_idx, y:y+height_bb, x:x+width_bb] = _create_ground_truth_bb(object_class, width_bb, height_bb)
        
    return torch.Tensor(maps)     

def compute_IoU_whole_img(threshold, ground_truth_map, estimated_heat_map):
    '''
    Custom Intersection over Union function 
    Due output format it is not possible to compute IoU per bounding box
    '''
    # make maps boolean
    est_bool = estimated_heat_map.numpy() >= threshold
    gt_bool = ground_truth_map.numpy() >= threshold                                             # TODO alternative: == 1??

    # numpy magic
    intersection = np.sum(np.logical_and(est_bool, gt_bool))
    union = np.sum(np.logical_or(est_bool, gt_bool))
    
    return intersection/union

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

def pool_range_tensor(range_tensor):
    '''
    unused
    downsizing WxH while preserving 99.9% of the values
    i.e. making feature maps more dense
        WxH         lidar:zeros   lidar points absolute     iteration
    --> 1920x1280:  1:150         16159                     ORIGINAL 
    --> 640x960:    1:38          16128                     0
    --> 320x480:    1:10          16004                     1
    '''
    max_pool = torch.nn.MaxPool2d(2, stride=2)                                                  # consider (3, stride=2, padding=1) for more dense feature maps while downsampling WxH
    for _ in range(2):
        range_tensor = max_pool(range_tensor)

    return range_tensor

def lidar_array_to_image_like_tensor(lidar_array, shape=(1,1,1280,1920)):
    
    range_tensor = torch.zeros(shape)
    for [x, y, d] in lidar_array:
        range_tensor[1,1,int(y),int(x)] = d.item()                                              # tensor does not accept np.float32
    
    return range_tensor

def extract_lidar_array_from_point_cloud(points, cp_points): 
    '''
    selected collection of lines from Waymo Open Dataset Tutorial.ipynb
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

def distribute_data_into_train_val_test(data_root, split):
    '''
    move image, lidar and label data from their respective subdirectory
    to train, val, test subdirectories preserving their respective subdirectories
    
    sampling is randomized; assuming same order of files in all subdirectories

    Arguments:
        data_root: dir path above subdirectories of diff datatypes
        split: list: [train_percentage, val_percentage, test_percentage]
    '''

    # same indices for all subdirs
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split = np.array(split)*num_samples
    split = np.array([0, split[0], split[0]+split[1], num_samples])
        
    for data_type in ['labels','images','lidar']:
        filenames = listdir(os.path.join(data_root, data_type))
        num_samples = len(filenames)

        for i, sub_dir in enumerate(['train', 'val', 'test'])):
            save_path = os.path.join(data_root, sub_dir, data_type)
            Path(save_path).mkdir(exist_ok=True)

            for filename in filenames[indices[split[i:i+1]]]:
                os.rename(os.path.join(data_root, filename), os.path.join(save_path, filename))

def waymo_to_pytorch_offline(idx_dataset_batch):
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
    '''    

    data_root = os.path.join('content', 'mnt', 'My Drive', 'Colab Notebooks', 'DeepCV_Packages')
    save_path = os.path.join(data_root, 'data')                                                
    save_path_labels = os.path.join(save_path, 'labels')
    save_path_images = os.path.join(save_path, 'images')
    save_path_lidar = os.path.join(save_path, 'lidar')

    # create save dirs if not exist
    Path(save_path_labels).mkdir(exist_ok=True)
    Path(save_path_images).mkdir(exist_ok=True)
    Path(save_path_lidar).mkdir(exist_ok=True)

    # read all entries in data root directory
    raw_dir_entries = os.listdir(data_root)
    for idx_entry, entry in enumerate(raw_dir_entries):

        # for all tfrecord files
        if entry.endswith('.tfrecord'):                                                     
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
                    img_tensor = torch.Tensor(np_img).unsqueeze(0)                              # np array -> torch Tensor: add batch size as first dim      
                    img_filename = 'img_%d_%d_%d_%d' %(idx_dataset_batch, idx_entry, idx_data, idx_img) 
                    torch.save(img_tensor, os.path.join(save_path_images, img_filename))                     
                    
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
                    # range_tensor = pool_range_tensor(range_tensor)                            # while preserving most data points WxH --> W/4xH/4
                    lidar_filename = 'lidar_' + img_filename
                    torch.save(range_tensor, os.path.join(save_path_lidar, lidar_filename))

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