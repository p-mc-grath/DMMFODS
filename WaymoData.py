from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoDataset:
    def __init__(self, mode, data_root):
        self.mode = mode
        self.data_root = data_root
        
        dataset = tf.data.TFRecordDataset(data_root, compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            frame.camera_labels
            for camera_labels in frame.camera_labels:
                # Iterate over the individual labels.
                for label in camera_labels.labels:
                # Draw the object bounding box.
                    label.box.center_x 
                    label.box.center_y 
                    label.box.width
                    label.box.length,
                    label.box.width,

        ###
            for index, image in enumerate(frame.images):
                tf.image.decode_jpeg(image.image)

            self.imgs
            self.labels

    def __getitem__(self, idx):
        dataset.as_numpy_iterator()
        dataset.range(index-1, index)
        sample = dataset.from_tensor_slices[idx]
        return sample
    
    def __len__(self):
        return len(self.imgs)

class WaymoDataset_Loader:
    def __init__(self, config):
        '''
        Args:
        config.data_root
        config.mode
        config.batch_size
        config.data_loader_workers
        config.pin_memory
        '''
        self.config = config
        assert self.config.mode in ['train', 'test']

        if self.config.mode == 'train':
            train_set = WaymoDataset('train', self.config.data_root)
            valid_set = WaymoDataset('val', self.config.data_root)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = WaymoDataset('test', self.config.data_root)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')