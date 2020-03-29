import torch
import pickle
from os import listdir
from os.path import join, isdir
from torch.utils.data import Dataset, DataLoader

class WaymoDataset(Dataset):
    def __init__(self, mode, config):
        '''
        Dirs are expected to ONLY CONTAIN the respective DATA !!!
        DATA is expected to be SORTED the SAME in all dirs !!!
        '''
        super().__init__()

        # allocation
        self.files = {}
        for datatype in config.dataset.datatypes:
            self.files[datatype] = []

        # data dirs
        self.root = config.dir.data.root
        waymo_buckets = listdir(self.root)

        # filenames incl path
        for waymo_bucket in waymo_buckets:
            tf_data_dirs = listdir(join(self.root, waymo_bucket))
            for tf_data_dir in tf_data_dirs:
                for datatype in config.dataset.datatypes:
                    current_dir_no_root = join(waymo_bucket, tf_data_dir, mode, datatype)                           # used to make storage req smaller
                    current_dir = join(self.root, current_dir_no_root)
                    if isdir(current_dir):
                        self.files[datatype] = self.files[datatype] + [join(current_dir_no_root, file) for file in listdir(current_dir)]

        self._check_data_integrity()
        print('Your %s dataset consists of %d images' %(mode, len(self.files['images'])))

    def __getitem__(self, idx):
        '''
        returns dataset items at idx
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        image = torch.load(join(self.root, self.files['images'][idx]))    
        lidar = torch.load(join(self.root, self.files['lidar'][idx]))
        ht_map= torch.load(join(self.root, self.files['heat_maps'][idx]))
    
        return image, lidar, ht_map

    def __len__(self):
        return len(self.files['images'])

    def _check_data_integrity(self):
        '''
        check if names match as expected
        '''
        for i in range(self.__len__()):
            assert self.files['lidar'][i].endswith(self.files['images'][i][-11:]), str(i) + ' ' + self.files['lidar'][i] + ' ' + self.files['images'][i]
            assert self.files['heat_maps'][i].endswith(self.files['images'][i][-11:]), str(i) + ' ' + self.files['heat_maps'][i] + ' ' + self.files['images'][i]

class WaymoDataset_Loader:

    def __init__(self, config):
        self.mode = config.loader.mode

        if self.mode == 'train':
            # dataset
            train_set = WaymoDataset('train', config)
            valid_set = WaymoDataset('val', config)

            # actual loader
            self.train_loader = DataLoader(train_set, 
                batch_size=config.loader.batch_size, 
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                drop_last=config.loader.drop_last)
            self.valid_loader = DataLoader(valid_set, 
                batch_size=config.loader.batch_size,
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                drop_last=config.loader.drop_last)
            
            # iterations
            self.train_iterations = (len(train_set) + config.loader.batch_size) // config.loader.batch_size
            self.valid_iterations = (len(valid_set) + config.loader.batch_size) // config.loader.batch_size

        elif self.mode == 'test':
            # dataset
            test_set = WaymoDataset('test', config)

            # loader also called VALID -> In Agent: valid function == test function; TODO find better solution 
            # !! dataset input is different
            self.valid_loader = DataLoader(test_set, 
                batch_size=config.loader.batch_size,
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                drop_last=config.loader.drop_last)
            
            # iterations
            self.valid_iterations = (len(test_set) + config.loader.batch_size) // config.loader.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')