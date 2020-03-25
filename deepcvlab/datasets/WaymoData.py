import torch
import pickle
from os import listdir
from os.path import join, isdir
from torch.utils.data import Dataset, DataLoader
from ..utils.Dense_U_Net_lidar_helper import load_dict

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
        root = config.dir.data.root
        subdirs = listdir(root)

        # filenames incl path
        for subdir in subdirs:
            for datatype in config.dataset.datatypes:
                current_dir = join(root, subdir, mode, datatype)
                if isdir(current_dir):
                    self.files[datatype] = self.files[datatype] + [join(current_dir, file) for file in listdir(current_dir)]

        self._check_data_integrity()
        print('Your dataset consists of %d images' %len(self.files['images']))

    def __getitem__(self, idx):
        '''
        returns dataset items at idx
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        image = torch.load(self.files['images'][idx])    
        lidar = torch.load(self.files['lidar'][idx])
        labels= load_dict(self.files['labels'][idx])
        ht_map= torch.load(self.files['heat_maps'][idx])
    
        return image, lidar, labels, ht_map

    def __len__(self):
        return len(self.files['images'])

    def _check_data_integrity(self):
        '''
        check if names match as expected
        '''
        for i in range(self.__len__()):
            assert self.files['lidar'][i].endswith(self.files['images'][i][-11:]), self.files['lidar'][i] + ' ' + self.files['images'][i]
            assert self.files['labels'][i].endswith(self.files['images'][i][-11:]), self.files['labels'][i] + ' ' + self.files['images'][i]
            assert self.files['heat_maps'][i].endswith(self.files['images'][i][-11:]), self.files['heat_maps'][i] + ' ' + self.files['images'][i]

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
                pin_memory=config.loader.pin_memory)
            self.valid_loader = DataLoader(valid_set, 
                batch_size=config.loader.batch_size,
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory)
            
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
                pin_memory=config.loader.pin_memory)
            
            # iterations
            self.valid_iterations = (len(test_set) + config.loader.batch_size) // config.loader.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')