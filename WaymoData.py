import torch
import pickle
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from ..utils.Dense_U_Net_lidar_helper import load_dict

class WaymoDataset(Dataset):
    def __init__(self, mode, config):
        '''
        Assumes dirs to only contain the respective data !!!
        Assumes data to be sorted the same in all dirs !!!
        '''
        super().__init__()
        
        root = config.dir.data[mode].__dict__

        # filenames incl path
        self.files = {}
        for k in root:
            self.files[k] = [join(root[k], file) for file in listdir(root[k])] 
        
        self._check_data_integrity()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        image = torch.load(self.files['images'][idx])    
        lidar = torch.load(self.files['lidar'][idx])
        labels= load_dict(self.files['labels'][idx])
    
        return {'image': image, 'lidar': lidar, 'labels': labels}

    def __len__(self):
        return len(self.files['images'])

    def _check_data_integrity(self):
        for i in range(self.__len__):
            assert self.files['lidar'][i].endswith(self.files['images'][i]), 'something in lidar data dir that does not belong' 
            assert self.files['labels'][i].endswith(self.files['images'][i]), 'something in label data dir that does not belong'

class WaymoDataset_Loader:
    # TODO batch size
    # TODO num_workers
    def __init__(self, config):
        self.mode = config.loader.mode

        if self.mode == 'train':
            train_set = WaymoDataset('train', config)
            valid_set = WaymoDataset('val', config)

            self.train_loader = DataLoader(train_set, batch_size=config.loader.batch_size)
            self.valid_loader = DataLoader(valid_set, batch_size=config.loader.batch_size)
            self.train_iterations = (len(train_set) + config.loader.batch_size) // config.loader.batch_size
            self.valid_iterations = (len(valid_set) + config.loader.batch_size) // config.loader.batch_size

        elif self.mode == 'test':

            test_set = WaymoDataset('test', config)

            self.test_loader = DataLoader(test_set, batch_size=config.loader.batch_size)
            self.test_iterations = (len(test_set) + config.loader.batch_size) // config.loader.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')


        