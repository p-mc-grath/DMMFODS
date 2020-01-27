import torch
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from Dense_U_Net_lidar_helper import load_dict

class WaymoDataset(Dataset):
    def __init__(self, rgb_root, lidar_root, label_root):

        super().__init__()

        # filenames incl path
        self.rgb_files = [join(rgb_root, file) for file in listdir(rgb_root)]
        self.lidar_files = [join(lidar_root, file) for file in listdir(lidar_root)]
        self.label_files = [join(label_root, file) for file in listdir(label_root)]  
        
        self._check_data_integrity()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        image = torch.load(self.rgb_files[idx])    
        lidar = torch.load(self.lidar_files[idx])
        labels= load_dict(self.label_files[idx])
    
        return {'image': image, 'lidar': lidar, 'labels': labels}

    def __len__(self):
        return len(self.rgb_files)

    def _check_data_integrity(self):
        for i in range(len(self.rgb_files)):
            assert self.lidar_files[i].endswith(self.rgb_files[i]), 'something in lidar data dir that does not belong' 
            assert self.label_files[i].endswith(self.rgb_files[i]), 'something in label data dir that does not belong'

class WaymoDataset_Loader:
    # TODO batch size
    # TODO num_workers
    def __init__(self, mode, data_root, bn):
        self.mode = mode

        if self.mode == 'train':
            train_set = WaymoDataset(join(data_root, 'train', 'images'), 
                                    join(data_root, 'train', 'lidar'), 
                                    join(data_root, 'train', 'labels'))
            valid_set = WaymoDataset(join(data_root, 'val', 'images'), 
                                    join(data_root, 'val', 'lidar'), 
                                    join(data_root, 'val', 'labels'))

            self.train_loader = DataLoader(train_set, batch_size=bn)
            self.valid_loader = DataLoader(valid_set, batch_size=bn)
            self.train_iterations = (len(train_set) + bn) // bn
            self.valid_iterations = (len(valid_set) + bn) // bn

        elif self.mode == 'test':

            test_set = WaymoDataset(join(data_root, 'test', 'images'), 
                                    join(data_root, 'test', 'lidar'), 
                                    join(data_root, 'test', 'labels'))

            self.test_loader = DataLoader(test_set, batch_size=bn)
            self.test_iterations = (len(test_set) + bn) // bn

        else:
            raise Exception('Please choose a proper mode for data loading')