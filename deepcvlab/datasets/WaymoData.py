import torch
import pickle
from os import listdir
from os.path import join, isdir, isfile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ..utils.Dense_U_Net_lidar_helper import load_json_file, save_json_file

class WaymoDataset(Dataset):
    def __init__(self, mode, config):
        '''
        Allows for data to be stored as one file per sample and datatype 
        or as batches of 32 samples each containing image, lidar and heatmap data all in one file
        --> this speeds up colab workflow significantly

        Arguments:
            mode: One of 'train', 'val', 'test'
            config: as defined in utils

        Dirs are expected to ONLY CONTAIN the respective DATA !!!
        DATA is expected to be SORTED the SAME in all dirs !!!

        colab: there are a lot of unnessecary subdirs because of googlefilestream limitations
        '''
        super().__init__()
        self.root = config.dir.data.root
        self.data_is_batched = config.dataset.batch_size > 1

        # to load or to save crawled data
        json_file_path = join(config.dir.data.file_lists, mode + '_' + config.dataset.file_list_name)
       
        # load from json file if possible
        if isfile(json_file_path):
            self.files = load_json_file(json_file_path)

        # crawl directories
        else:
            
            # batches
            if self.data_is_batched:
                if config.loader.batch_size is not None:
                    raise ValueError('config.loader.batch_size needs to be None if loading batched dataset')
                
                self.files = []
                subdirs = listdir(join(self.root, mode))
                for subdir in subdirs:
                    batched_data = listdir(join(self.root, mode, subdir))
                    self.files += [join(mode, subdir, batch) for batch in batched_data]


            # single files
            elif not self.data_is_batched:

                # allocation
                self.files = {}
                for datatype in config.dataset.datatypes:
                    self.files[datatype] = []

                # filenames incl path
                waymo_buckets = listdir(self.root)
                waymo_buckets = [wb for wb in waymo_buckets if wb.startswith('training_0')]
                waymo_buckets.sort()
                for waymo_bucket in waymo_buckets:
                    tf_data_dirs = listdir(join(self.root, waymo_bucket))
                    for tf_data_dir in tf_data_dirs:
                        for datatype in config.dataset.datatypes:
                            current_dir_no_root = join(waymo_bucket, tf_data_dir, mode, datatype)                           # used to make storage req smaller
                            current_dir = join(self.root, current_dir_no_root)
                            if isdir(current_dir):
                                self.files[datatype] = self.files[datatype] + [join(current_dir_no_root, f) for f in listdir(current_dir)]
                print('Your %s dataset consists of %d images' %(mode, len(self.files['images'])))

                # make sure all names match
                self._check_data_integrity()

            else: 
                raise ValueError('make sure that config.dataset.batch_size >= 1')

            # save for next time
            Path(config.dir.data.file_lists).mkdir(exist_ok=True)
            save_json_file(json_file_path, self.files)

    def get_batch(self, idx):
        '''
        returns dataset batch at idx
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        batch = torch.load(join(self.root, self.files[idx]))

        image_batch = batch[:,:3,:,:]   
        lidar_batch = batch[:,3,:,:].unsqueeze(1)                           
        ht_map_batch = batch[:,4:,:,:]
    
        return image_batch, lidar_batch, ht_map_batch

    def get_single_sample(self, idx):
        '''
        returns dataset item at idx
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        image = torch.load(join(self.root, self.files['images'][idx]))    
        lidar = torch.load(join(self.root, self.files['lidar'][idx]))
        ht_map= torch.load(join(self.root, self.files['heat_maps'][idx]))
    
        return image, lidar, ht_map

    def __getitem__(self, idx):
        '''
        returns dataset items at idx
        '''

        if self.data_is_batched:
            return self.get_batch(idx)
        
        elif not self.data_is_batched:
            return self.get_single_sample(idx)

        else:
            raise ValueError('make sure that config.dataset.batch_size >= 1')

    def __len__(self):
        '''
        returns number of batches/ samples
        '''
        if self.data_is_batched:
            return len(self.files)

        elif not self.data_is_batched:
            return len(self.files['images'])

        else:
            raise ValueError('make sure that config.dataset.batch_size >= 1')

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
            if train_set.data_is_batched:
                self.train_iterations = len(train_set) 
                self.valid_iterations = len(valid_set) 
            else:
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
            if test_set.data_is_batched:
                self.valid_iterations = len(test_set) 
            else:
                self.valid_iterations = (len(test_set) + config.loader.batch_size) // config.loader.batch_size

        else:
            raise ValueError('Please choose a one of the following modes: train, val, test')