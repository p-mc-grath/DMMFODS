import torch
from os.path import join
from torch.utils.data import DataLoader
from ..datasets.WaymoData import WaymoDataset as StandardWaymoDataset
from ..utils.Dense_U_Net_lidar_helper import load_dict


class WaymoDataset(StandardWaymoDataset):
    def __init__(self, mode, config):

        super().__init__(mode, config)
        self.img_size = (128, 192)

    def get_batch(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load data corresponding to idx
        file_path = join(self.root, self.files[idx])
        batch = torch.load(file_path)

        image_batch = batch[:, :3, :, :]
        lidar_batch = batch[:, 3, :, :].unsqueeze(1)
        ht_map_batch = batch[:, 4:, :, :]

        split_path = file_path.split('/')
        bbs_batch = load_dict(join('/', *split_path[:-1], 'labels', split_path[-1]))

        return image_batch, lidar_batch, ht_map_batch, self.format_bbs(bbs_batch, ht_map_batch)

    def format_maps(self, maps):
        # segmentation masks per bb!!!
        pass

    def format_bbs(self, bbs, ht_maps):
        '''
        Change from serialized version such that working with
        torchvision.models.detection.maskrcnn_resnet50_fpn

        From model above:
        During training, the model expects both the input tensors,
        as well as a targets (list of dictionary), containing:
        boxes (FloatTensor[N, 4]):
            the ground-truth boxes in [x1, y1, x2, y2] format,
            with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        labels (Int64Tensor[N]): the class label for each ground-truth box
        masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance; H,W -> whole image
        '''

        formatted_bbs = []
        # for each image
        for j, current_sample in enumerate(bbs.values()):

            # make space
            boxes = torch.zeros((len(current_sample), 4))
            labels = torch.zeros((len(current_sample)))
            masks = torch.zeros((len(current_sample), *self.img_size))

            # for each obj in ea image
            for i, bb in enumerate(current_sample.values()):
                bb = bb / 10
                boxes[i] = torch.tensor([bb['x'], bb['y'], bb['x'] + bb['width'], bb['y'] + bb['height']])

                obj_cls = bb['type']  # object_class == 1  2  4: VEHICLE, PEDESTRIAN, CYCLIST
                obj_idx = (obj_cls == 1) * 0 + (obj_cls == 2) * 1 + (obj_cls == 4) * 2

                labels[i] = obj_idx

                masks[i] = ht_maps[j, obj_idx]  # TODO actually need to remove all other objects?

            current_dict = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks
            }
            formatted_bbs.append(current_dict)

        return formatted_bbs


class WaymoDataset_Loader():

    def __init__(self, config):
        '''
        Creates Waymodatasets and calls default dataloader
        '''

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
