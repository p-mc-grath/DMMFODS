from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class Waymo_Dataset(CustomDataset):

    CLASSES = ('pedestrian', 'car', 'cyclist')
    def load_annotations(self, ann_file): 
        pass
    
    def get_ann_info(self, idx):
        pass