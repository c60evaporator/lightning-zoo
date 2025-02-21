from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.voc import DATASET_YEAR_DICT
import shutil
import os

from torch_extend.dataset import VOCDetectionTV

from .base_detection import DetectionDataModule

def download_voc_dataset(root):
    """Download Pascal VOC 2012 dataset"""
    # Download Pascal VOC 2012 dataset
    url = DATASET_YEAR_DICT["2012"]["url"]
    filename = DATASET_YEAR_DICT["2012"]["filename"]
    md5 = DATASET_YEAR_DICT["2012"]["md5"]
    if not os.path.isfile(f'{root}/{filename}'):
        download_and_extract_archive(url, root, filename=filename, md5=md5)
        # Move the extracted files to the root directory
        base_dir = DATASET_YEAR_DICT["2012"]["base_dir"]
        voc_root = os.path.join(root, base_dir)
        if os.path.isdir(voc_root):
            shutil.move(f'{voc_root}/Annotations', root)
            shutil.move(f'{voc_root}/ImageSets', root)
            shutil.move(f'{voc_root}/JPEGImages', root)
            shutil.move(f'{voc_root}/SegmentationClass', root)
            shutil.move(f'{voc_root}/SegmentationObject', root)
            # Delete the parent of voc_root
            shutil.rmtree(os.path.join(root, os.path.dirname(base_dir)))

class VOCDetectionDataModule(DetectionDataModule):
    def __init__(self, batch_size, num_workers, 
                 root, idx_to_class=None,
                 dataset_name='VOC2012Detection',
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None):
        super().__init__(batch_size, num_workers, dataset_name,
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform)
        self.root = root
        self.idx_to_class = idx_to_class

    ###### Dataset Methods ######
    def prepare_data(self):
        """Download Pascal VOC 2012 dataset"""
        download_voc_dataset(self.root)

    def _get_datasets(self, ignore_transforms=False):
        """Dataset initialization"""
        train_dataset = VOCDetectionTV(
            self.root, image_set='train', download=False,
            transforms=self._get_transforms('train', ignore_transforms),
            transform=self._get_transform('train', ignore_transforms),
            target_transform=self._get_target_transform('train', ignore_transforms),
        )
        val_dataset = VOCDetectionTV(
            self.root, image_set='val', download=False,
            transforms=self._get_transforms('val', ignore_transforms),
            transform=self._get_transform('val', ignore_transforms),
            target_transform=self._get_target_transform('val', ignore_transforms),
        )
        test_dataset = VOCDetectionTV(
            self.root, image_set='val', download=False,
            transforms=self._get_transforms('test', ignore_transforms),
            transform=self._get_transform('test', ignore_transforms),
            target_transform=self._get_target_transform('test', ignore_transforms),
        )
        return train_dataset, val_dataset, test_dataset
    
    ###### Validation Methods ######
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        print('Exporting the filtered annotaion file...')
        # TODO: Implement this method
    
    ###### Other Methods ######
