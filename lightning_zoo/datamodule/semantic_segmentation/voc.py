from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.voc import DATASET_YEAR_DICT

from torch_extend.dataset import VOCSemanticSegmentation

from ..detection.voc import download_voc_dataset
from .base_semantic import SemanticSegDataModule

class VOCSemanticSegDataModule(SemanticSegDataModule):
    def __init__(self, batch_size, num_workers, 
                 root, idx_to_class=None,
                 dataset_name='VOC2012SemanticSeg',
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None,
                 out_fmt='torchvision', processor=None,
                 border_idx=255, bg_idx=0):
        super().__init__(batch_size, num_workers, dataset_name,
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform,
                         out_fmt, processor,
                         border_idx, bg_idx)
        self.root = root
        self.idx_to_class = idx_to_class

    ###### Dataset Methods ######
    def prepare_data(self):
        """Download Pascal VOC 2012 dataset"""
        download_voc_dataset(self.root)

    def _get_datasets(self, ignore_transforms=False):
        """Dataset initialization"""
        train_dataset = VOCSemanticSegmentation(
            self.root, border_idx=self.border_idx, bg_idx=self.bg_idx,
            image_set='train', download=False,
            transforms=self._get_transforms('train', ignore_transforms),
            transform=self._get_transform('train', ignore_transforms),
            target_transform=self._get_target_transform('train', ignore_transforms),
        )
        val_dataset = VOCSemanticSegmentation(
            self.root, border_idx=self.border_idx, bg_idx=self.bg_idx,
            image_set='val', download=False,
            transforms=self._get_transforms('val', ignore_transforms),
            transform=self._get_transform('val', ignore_transforms),
            target_transform=self._get_target_transform('val', ignore_transforms),
        )
        test_dataset = VOCSemanticSegmentation(
            self.root, border_idx=self.border_idx, bg_idx=self.bg_idx,
            image_set='val', download=False,
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
