from torch_extend.dataset import CocoInstanceSegmentation

from .base_instance import InstanceSegDataModule

class CocoInstanceSegDataModule(InstanceSegDataModule):
    def __init__(self, batch_size, num_workers, 
                 root, train_dir='train2017', val_dir='val2017',
                 train_annFile=None, val_annFile=None,
                 dataset_name='COCOInstanceSegmentation',
                 train_transforms=None, train_transform=None, train_target_transform=None,
                 eval_transforms=None, eval_transform=None, eval_target_transform=None,
                 out_fmt='torchvision', processor=None):
        super().__init__(batch_size, num_workers, dataset_name,
                         train_transforms, train_transform, train_target_transform,
                         eval_transforms, eval_transform, eval_target_transform,
                         out_fmt, processor)
        self.root = root
        self.train_dir = train_dir
        self.val_dir = val_dir
        # Annotation files
        if train_annFile is not None:
            self.train_annFile = train_annFile
        else:
            self.train_annFile = f'{self.root}/annotations/instances_{self.train_dir}.json'
        if val_annFile is not None:
            self.val_annFile = val_annFile
        else:
            self.val_annFile = f'{self.root}/annotations/instances_{self.val_dir}.json'

    ###### Dataset Methods ######
    def _get_datasets(self, ignore_transforms=False):
        """Dataset initialization"""
        train_dataset = CocoInstanceSegmentation(
            f'{self.root}/{self.train_dir}',
            annFile=self.train_annFile,
            out_fmt=self.out_fmt,
            transforms=self._get_transforms('train', ignore_transforms),
            transform=self._get_transform('train', ignore_transforms),
            target_transform=self._get_target_transform('train', ignore_transforms),
            processor=self.processor,
        )
        val_dataset = CocoInstanceSegmentation(
            f'{self.root}/{self.val_dir}',
            annFile=self.val_annFile,
            out_fmt=self.out_fmt,
            transforms=self._get_transforms('val', ignore_transforms),
            transform=self._get_transform('val', ignore_transforms),
            target_transform=self._get_target_transform('val', ignore_transforms),
            processor=self.processor,
        )
        test_dataset = CocoInstanceSegmentation(
            f'{self.root}/{self.val_dir}',
            annFile=self.val_annFile,
            out_fmt=self.out_fmt,
            transforms=self._get_transforms('test', ignore_transforms),
            transform=self._get_transform('test', ignore_transforms),
            target_transform=self._get_target_transform('test', ignore_transforms),
            processor=self.processor,
        )
        return train_dataset, val_dataset, test_dataset
    
    ###### Validation Methods ######
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        print('Exporting the filtered annotaion file...')
        # TODO: Implement this method
    
    ###### Other Methods ######
