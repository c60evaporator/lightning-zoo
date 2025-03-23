import json
import os

from torch_extend.dataset import CocoDetection

from .base_detection import DetectionDataModule

class CocoDetectionDataModule(DetectionDataModule):
    def __init__(self, batch_size, num_workers, 
                 root, train_dir='train2017', val_dir='val2017',
                 train_annFile=None, val_annFile=None,
                 dataset_name='COCO',
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
        train_dataset = CocoDetection(
            f'{self.root}/{self.train_dir}',
            annFile=self.train_annFile,
            out_fmt=self.out_fmt,
            transforms=self._get_transforms('train', ignore_transforms),
            transform=self._get_transform('train', ignore_transforms),
            processor=self.processor,
        )
        val_dataset = CocoDetection(
            f'{self.root}/{self.val_dir}',
            annFile=self.val_annFile,
            out_fmt=self.out_fmt,
            transforms=self._get_transforms('val', ignore_transforms),
            transform=self._get_transform('val', ignore_transforms),
            processor=self.processor,
        )
        test_dataset = CocoDetection(
            f'{self.root}/{self.val_dir}',
            annFile=self.val_annFile,
            out_fmt=self.out_fmt,
            transforms=self._get_transforms('test', ignore_transforms),
            transform=self._get_transform('test', ignore_transforms),
            processor=self.processor,
        )
        return train_dataset, val_dataset, test_dataset
    
    ###### Validation Methods ######
    def _output_filtered_annotation(self, df_img_results, result_dir, image_set):
        print('Exporting the filtered annotaion file...')
        del_img_ids = df_img_results[df_img_results['anomaly']]['image_id'].tolist()
        if image_set=='train':
            coco_dataset = self.train_dataset.coco.dataset
        elif image_set == 'val':
            coco_dataset = self.val_dataset.coco.dataset
        else:
            raise RuntimeError('The `image_set` argument should be "train" or "val"')
        # Load the coco fields
        coco_info = coco_dataset['info']
        coco_licenses = coco_dataset['licenses']
        coco_images = coco_dataset['images']
        coco_annotations = coco_dataset['annotations']
        coco_categories = coco_dataset['categories']
        # Filter the images
        filtered_images = [image for image in coco_images if image['id'] not in del_img_ids]
        # Filter the annotations
        filtered_annotations = [ann for ann in coco_annotations if ann['image_id'] not in del_img_ids]
        # Output the filtered annotation JSON file
        filtered_coco = {
            'info': coco_info,
            'licenses': coco_licenses,
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': coco_categories
        }
        os.makedirs(f'{result_dir}/filtered_ann', exist_ok=True)
        with open(f'{result_dir}/filtered_ann/instances_{image_set}_filtered.json', 'w') as f:
            json.dump(filtered_coco, f, indent=None)
    
    ###### Other Methods ######
    def _sample_dataset(self, dataset, image_ratio, labels):
        """Sample the dataset"""
        # Sample the anntations

        # Save the sampled images
        pass

    def sample_dataset(self, image_ratio, labels):
        """Sample the dataset"""
