from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from ..custom_path import Path
from torchvision import transforms
from ..custom_transforms import FixScaleCrop, RandomHorizontalFlip
from ..custom_transforms import RandomScaleCrop, RandomGaussianBlur
from ..custom_transforms import Normalize, ToTensor, AutoAdjustSize


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 server='039614',
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        base_dir = Path.db_root_dir('pascal', data_root=args.data_root, server=server)
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')

        if args.mode == 'weakly':
            self._cat_dir = os.path.join(self._base_dir, 'pascal_2012_scribble')
        else:
            self._cat_dir = os.path.join(self._base_dir, 'SegmentationClassAug')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'SegmentationAug')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.image_names = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                assert os.path.isfile(_image)
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_cat)

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.image_names.append(line)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def _make_img_gt_point_pair(self, index):
        image_path = self.images[index]
        category_path = self.categories[index]

        _img = Image.open(image_path).convert('RGB')
        _target = Image.open(category_path) if category_path else None

        return _img, _target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        _padding_mask = Image.fromarray(np.ones((_img.height, _img.width), dtype=np.uint8))
        sample = {'image': _img, 'padding_mask': _padding_mask, 'size': (_img.height, _img.width)}

        for split in self.split:
            if split in {'train', 'val'}:
                sample.update({'label': _target})

        for split in self.split:
            if split == "train":
                sample = self.transform_tr_part1(sample)
            elif split in {'val', 'test'}:
                sample = self.transform_val_part1(sample)
            else:
                assert False

            if split == 'train':
                sample = self.transform_tr_part2(sample)
            elif split in {'val', 'test'}:
                sample = self.transform_val_part2(sample)

        if 'padding_mask' in sample:
            del sample['padding_mask']

        sample.update({'name': self.image_names[index]})

        return sample

    def transform_tr_part1(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            RandomGaussianBlur()])  # Zhiwei

        return composed_transforms(sample)

    def transform_tr_part1_1(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size)])  # Zhiwei

        return composed_transforms(sample)

    def transform_tr_part1_2(self, sample):
        composed_transforms = transforms.Compose([RandomGaussianBlur()])

        return composed_transforms(sample)

    def transform_tr_part2(self, sample):
        composed_transforms = transforms.Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val_part1(self, sample):
        if self.args.enable_adjust_val:
            composed_transforms = transforms.Compose([
                AutoAdjustSize(factor=self.args.adjust_val_factor, fill=254)])
        else:
            composed_transforms = transforms.Compose([
                FixScaleCrop(crop_size=self.args.crop_size)])

        return composed_transforms(sample)

    def transform_val_part2(self, sample):
        composed_transforms = transforms.Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

if __name__ == '__main__':
    from ..utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)