class Path(object):
  @staticmethod
  def db_root_dir(dataset, data_root=None, server='039614', enable_testset=False):
    if data_root is None:
      if dataset == 'pascal':
        # folder that contains pascal/. It should have three subdirectories
        # called "JPEGImages", "SegmentationClassAug", and "pascal_2012_scribble"
        # containing RGB images, groundtruth, and scribbles respectively.
        if server == '039614':
          if enable_testset:
            return '/home/users/u5710355/Datasets/PASCAL/VOC2012_test/'
          else:
            return '/home/users/u5710355/Datasets/Weakly-Seg/pascal_scribble/'
        else:
          if enable_testset:
            return '/home/xu064/WorkSpace/git-lab/pytorch-projects/train_superpixel/data/VOC2012_test/'
          else:
            return '/home/xu064/WorkSpace/git-lab/pytorch-projects/train_superpixel/data/pascal_scribble/'
      elif dataset == 'coco':
        if server == 'data61':
          return '/home/xu064/Datasets/COCO/'
        else:
          return '/path/to/datasets/coco/'
      else:
        print('Dataset {} not available.'.format(dataset))
        raise NotImplementedError

    return data_root