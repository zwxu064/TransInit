from .datasets import cityscapes, coco, combine_dbs, pascal, sbd, ade20k, pcontext
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    drop_last = False
    server = args.server
    train_loader, val_loader, test_loader = None, None, None
    num_class = 21

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train', server=server)
        val_set = pascal.VOCSegmentation(args, split='val', server=server)

        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'], server=server)
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=drop_last, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train', server=server)
        val_set = cityscapes.CityscapesSegmentation(args, split='val', server=server)
        test_set = cityscapes.CityscapesSegmentation(args, split='test', server=server)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=drop_last, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train', server=server)
        val_set = coco.COCOSegmentation(args, split='val', server=server)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'pcontext':
        train_set = pcontext.ContextSegmentation(split='train', mode='train')
        val_set = pcontext.ContextSegmentation(split='val', mode='val')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        num_class = train_set.num_class
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'ade20k':
        train_set = ade20k.ADE20KSegmentation(split='train', mode='train')
        val_set = ade20k.ADE20KSegmentation(split='val', mode='val')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        num_class = train_set.num_class
        test_loader = None

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError