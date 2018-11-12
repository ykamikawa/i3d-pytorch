import os
import json
import random
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from args import argparser
from utils.utils import Logger
from models.model import generate_model

from train import train_epoch
from validate import val_epoch
from datasets.breakfast import BreakFast
from datasets.util import get_ann_list, split_breakfast_train_val, get_action_labels
from transforms.spatial_transforms import (
        Compose, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
        MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    # Setup manual seed
    reset_seed(args.manual_seed)

    # Architecture
    args.arch = 'i3d_{}'.format(args.modality)

    print(args, '\n')

    # Augumentation setting
    args.scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            args.scales, args.sample_size, crop_positions=['c'])

    # Spatial transform
    train_spatial_transform = Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(args.norm_value)])
    val_spatial_transform = Compose([
        Scale(args.sample_size),
        CenterCrop(args.sample_size),
        ToTensor(args.norm_value)])

    # Load training dataset
    ann_list = get_ann_list(args.ann_dir, args.ann_type)
    class_label_map = get_action_labels(ann_list, args.action_coarse, args.SIL)
    print('{} classes'.format(len(class_label_map)))
    print(class_label_map, '\n')
    args.num_classes = len(class_label_map)
    train, val = split_breakfast_train_val(ann_list)

    # BreakFast dataset class
    print('Initialize train dataset')
    train_dataset = BreakFast(train,
            class_label_map, args,
            spatial_transform=train_spatial_transform)
    print('Initialize valid dataset')
    val_dataset = BreakFast(val,
            class_label_map, args,
            spatial_transform=val_spatial_transform)
    assert (len(train_dataset) != 0)
    assert (len(val_dataset) != 0)
    print('train: {} samples'.format(len(train_dataset)))
    print('val: {} samples\n'.format(len(val_dataset)))

    # Training iterotor
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.n_threads,
        pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.n_threads,
        pin_memory=True, drop_last=True)

    # Build model
    model, parameters = generate_model(args)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    if not args.no_cuda:
        criterion = criterion.cuda()

    # Setup optimizer
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening
    optimizer = optim.SGD(parameters, lr=args.learning_rate,
        momentum=args.momentum, dampening=dampening,
        weight_decay=args.weight_decay, nesterov=args.nesterov)
    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # Prepare result directory
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if args.resume_path:
        args.result_path = os.path.join(args.result_path,
                args.resume.split('/')[-2])
    else:
        out_counter = len([None for out in os.listdir(args.result_path) \
                if args.arch in out])
        args.result_path = os.path.join(args.result_path,
                args.arch + '-' + str(out_counter+1))
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # Save configs
    with open(os.path.join(args.result_path, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

    # Logger
    log_dir = os.path.join(args.result_path, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train_logger = Logger(os.path.join(log_dir, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(os.path.join(log_dir, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    val_logger = Logger(os.path.join(log_dir, 'val.log'),
        ['epoch', 'loss', 'acc'])

    # Resume pretrained weights
    if args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        assert args.arch == checkpoint['arch']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Training Loop
    for i in range(args.start_epoch, args.max_epochs + 1):
        # Train
        train_epoch(i, train_loader, model, criterion,
                optimizer, args, train_logger, train_batch_logger)
        # Validate
        validation_loss = val_epoch(i, val_loader, model, criterion,
                args, val_logger)
        # Learning rate scheduler
        scheduler.step(validation_loss)


if __name__ == '__main__':
    args = argparser()
    main(args)
