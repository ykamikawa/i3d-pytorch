import os
import math
import functools
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data as data


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(image_paths, image_loader):
    video = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def get_labels(ann_list):
    labels = []
    for ann_path in ann_list:
        p = ann_path.split('/')[-1][:3]
        food = ann_path.split('_')[-1].split('.')[0]
        with open(ann_path, 'r') as f:
            labels.extend([(p, food, l.split(' ')[0], l.split(' ')[1]) \
                for l in f.readlines()])

    return labels


def get_sequence(labels, args):
    image_dir = args.image_dir
    cam_type = args.cam_type
    step = args.step
    sample_duration = args.sample_duration
    seq_list = []
    for label in tqdm(labels):
        p = label[0]
        food = label[1]
        start = int(label[2].split('-')[0])
        end = int(label[2].split('-')[1])
        if args.action_coarse:
            action = label[3].split('_')[0]
        else:
            action = label[3]

        if not args.SIL and action == 'SIL':
            continue

        for start_idx in range(start, end - sample_duration, step):
            seq = []
            exist_flag = True
            for idx in range(start_idx, start_idx+sample_duration):
                image_path = os.path.join(image_dir, p,
                        cam_type, food, 'image_%05d.jpg'%idx)
                seq.append(image_path)
                if not os.path.exists(image_path):
                    exist_flag = False
            if exist_flag:
                seq_list.append((seq, action))

    return seq_list


def make_dataset(ann_list, args):

    labels = get_labels(ann_list)
    dataset = get_sequence(labels, args)

    return dataset


class BreakFast(data.Dataset):

    def __init__(self, ann_list,
            class_label_map, args, spatial_transform=None,
            get_loader=get_default_video_loader):

        self.data = make_dataset(ann_list, args)
        self.class_label_map = class_label_map

        self.spatial_transform = spatial_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
            where target is class_index of the target class.
        """
        path_list = self.data[index][0]

        clip = self.loader(path_list)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        action = self.data[index][1]
        target = self.class_label_map[action]

        return clip, target

    def __len__(self):
        return len(self.data)
