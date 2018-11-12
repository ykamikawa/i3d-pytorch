import os
import collections
import numpy as np


def split_breakfast_train_val(label_list, split='s4'):
    if split == 's1':
        val_list = list(range(3, 16))
    elif split == 's2':
        val_list = list(range(16, 29))
    elif split == 's3':
        val_list = list(range(29, 42))
    elif split == 's4':
        val_list = list(range(42, 55))
    else:
        raise ValueError('Invalid train val split!')
    train = [label for label in label_list \
            if not int(label.split('/')[-1].split('_')[0][-2:]) in val_list]
    val = [label for label in label_list \
            if int(label.split('/')[-1].split('_')[0][-2:]) in val_list]
    return train, val


def get_ann_list(label_dir, ann_type):
    return [os.path.join(label_dir, p, f) \
             for p in os.listdir(label_dir) \
             if os.path.isdir(os.path.join(label_dir, p)) \
             for f in os.listdir(os.path.join(label_dir, p)) \
             if ann_type in f]


def get_action_labels(ann_list, action_coarse, SIL):
    all_labels = []
    for ann_path in ann_list:
        with open(ann_path, 'r') as f:
            all_labels.extend(
                    [line.split(' ')[1] for line in f.readlines()])
    if action_coarse:
        all_labels = [action.split('_')[0] for action in all_labels]
    else:
        all_labels = [action for action in all_labels]

    if not SIL:
        all_labels = [action for action in all_labels if not action == 'SIL']

    class_names = np.unique(np.array(all_labels))
    action_counter = collections.Counter(all_labels)
    print(action_counter)

    class_label_map = {}
    for i, class_name in enumerate(class_names):
        class_label_map[class_name] = i

    return class_label_map
