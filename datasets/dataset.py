from kinetics import Kinetics
from activitynet import ActivityNet
from ucf101 import UCF101
from hmdb51 import HMDB51


def get_training_set(args, spatial_transform, temporal_transform,
                     target_transform):
    assert args.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    if args.dataset == 'kinetics':
        training_data = Kinetics(
            args.video_path,
            args.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif args.dataset == 'activitynet':
        training_data = ActivityNet(
            args.video_path,
            args.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif args.dataset == 'ucf101':
        training_data = UCF101(
            args.video_path,
            args.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif args.dataset == 'hmdb51':
        training_data = HMDB51(
            args.video_path,
            args.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(args, spatial_transform, temporal_transform,
                       target_transform):
    assert args.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']

    if args.dataset == 'kinetics':
        validation_data = Kinetics(
            args.video_path,
            args.annotation_path,
            'validation',
            args.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    elif args.dataset == 'activitynet':
        validation_data = ActivityNet(
            args.video_path,
            args.annotation_path,
            'validation',
            False,
            args.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    elif args.dataset == 'ucf101':
        validation_data = UCF101(
            args.video_path,
            args.annotation_path,
            'validation',
            args.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    elif args.dataset == 'hmdb51':
        validation_data = HMDB51(
            args.video_path,
            args.annotation_path,
            'validation',
            args.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    return validation_data


def get_test_set(args, spatial_transform, temporal_transform, target_transform):
    assert args.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51']
    assert args.test_subset in ['val', 'test']

    if args.test_subset == 'val':
        subset = 'validation'
    elif args.test_subset == 'test':
        subset = 'testing'
    if args.dataset == 'kinetics':
        test_data = Kinetics(
            args.video_path,
            args.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    elif args.dataset == 'activitynet':
        test_data = ActivityNet(
            args.video_path,
            args.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    elif args.dataset == 'ucf101':
        test_data = UCF101(
            args.video_path,
            args.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)
    elif args.dataset == 'hmdb51':
        test_data = HMDB51(
            args.video_path,
            args.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=args.sample_duration)

    return test_data
