import argparse


def argparser():
    parser = argparse.ArgumentParser()
    # Dataset paths
    parser.add_argument('--ann_dir', default='/NAS6/share/Breakfast/lab_raw', type=str,
            help='Breakfast dataset annotation directory')
    parser.add_argument('--image_dir', default='/NAS6.share/Breakfast/Breakfast_images_jpg', type=str,
            help='Directory path of Videos')
    parser.add_argument('--ann_type', default='coarse', type=str,
            help='Dataset annotation type')
    parser.add_argument('--cam_type', default='cam01', type=str,
            help='Camera type')
    parser.add_argument('--result_path', default='results', type=str,
            help='Result directory path')
    parser.add_argument('--n_classes', default=400, type=int,
            help='Number of classes')

    # Training options
    parser.add_argument('--start_epoch', default=1, type=int,
            help='Training starts at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--max_epochs', default=100, type=int,
            help='Number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int,
            help='Batch Size')
    parser.add_argument('--n_threads', default=16, type=int,
            help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=2, type=int,
            help='Trained model is saved at every this epochs.')
    parser.add_argument('--print_frequency', default=32, type=int,
            help='Print report frequency')

    # Optimize params
    parser.add_argument('--optimizer', default='sgd', type=str,
            help='Currently only support SGD')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
            help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_patience', default=5, type=int,
            help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--nesterov', default=False, type=bool,
            help='Nesterov momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
            help='Weight Decay')
    parser.add_argument('--momentum', default=0.9, type=float,
            help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float,
            help='dampening of SGD')

    # Input options
    parser.add_argument('--sample_size', default=224, type=int,
            help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int,
            help='Temporal duration of inputs')
    parser.add_argument('--step', default=8, type=int,
            help='Sequence slide step')
    parser.add_argument('--norm_value', default=255, type=int,
            help='Image normalize value')
    parser.add_argument('--action_coarse', default='',
            help='If True, action level is coarse.')
    parser.add_argument('--SIL', default='',
            help='If True, Use SIL label')
    parser.add_argument('--modality', default='rgb',
            help='Modality of input, rgb or flow')

    # Augumentation
    parser.add_argument('--train_crop', default='corner', type=str,
        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--initial_scale', default=1.0, type=float,
            help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
        help='Scale step for multiscale cropping')

    # GPU options
    parser.add_argument('--no_cuda', default=False,
            help='If true, cuda is not used.')

    # Trained parameters
    parser.add_argument('--pretrain_path', default='pretrained/rgb.pth', type=str,
            help='Pretrained model (.pth)')
    parser.add_argument('--ft_begin_index', default=6, type=int,
            help='Begin block index of fine-tuning')
    parser.add_argument('--resume_path', default='', type=str,
            help='Save data (.pth) of previous training')

    # Seed
    parser.add_argument('--manual_seed', default=1, type=int,
            help='Manually set random seed')

    args = parser.parse_args()
    return args
