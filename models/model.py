import torch

from models.i3d import I3D, get_fine_tuning_parameters
from models.layers import Unit3Dpy


def generate_model(args):
    model = I3D(num_classes=400, modality=args.modality)

    if args.pretrain_path:
        model.load_state_dict(torch.load(args.pretrain_path))

    model.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=args.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    parameters = get_fine_tuning_parameters(
            model,
            args.ft_begin_index)

    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=None)

    return model, parameters
