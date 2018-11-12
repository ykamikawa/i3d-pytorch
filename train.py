import os
import sys
import time
from tqdm import tqdm

import torch
from torch.autograd import Variable

from utils.utils import AverageMeter, calculate_accuracy


def train_epoch(epoch, data_loader, model, criterion,
        optimizer, args, epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    # Training mode
    model.train()

    # Training meter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # Prepare checkpoint dir
    checkpoint_dir = os.path.join(args.result_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    end_time = time.time()

    # train loop
    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()

        data_time.update(time.time() - end_time)
        if not args.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)

        out, out_logits = model(inputs)
        loss = criterion(out, targets)
        acc = calculate_accuracy(out, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[-1]['lr']
        })

        if i % args.print_frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[-1]['lr']})

    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
              epoch,
              i+1,
              len(data_loader),
              batch_time=batch_time,
              data_time=data_time,
              loss=losses,
              acc=accuracies))

    if epoch % args.checkpoint == 0:
        save_file_path = os.path.join(checkpoint_dir,
                'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}
        torch.save(states, save_file_path)
