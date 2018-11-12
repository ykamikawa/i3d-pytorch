import os
import sys
import time
from tqdm import tqdm

import torch
from torch.autograd import Variable

from utils.utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, args, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
        data_time.update(time.time() - end_time)

        if not args.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs)
        targets = Variable(targets)

        with torch.no_grad():
            out, out_logits = model(inputs)
            loss = criterion(out, targets)
        acc = calculate_accuracy(out, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

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

    logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg})

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

    return losses.avg
