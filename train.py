import time
import torch
import os
import logging
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
from data.wflw_dataset import createDatasets
from models.pfld_model import createModel
from models.pfld_model import PFLDLoss
from util.util import AverageMeter
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def train(trainDataLoader, pfldNet, auxiliaryNet, optimizer, epoch, criterion, train_batchsize):
    print("===> Train:")
    losses = AverageMeter()
    for img, landmark_gt, attribute_gt, euler_angle_gt in trainDataLoader:
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        attribute_gt = attribute_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        pfldNet = pfldNet.to(device)
        auxiliaryNet = auxiliaryNet.to(device)
        features, landmarks = pfldNet(img)
        angle = auxiliaryNet(features)


        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, opt.train_batchsize)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        losses.update(loss.item())

    return weighted_loss, loss

def validate(valDataLoader, pfldNet, auxiliaryNet, criterion):
    pfldNet.eval()
    auxiliaryNet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in valDataLoader:
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfldNet = pfldNet.to(device)
            auxiliaryNet = auxiliaryNet.to(device)
            _, landmark = pfldNet(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))

    return np.mean(losses)



if __name__ == '__main__':
    opt = TrainOptions().parse()
    trainDataLoader, valDataLoader = createDatasets(opt)
    pfldNet, auxiliaryNet = createModel(opt)
    criterion = PFLDLoss()

    print('---------- Networks initialized -------------')
    num_params = 0
    for param in pfldNet.parameters():
        num_params += param.numel()
    if opt.verbose:
        print(pfldNet)

    print('pfldNet Total number of parameters : %.3f M' % (num_params / 1e6))

    num_params = 0
    for param in auxiliaryNet.parameters():
        num_params += param.numel()
    if opt.verbose:
        print(auxiliaryNet)

    print('auxiliaryNet Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')

    optimizer = torch.optim.Adam(
        [{'params': pfldNet.parameters()}, {'params': auxiliaryNet.parameters()}],
        lr = opt.lr,
        weight_decay = opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = opt.lr_patience, verbose = True)

    writer = SummaryWriter(opt.tensorboard)
    for epoch in range(opt.start_epoch, opt.end_epoch + 1):
        start_train = time.time()
        weighted_train_loss, train_loss = train(trainDataLoader, pfldNet, auxiliaryNet, optimizer, epoch, criterion, opt.train_batchsize)

        filename = os.path.join(
            str(opt.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({
            'epoch': epoch,
            'pfldNet': pfldNet.state_dict(),
            'auxiliaryNet': auxiliaryNet.state_dict()
        }, filename)

        val_loss = validate(valDataLoader, pfldNet, auxiliaryNet, criterion)
        end_val = time.time()
        d_time = end_val - start_train
        print("one epoch train + val time is {:.4f}".format(d_time))

        scheduler.step(val_loss)
        writer.add_scalar('datasets/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('datasets/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()


