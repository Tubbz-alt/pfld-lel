import time
import torch
import numpy as np
from options.train_options import TrainOptions
from data.wflw_dataset import createDatasets
from models.pfld_model import createModel
from util.visualizer import Visualizer
from util.util import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(trainDataLoader, pfldNet, auxiliaryNet, optimizer, epoch):
    losses = AverageMeter()
    for img, landmark_gt, attribute_gt, euler_angle_gt in enumerate(trainDataLoader):
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        attribute_gt = attribute_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        pfldNet = pfldNet.to(device)
        auxiliaryNet = auxiliaryNet.to(device)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    trainDataLoader, valDataLoader = createDatasets(opt)
    pfldNet, auxiliaryNet = createModel(opt)

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

    visualizer = Visualizer(opt)        # create a visualizer that display/save images and plots

    total_iters = 0
    for epoch in range(opt.start_epoch, opt.end_epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        visualizer.reset()
        weighted_train_loss, train_loss = train(trainDataLoader, pfldNet, auxiliaryNet, optimizer, epoch)



