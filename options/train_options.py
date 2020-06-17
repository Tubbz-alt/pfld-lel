import argparse
import os
import torch
import models
import data
from util import util


class TrainOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False;
    def initialize(self, parser):
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type = int, default = 200, help = 'frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type = int, default = 4, help = 'if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type = int, default = 1, help = 'window id of the web display')
        parser.add_argument('--display_server', type = str, default = "http://localhost", help = 'visdom server of the web display')
        parser.add_argument('--display_env', type = str, default = 'main', help = 'visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type = int, default = 8097, help = 'visdom port of the web display')
        parser.add_argument('--update_html_freq', type = int, default = 400, help = 'frequency of saving training results to html')
        parser.add_argument('--print_freq', type = int, default = 200, help = 'frequency of showing training results on console')
        parser.add_argument('--no_html', action ='store_true', help = 'do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--display_winsize', type = int, default= 256, help='display window size for both visdom and HTML')

        # basic parameters
        parser.add_argument('--dataroot', type = str, required = True, help = 'name of the experiment, it decides where to store samples and models')
        parser.add_argument('--name', type = str, default = 'Wflw_pfld', help = 'name of the experiment, it decides where to store samples and models')
        parser.add_argument('--gpu_ids', type = str, default = '0', help = 'GPU id is: e.g. 0 0,1,2 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type = str, default = './checkpoints', help = 'models are save here')
        # -- snapshot、tensorboard log and checkpoint
        parser.add_argument('--snapshot', type = str, default = './checkpoints/snapshot/', metavar='PATH')
        parser.add_argument('--log_file', type = str, default = "./checkpoints/train.logs")
        parser.add_argument('--tensorboard', type = str, default="./checkpoints/tensorboard")
        parser.add_argument('--resume', type = str, default = '', metavar = 'PATH')  # TBD

        # model parameters
        parser.add_argument('--model', type = str, default = 'pfld', help = 'model name is pfld')

        # dataset parameters
        parser.add_argument('--dataset_mode', type = str, default = 'wflw', help = 'chooses how datasets are loaded. [wflw | other]')
        parser.add_argument('--batch_size', type = int, default = 1, help = 'input batch size')
        #parser.add_argument('--serial_batches', action = 'store_true', help = 'if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default = 6, type = int, help = '# threads for loading data')
        parser.add_argument('--max_dataset_size', type = int, default = float("inf"), help = 'Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--save_train_dir', type = str, default = 'train_data', help = 'chooses how datasets are loaded. [wflw | other]')
        parser.add_argument('--shuffle', type = bool, default = True, help = 'DataLoader disorder')
        parser.add_argument('--save_test_dir', type = str, default = 'test_data', help = 'chooses how datasets are loaded. [wflw | other]')

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', type = str, default = '', help = 'customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # lr
        parser.add_argument('--lr', type = float, default = 0.0001, help = 'initial learning rate for adam')
        parser.add_argument("--lr_patience", type = int, default = 40)
        # train
        parser.add_argument('--phase', type = str, default = 'train', help = 'train, val, test, etc')
        parser.add_argument('--image_dir', type = str, default = 'WFLW_images', help = 'train, val, test, etc')
        parser.add_argument('--annotations_dir', type = str, default = 'WFLW_annotations', help = 'train, val, test, etc')
        parser.add_argument('--weight_decay', type = float, default = 1e-6, help = 'example L2')
        # epoch
        parser.add_argument('--start_epoch', type = int, default = 1, help = "start epoch, default 1")
        parser.add_argument('--end_epoch', type = int, default = 1000, help = "end epoch, default = 1000")
        # train dataset
        parser.add_argument('--train_list', type = str, default = 'train_data/list.txt', help = 'train list path')
        parser.add_argument('--train_batchsize', type = int, default = 128)
        parser.add_argument('--val_batchsize', type = int, default = 8)
        # val dataset
        parser.add_argument('--val_list', type = str, default = 'test_data/list.txt', help = 'test list path')


        self.isTrain = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        """
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        """
        # save and return the parser
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End ---------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    def parse(self):
        """ Parse our options, create checkpoints directory suffix, and set up gpu device."""

        # process opt.suffix
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if op.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


