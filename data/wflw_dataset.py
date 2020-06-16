import os.path
import random
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class WFLWDatasets(data.Dataset):
    def __init__(self, dataroot, file_name, transforms=None):
        file_list = os.path.join(dataroot, file_name)
        print(file_list)
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()
        print(len(self.lines))

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)
        self.attribute = np.asarray(self.line[197:203], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[203:206], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.attribute, self.euler_angle)

    def __len__(self):
        return len(self.lines)


def createDatasets(opt):
    transform = transforms.Compose([transforms.ToTensor])
    trainDatasets = WFLWDatasets(opt.dataroot, opt.train_list, transform)
    trainDataLoader = torch.utils.data.DataLoader(
        trainDatasets,
        batch_size = opt.train_batchsize,
        shuffle = opt.shuffle,
        num_workers = opt.num_threads,
        drop_last = False
    )

    valDatasets = WFLWDatasets(opt.dataroot, opt.val_list, transform)
    valDataLoader = torch.utils.data.DataLoader(
        valDatasets,
        batch_size = opt.val_batchsize,
        shuffle = opt.shuffle,
        num_workers = opt.num_threads,
        drop_last = False
    )

    print(len(trainDataLoader))
    print(len(valDataLoader))
    return trainDataLoader, valDataLoader





