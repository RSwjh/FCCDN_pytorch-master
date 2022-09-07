from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
import torch
def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp

def get_transform(transform_list):
    transCompose=[]
    if 'ToTensor' in transform_list:
        transCompose.append(transforms.ToTensor())
    if 'Normalize' in transform_list:
        transCompose.append(transforms.Normalize(mean=0,std=1))
    return transforms.Compose(transCompose)


def make_one_hot(input,num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

class LoadDatasetFromFolder(Dataset):
    def __init__(self,cfg ,mode='train'):
        time1_path = cfg.DATAPATH_TIME1
        time2_path = cfg.DATAPATH_TIME2
        label_path = cfg.DATAPATH_LABEL

        super(LoadDatasetFromFolder,self).__init__()
        namelist=[imgname for imgname in os.listdir(time1_path)]
        self.tm1_filenames=[os.path.join(time1_path,name) for name in namelist ]
        self.tm2_filenames = [os.path.join(time2_path, name) for name in namelist]
        self.lab_filenames = [os.path.join(label_path, name) for name in namelist]

        self.transform=get_transform(['ToTensor','Normalize'])
        self.label_transform = get_transform(['ToTensor'])

        self.avgpool = torch.nn.AvgPool2d((2, 2))

    def __getitem__(self,index):
        tm1=self.transform(Image.open(self.tm1_filenames[index]).convert('RGB'))
        tm2 = self.transform(Image.open(self.tm2_filenames[index]).convert('RGB'))
        label=self.label_transform(Image.open(self.lab_filenames[index]))
        labels = [label]                         #labels的生成方式
        labels.append(self.avgpool(label))
        return tm1,tm2,labels

    def __len__(self):
        return len(self.tm1_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self,cfg ,mode='train'):
        time1_path = cfg.DATAPATH_TIME1
        time2_path = cfg.DATAPATH_TIME2
        label_path = cfg.DATAPATH_LABEL

        super(ValDatasetFromFolder,self).__init__()
        namelist=[imgname for imgname in os.listdir(time1_path)]
        self.tm1_filenames = [os.path.join(time1_path, name) for name in namelist ]
        self.tm2_filenames = [os.path.join(time2_path, name) for name in namelist]
        self.lab_filenames = [os.path.join(label_path, name) for name in namelist]

        self.transform=get_transform(['ToTensor','Normalize'])
        self.label_transform = get_transform(['ToTensor'])

        # self.avgpool = torch.nn.AvgPool2d((2, 2))

    def __getitem__(self,index):
        tm1=self.transform(Image.open(self.tm1_filenames[index]).convert('RGB'))
        tm2 = self.transform(Image.open(self.tm2_filenames[index]).convert('RGB'))
        label=self.label_transform(Image.open(self.lab_filenames[index]))
        # labels = [label]                         #labels的生成方式
        # labels.append(self.avgpool(label))
        return tm1,tm2,label #labels

    def __len__(self):
        return len(self.tm1_filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self,cfg ,mode='train'):
        time1_path = cfg.DATAPATH_TIME1
        time2_path = cfg.DATAPATH_TIME2
        label_path = cfg.DATAPATH_LABEL

        super(TestDatasetFromFolder,self).__init__()
        namelist=[imgname for imgname in os.listdir(time1_path)]
        self.tm1_filenames=[os.path.join(time1_path,name) for name in namelist ]
        self.tm2_filenames = [os.path.join(time2_path, name) for name in namelist]

        self.transform=get_transform(['ToTensor','Normalize'])
        self.label_transform = get_transform(['ToTensor'])


    def __getitem__(self,index):
        tm1=self.transform(Image.open(self.tm1_filenames[index]).convert('RGB'))
        tm2 = self.transform(Image.open(self.tm2_filenames[index]).convert('RGB'))
        image_name = self.tm1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name)-1]
        return tm1,tm2,image_name

    def __len__(self):
        return len(self.tm1_filenames)