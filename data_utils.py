from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
import torch
import cv2
import albumentations as A
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
    def __init__(self, cfg):
        time1_path = cfg.DATAPATH_TIME1
        time2_path = cfg.DATAPATH_TIME2
        label_path = cfg.DATAPATH_LABEL

        super(LoadDatasetFromFolder, self).__init__()
        namelist = [imgname for imgname in os.listdir(time1_path)]
        self.tm1_filenames = [os.path.join(time1_path, name) for name in namelist]
        self.tm2_filenames = [os.path.join(time2_path, name) for name in namelist]
        self.lab_filenames = [os.path.join(label_path, name) for name in namelist]

        self.album_transform = A.Compose([A.Flip(p=0.5),
                                          A.Transpose(p=0.5),
                                          A.Rotate(p=0.3, limit=(-45, 45)),
                                          A.ShiftScaleRotate(p=0.3)
                                          ])
        self.image_transform = A.Compose(
            [A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=5, val_shift_limit=10, p=0.3),
             A.GaussNoise(p=0.3)
             ])

        self.avgpool = torch.nn.AvgPool2d((2, 2))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])  #
        self.label_transform = transforms.ToTensor()

    def __getitem__(self, index):
        tm1 = cv2.imread(self.tm1_filenames[index])
        tm1 = cv2.cvtColor(tm1, cv2.COLOR_BGR2RGB)
        tm2 = cv2.imread(self.tm2_filenames[index])
        tm2 = cv2.cvtColor(tm2, cv2.COLOR_BGR2RGB)
        lab = cv2.imread(self.lab_filenames[index])

        data_album = self.album_transform(image=lab, masks=[tm1, tm2])
        tm1 = data_album['masks'][0]
        tm2 = data_album['masks'][1]
        lab = data_album['image']

        image_album = self.image_transform(image=tm1, mask=tm2)
        tm1 = image_album['image']
        tm2 = image_album['mask']

        tm1 = self.transform(tm1)
        tm2 = self.transform(tm2)
        lab = self.label_transform(lab)

        labels = [lab]  # labels的生成方式
        labels.append(self.avgpool(lab))
        return tm1, tm2, labels

    def __len__(self):
        return len(self.tm1_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self,cfg ,mode='train'):
        time1_path = cfg.VALPATH_TIME1
        time2_path = cfg.VALPATH_TIME2
        label_path = cfg.VALPATH_LABEL

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
        time1_path = cfg.TESTPATH_TIME1
        time2_path = cfg.TESTPATH_TIME2
        label_path = cfg.TESTPATH_LABEL

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
