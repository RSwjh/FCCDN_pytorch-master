from networks import GenerateNet
from loss import Class_FCCDN_loss_BCD,dice_bce_loss#FCCDN_loss_BCD#FCCDN_loss
import torch
from data_utils import LoadDatasetFromFolder,ValDatasetFromFolder,calMetric_iou
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .adamW import AdamW
import itertools
import re
import os
import time
from torch.utils.tensorboard import SummaryWriter
class Config():
    def __init__(self):
        self.MODEL_NAME = 'FCCDN'
        self.MODEL_OUTPUT_STRIDE = 16
        self.BAND_NUM = 3
        self.USE_SE = True
        self.DATAPATH_TIME1='./datasets/train/time1'
        self.DATAPATH_TIME2='./datasets/train/time2'
        self.DATAPATH_LABEL='./datasets/train/label'
        self.TESTPATH_TIME1 = './datasets/test/time1'
        self.TESTPATH_TIME2 = './datasets/test/time2'
        self.TESTPATH_LABEL = './datasets/test/label'
        self.BATCHSIZE=2
        self.VAL_BATCHSIZE=2
        self.NUM_WORKERS=2
        self.LR=0.002
        self.NUM_EPOCHS=100
        self.SAVE_EPOCH_FREQ=10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def makeModel(args={},modelname='220706'):
    modeldir=os.path.join('./trained',modelname)
    fileArgs=modelname+'_Args.txt'
    txtArgs=os.path.join(modeldir,fileArgs)
    fileTrain=modelname+'.txt'
    txtTrain=os.path.join(modeldir,fileTrain)

    # 判断模型是否已训练
    # 不存在
    if not os.path.exists(modeldir):
        # 新建(name).txt和(name)_args.txt文件
        os.mkdir(modeldir)
        txtFileArgs=open(txtArgs,'w')
        txtFileTrain=open(txtTrain,'w')

    # 存在
    else:
        txtFileArgs = open(txtArgs, 'a')
        txtFileTrain = open(txtTrain, 'a')
    #编写训练参数的txt
    writetime=time.strftime("---------------%Y-%m-%d %H:%M:%S--------------- \n", time.localtime())
    txtFileArgs.write(writetime)
    txtFileTrain.write(writetime)
    keys=args.keys()
    # 填写训练参数 args.txt
    for key in keys:
        strwrite=str(key)+':'+str(args[key])
        txtFileArgs.writelines(strwrite)
        txtFileArgs.writelines('\n')
    txtFileArgs.close()
    txtFileTrain.close()


def nowEpoch(model="./trained/FCCDN_epoch_1.pth"):
    if os.path.isfile(model):
        name=os.path.basename(model)
        epoch=str(name).split('.')[0]
        epoch=epoch.split('_')[-1]
        return int(epoch)
    else:
        print ("不存在文件\'"+model+"\'")

if __name__ == '__main__':
    mloss = 0

    cfg = Config()
    argsDict=cfg.__dict__
    #dataset
    dataset_train = LoadDatasetFromFolder(cfg)
    dataset_val = ValDatasetFromFolder(cfg)
    dataloader_train=DataLoader(dataset_train,num_workers=cfg.NUM_WORKERS,batch_size=cfg.BATCHSIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, num_workers=cfg.NUM_WORKERS, batch_size=cfg.VAL_BATCHSIZE, shuffle=True)

    #model
    CDNet = GenerateNet(cfg)
    CDNet = CDNet#.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        CDNet = torch.nn.DataParallel(CDNet, device_ids=range(torch.cuda.device_count()))

    optimizer=AdamW(itertools.chain(CDNet.parameters()), lr= cfg.LR, betas=(0.9, 0.999),weight_decay=0.001)
    CD_loss = Class_FCCDN_loss_BCD()
    val_loss = dice_bce_loss()

    modelname = argsDict['MODEL_NAME']
    modeldir = os.path.join('./trained', modelname)
    fileArgs = modelname + '_Args.txt'
    txtArgs = os.path.join(modeldir, fileArgs)
    fileTrain = modelname + '.txt'
    txtTrain = os.path.join(modeldir, fileTrain)
    fileBest = 'best.txt'
    txtBest = os.path.join(modeldir, fileBest)

    # 判断模型是否已训练
    # 不存在
    first = 1
    a = os.path.exists(modeldir)
    if not os.path.exists(modeldir):
        # 新建(name).txt和(name)_args.txt文件
        os.mkdir(modeldir)
        txtFileArgs = open(txtArgs, 'w')
        txtFileTrain = open(txtTrain, 'w')
        txtFileBest = open(txtBest, 'w')
        txtFileBest.close()
        CDNet.load_state_dict(torch.load('./pretrained/FCCDN_test_LEVIR_CD.pth'))


    # 存在
    else:
        txtFileArgs = open(txtArgs, 'a')
        txtFileTrain = open(txtTrain, 'a')
        # 加载模型参数，找到latest
        fileList = os.listdir(modeldir)
        latest = [file for file in fileList if re.match('^latest_\d{1,}.pth', file)][0]
        pthPath = os.path.join(modeldir, latest)

        CDNet.load_state_dict(torch.load(pthPath))
        print('已加载模型参数：', latest)
        # 提取最后训练的epoch，继续训练
        first = int(re.findall('^latest_(.+?).pth', latest)[0]) + 1

        txtFileBest = open(txtBest, 'r')
        bestLoss = txtFileBest.readline()
        bestLoss = float(bestLoss.replace('\n', ''))
        bestIou = txtFileBest.readline()
        bestIou = float(bestIou.replace('\n', ''))
        mloss = bestIou
        txtFileBest.close()
    # 编写训练参数的txt
    writetime = time.strftime("---------------%Y-%m-%d %H:%M:%S--------------- \n", time.localtime())
    txtFileArgs.write(writetime)
    txtFileTrain.write(writetime)
    keys = argsDict.keys()
    # 填写训练参数 args.txt
    for key in keys:
        strwrite = str(key) + ':' + str(argsDict[key])
        txtFileArgs.writelines(strwrite)
        txtFileArgs.writelines('\n')
    txtFileArgs.close()
    txtFileTrain.close()
    # 编写训练epoch信息的
    writer = SummaryWriter(os.path.join(modeldir, './logs'))

    for epoch in range(1,cfg.NUM_EPOCHS+1):
        train_bar=tqdm(dataloader_train)
        running_results = {'batch_sizes': 0, 'SR_loss': 0, 'CD_loss': 0, 'loss': 0}

        CDNet.train()
        for img1,img2,labels in train_bar:
            running_results['batch_sizes'] += cfg.BATCHSIZE
            img1 = img1.to(device, dtype=torch.float)
            img2 = img2.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            #label = torch.argmax(label, 1).unsqueeze(1).float()

            y,y1,y2=CDNet(img1,img2)
            result=[y,y1,y2]
            loss=CD_loss(result,labels)
            CDNet.zero_grad()
            loss.backward()
            optimizer.step()

            running_results['CD_loss'] += loss.item() * cfg.batchsize
            train_bar.set_description(
                desc='[%d/%d] loss: %.4f' % (
                    epoch, cfg.NUM_EPOCHS,
                    running_results['CD_loss'] / running_results['batch_sizes'],))
        epochLoss = running_results['CD_loss'] / running_results['batch_sizes']
        writer.add_scalar('loss/loss', epochLoss, epoch)


        CDNet.eval()
        with torch.no_grad():
            val_bar = tqdm(dataloader_val)
            inter, unin = 0, 0
            valing_results = {'batch_sizes': 0, 'IoU': 0}

            for hr_img1, hr_img2, label in val_bar:
                valing_results['batch_sizes'] += cfg.VAL_BATCHSIZE

                hr_img1 = hr_img1.to(device, dtype=torch.float)
                hr_img2 = hr_img2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)
                #label = torch.argmax(label, 1).unsqueeze(1).float()

                y,y1,y2= CDNet(hr_img1, hr_img2)

                loss = val_loss(val_loss, label)

                # cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()

                gt_value = (label > 0).float()
                prob = (y > 0).float()
                prob = prob.cpu().detach().numpy()

                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)
                intr, unn = calMetric_iou+(gt_value, result)
                inter = inter + intr
                unin = unin + unn

                valing_results['IoU'] = (inter * 1.0 / unin)

                val_bar.set_description(
                    desc='IoU: %.4f' % (valing_results['IoU'],))

            # save model parameters
        val_iou = valing_results['IoU']
        writer.add_scalar('loss/IoU', val_iou, epoch)
        #写train情况txt
        txtFileTrain = open(txtTrain, 'a')
        txtFileTrain.write(time.strftime("---------------%Y-%m-%d %H:%M:%S--------------- \n", time.localtime()))
        txtFileTrain.write('[%d/%d] loss: %.4f  ' % (
                    epoch, cfg.NUM_EPOCHS,
                    epochLoss,))
        txtFileTrain.write('IoU: %.4f\n' % (val_iou,))

        txtFileTrain.close()
        # 保存最新的模型
        print("saveing model.............")
        torch.save(CDNet.state_dict(), os.path.join(modeldir,'latest_%d.pth'%(epoch)))
        #按照args.save_epoch_freq来保存模型参数文件
        if epoch % cfg.SAVE_EPOCH_FREQ==0 or epoch==1:
            fName='netCD_epoch_%d.pth' % (epoch)
            sPath=os.path.join(modeldir,fName)
            torch.save(CDNet.state_dict(), sPath)
        if epoch!=1:
            os.remove(os.path.join(modeldir,'latest_%d.pth'%(epoch-1)))
        if val_iou > mloss :
            mloss = val_iou

            torch.save(CDNet.state_dict(),  os.path.join(modeldir,'best.pth'))
            print("saved the best model of epoch ",str(epoch))
            txtFileBest = open(txtBest, 'w')
            txtFileBest.write('%.4f\n%.4f'%(epochLoss,val_iou))
            txtFileBest.close()

        print("model_state_dict have saved!!!!!!!")
    # labels = [change_mask]                         #labels的生成方式
    # avgpool = torch.nn.AvgPool2d((2, 2))
    # labels.append(avgpool(change_mask))
    #


    # output = CDNet(input)
    # loss = FCCDN_loss(output, label)
