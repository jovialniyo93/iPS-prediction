import torch
from torch.utils.data import DataLoader
from torch import nn,optim
#from net import Unet
#from net import FCN8s
#from net import DeepSeg
from net import TransDeepSeg
#from net import StarDistUNet
#from net import Cellpose
#from net import AttU_Net
#from net import NestedUNet
from utils import *
import numpy as np
import cv2
from tqdm import tqdm
import logging
from torchvision.transforms import transforms
import os


model_path='checkpoints/'
imgs_path='data/imgs/'
mask_path='data/mask/'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
#cv2.equalizeHist(image) / 255

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

y_transforms = transforms.Compose([
    #transforms.Lambda(lambda mask:__normalize(mask)),
    transforms.ToTensor()
])

def __normalize(mask):
    min,max=np.unique(mask)[0],np.unique(mask)[-1]
    mask=mask/1.0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i][j]=(mask[i][j]-min)/(max-min)
    mask = mask.astype(np.float16)
    return mask

def record_result(string):
    file_name="train_record.txt"
    if not os.path.exists(file_name):
        with open(file_name,'w') as f:
            print("successfully create record file")
    with open(file_name,'a') as f:
        f.write(string+"\n")
    print(string+" has been recorded")

def train_model(model,criterion,optimizer,dataload,keep_training,num_epochs=100):
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
    model.to(device)

    if keep_training:
        checkpoints=os.listdir(model_path)
        checkpoints.sort()
        final_ckpt=checkpoints[-1]
        print("Continue training from ",final_ckpt)
        restart_epoch=final_ckpt.replace("CP_epoch","").replace(".pth","")
        restart_epoch=int(restart_epoch)
        model.load_state_dict(torch.load(model_path+final_ckpt))

    else:

        restart_epoch=1
        if os.path.isfile("train_record.txt"):
            os.remove("train_record.txt")
            print("Old result has been cleaned!")

    for epoch in range(restart_epoch-1,num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        data_size=len(dataload.dataset)
        epoch_loss=0
        step=0
        for x,y in tqdm(dataload):
            step+=1
            inputs=x.to(device)
            labels=y.to(device)

            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print("epoch %d loss:%.3f"%(epoch+1,epoch_loss/step))
        record_result("epoch %d loss:%.3f"%(epoch+1,epoch_loss/step))
        try:
            os.mkdir(model_path)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(model.state_dict(),model_path + f'CP_epoch{str(epoch + 1).zfill(2)}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

if __name__=="__main__":
    keep_training=False
    #model=Unet(1,1)
    #model = FCN8s(1, 1)
    #model = DeepSeg(1, 1)
    model = TransDeepSeg(1, 1)
    #model = StarDistUNet(1, 1)
    #model = Cellpose(1, 1)
    #model = AttU_Net(1, 1)
    #model = NestedUNet(1, 1)
    batch_size=16
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters())
    data=TrainDataset(imgs_path,mask_path,x_transforms,y_transforms)

    dataloader=DataLoader(data,batch_size,shuffle=True,num_workers=4)
    train_model(model,criterion,optimizer,dataloader,keep_training,num_epochs=100)



