# Code for MedT
import sys
sys.path.append('/hdd/tuannca/datn/tuannca181816')
import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os, glob
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from torch.nn.modules.loss import CrossEntropyLoss
from aiplatform.TransUNet.utils import DiceLoss
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit

from tensorboardX import SummaryWriter
train_writer = SummaryWriter("/hdd/tuannca/datn/tuannca181816/runs/log/medt/train_log")
val_writer = SummaryWriter("/hdd/tuannca/datn/tuannca181816/runs/log/medt/val_log")
parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=151, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=19, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', default = "/hdd/tuannca/datn/tuannca181816/data/Converted/size_128/train", type=str)
parser.add_argument('--val_dataset', default = "/hdd/tuannca/datn/tuannca181816/data/Converted/size_128/val", type=str)
parser.add_argument('--save_freq', type=int,default = 10)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='runs/medtfinal_model.pth', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='/hdd/tuannca/datn/tuannca181816/runs/medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=128)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='yes', type=str)
parser.add_argument('--num_classes', default=9, type=int)

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

device = torch.device(args.device)

if modelname == "axialunet":
    model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant, num_classes = args.num_classes)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)

criterion = LogNLLLoss()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)

total_iter = len(glob.glob("/hdd/tuannca/datn/tuannca181816/data/Converted/size_128/train/img/*.jpg"))/args.batch_size*args.epochs
print("total iterations: ", total_iter)
iter_num = 0
for epoch in range(args.epochs):
    print("Start epoch ", epoch)
    epoch_running_loss = [0]*2
    loss_ce_per_epoch = [0]*2
    loss_dice_per_epoch = [0]*2
    loss_ce_dice_per_epch = [0]*2
    
    print("------Train-----")
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
        
        X_batch = Variable(X_batch.to(device = device))
        y_batch = Variable(y_batch.to(device= device))

        # ===================forward=====================
   
        output = model(X_batch)
    
        # tmp2 = y_batch.detach().cpu().numpy()
        # tmp = output.detach().cpu().numpy()
        # tmp[tmp>=0.5] = 1
        # tmp[tmp<0.5] = 0
        # tmp2[tmp2>0] = 1
        # tmp2[tmp2<=0] = 0
        # tmp2 = tmp2.astype(int)
        # tmp = tmp.astype(int)

        # yHaT = tmp
        # yval = tmp2

        loss_ce = ce_loss(output, y_batch)
        loss_dice = dice_loss(output, y_batch, softmax=True) #forward
        loss_ce_dice = 0.5 * loss_ce + 0.5 * loss_dice
        loss = criterion(output, y_batch)
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_running_loss[0] += loss.item()
        loss_ce_dice_per_epch[0] += loss_ce_dice.item()
        loss_ce_per_epoch[0] += loss_ce.item()
        loss_dice_per_epoch[0] += loss_dice.item()
        
        iter_num+=1
        print(f"iter{batch_idx}/{total_iter} - loss_dice_train: {loss_dice}")
        del X_batch, y_batch, output
        
    print(f"epoch: {epoch}, sum_loss = {loss_ce_per_epoch[0]}")
    # log train loss 
    train_writer.add_scalar('info/LogNLLLoss_train', epoch_running_loss[0]/(batch_idx+1), epoch)
    train_writer.add_scalar('info/total_loss_train', loss_ce_dice_per_epch[0]/(batch_idx+1), epoch)
    train_writer.add_scalar('info/loss_ce_train', loss_ce_per_epoch[0]/(batch_idx+1), epoch)
    train_writer.add_scalar('info/loss_dice_train', loss_dice_per_epoch[0]/(batch_idx+1), epoch)

    
    if epoch == 10:
        for param in model.parameters():
            param.requires_grad =True
    
    print("------Val-----")
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
        # print(batch_idx)
        if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
        else:
                    image_filename = '%s.jpg' % str(batch_idx + 1).zfill(3)

        X_batch = Variable(X_batch.to(device=device))
        y_batch = Variable(y_batch.to(device=device))
        # start = timeit.default_timer()
        y_out = model(X_batch)
        
        loss_ce_val = ce_loss(y_out, y_batch)
        loss_dice_val = dice_loss(y_out, y_batch, softmax=True) #forward
        loss_ce_dice_val = 0.5 * loss_ce + 0.5 * loss_dice
        loss_val = criterion(y_out, y_batch)
        
        epoch_running_loss[1] += loss_val.item()
        loss_ce_dice_per_epch[1] += loss_ce_dice_val.item()
        loss_ce_per_epoch[1] += loss_ce_val.item()
        loss_dice_per_epoch[1] += loss_dice_val.item()
        epsilon = 1e-20
        
        del X_batch, y_batch,y_out

        fulldir = direc+"/{}/".format(epoch)
        
        print(f"iter{batch_idx}/{total_iter} - loss_dice: {loss_dice_val}")

    # log train loss 
    train_writer.add_scalar('info/LogNLLLoss_val', epoch_running_loss[1]/(batch_idx+1), epoch)
    train_writer.add_scalar('info/total_loss_val', loss_ce_dice_per_epch[1]/(batch_idx+1), epoch)
    train_writer.add_scalar('info/loss_ce_val', loss_ce_per_epoch[1]/(batch_idx+1), epoch)
    train_writer.add_scalar('info/loss_dice_val', loss_dice_per_epoch[1]/(batch_idx+1), epoch)
    if (epoch % args.save_freq) ==0:
        fulldir = direc+"/{}/".format(epoch)
        os.makedirs(fulldir, exist_ok=True)
        torch.save(model.state_dict(), fulldir+args.modelname+".pth")
        torch.save(model.state_dict(), direc+"final_model.pth")
        # ===================log========================
    print('epoch [{}/{}], loss_train:{:.4f}, loss_val:{:.4f}'
          .format(epoch, args.epochs, loss_dice_per_epoch[0]/(batch_idx+1), loss_dice_per_epoch[1]/(batch_idx+1)))

    del epoch_running_loss,loss_ce_per_epoch,loss_dice_per_epoch,loss_ce_dice_per_epch
