import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from aiplatform.TransUNet.utils import DiceLoss, Tversky_Loss
from torchvision import transforms
from torch.nn import functional as F
from aiplatform.swin_unetr.MONAI.monai.losses import TverskyLoss

def trainer_synapse(args, model, snapshot_path, checkpoint=None):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transform)
    print("The length of train set is: {}".format(len(db_train)))
    print("db_train:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n ", db_train)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    tversky_loss = Tversky_Loss(num_classes)
    dice_loss = DiceLoss(num_classes)
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    # print(trainloader)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # print("model input size: ",image_batch.size())
            outputs = model(image_batch)
            if args.model_name == "SegFormer":
                outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            # print(outputs.size(), label_batch[:].long().size()) # torch.Size([2, 9, 256, 256]) torch.Size([2, 256, 256])
            # print(outputs[3].size(), outputs[2].size(), outputs[1].size(), outputs[0].size()) # torch.Size([2, 512, 7, 7]) torch.Size([2, 256, 14, 14]) torch.Size([2, 128, 28, 28]) torch.Size([2, 64, 56, 56])
            loss_ce = ce_loss(outputs, label_batch[:].long())
            # print(label_batch[:].long().size()) #torch.Size([4, 224, 224])
            loss_dice = dice_loss(outputs, label_batch, softmax=True) #forward
            loss_tversky = tversky_loss(outputs, label_batch, softmax=True)
            # print(label_batch.size())
            loss = 0.3 * loss_ce + 0.7 * loss_tversky
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_tversky', loss_ce, iter_num)
            logging.info('iteration %d : loss: %f, loss_ce: %f, loss tversky: %f' % (iter_num, loss.item(), loss_ce.item(), loss_tversky.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        # save checkpoint after 1 epoch
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"