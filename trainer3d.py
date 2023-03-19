# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils3d.utils import distributed_all_gather

from monai.data import decollate_batch
from medpy import metric
from utils3d.data_utils import get_loader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from utils3d.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
from monai.inferers import sliding_window_inference
def dice(x, y):
    # print("x, y:",x,y)
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(), target.cuda()
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=True):
            logits = model(data)
            # print("logits:",logits.size())
            loss = loss_func(logits, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss.update(loss.item(), n=args.batch_size)
        """
        print(
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        """
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, post_label=None, post_pred=None):
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[96,96,96],
        sw_batch_size=1,
        predictor=model,
        overlap=0.5)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(), target.cuda()
            with autocast(enabled=True):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda()

            acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            print(
                "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "acc",
                avg_acc,
                "time {:.2f}s".format(time.time() - start_time))
            start_time = time.time()
    return avg_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.output_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    args,
    model_inferer=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = SummaryWriter(args.output_dir + '/log')
    scaler = GradScaler()
    val_acc_max = 0.0
    train_loader, val_loader = get_loader(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=args.max_epochs)
    loss_func = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)

    for epoch in range(start_epoch, args.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        print(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time))
        writer.add_scalar("train_loss", train_loss, epoch)

        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred)
            print(
                "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                "acc",
                val_avg_acc,
                "time {:.2f}s".format(time.time() - epoch_time))
            writer.add_scalar("val_acc", val_avg_acc, epoch)
            save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                # save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler, filename=f"{epoch}.pt")
                print("Copying to model.pt new best model!!!!")
                shutil.copyfile(os.path.join(args.output_dir, "model_final.pt"), os.path.join(args.output_dir, "best_model.pt"))

        scheduler.step()

    # print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
