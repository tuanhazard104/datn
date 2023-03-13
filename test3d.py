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
from medpy import metric
import argparse
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from networks.unetr import UNETR
from trainer import dice
from utils.data_utils import get_loader
import nibabel as nib
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="dataset/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model_final.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
slice_map = {
    "img0008.nii.gz": 124,
    "img0022.nii.gz": 100,
    "img0038.nii.gz": 74,
    "img0036.nii.gz": 152,
    "img0032.nii.gz": 111,
    "img0002.nii.gz": 93,
    "img0029.nii.gz": 79,
    "img0003.nii.gz": 126,
    "img0001.nii.gz": 102,
    "img0004.nii.gz": 94,
    "img0025.nii.gz": 69,
    "img0035.nii.gz": 68,
}

def calculate_metric_percase(pred, gt):
    # print("pred, gt:",pred.shape, gt.shape, pred.sum(), gt.sum()) # (148, 512, 512) (148, 512, 512)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        model_dict = torch.load(pretrained_pth)
        print(">>>>>>>>>>>>>>>>>>>>",pretrained_pth)
        model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        hd95_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inferenceeeee on case {}".format(img_name))
            
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)
            
            val_labels_copy = val_labels
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs_copy = val_outputs
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            print("val_outputs:", val_outputs[0].shape)

            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            val_inputs = val_inputs.cpu().numpy()[:, 0, :, :, :]
            print("val_labels:",val_labels[0].shape)
            print("val_inputs:",val_inputs[0].shape)
            H,W,D = val_inputs[0].shape
            """
            for d in range(D):
                plt.imsave(f"dataset/runss/case{img_name}_slice{d}_label.jpg", val_labels[0, :, :, d], cmap='gray')
                plt.imsave(f"dataset/runss/case{img_name}_slice{d}_pred.jpg", val_outputs[0, :, :, d], cmap='gray')
                plt.imsave(f"dataset/runss/case{img_name}_slice{d}_img.jpg", val_inputs[0, :, :, d], cmap='gray')
            """
            dice_list_sub = []
            metric_list = 0.0 #medpy
            metric_i = []
            hd = []
            for i in range(1, 14):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                metric_i.append(calculate_metric_percase(val_outputs[0] == i, val_labels[0] == i)) #medpy
                print(organ_Dice, metric_i[i-1][0], metric_i[i-1][1])
                dice_list_sub.append(organ_Dice)
                hd.append(metric_i[i-1][1])
            metric_list += np.array(metric_i)
            
            mean_dice = np.mean(dice_list_sub)
            mean_hd = np.mean(hd)
            print("Mean Organ Diceeee: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            hd95_list_case.append(mean_hd)
        metric_list = metric_list / 12
        for i in range(1, 14):
            print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Overall Mean HD95: {}".format(np.mean(hd95_list_case)))    
            #val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            #plt.imsave(f"dataset/runss/label{img_name}_slice{slice_map[img_name]}.jpg", val_labels_copy.cpu()[0, 0, :, :, slice_map[img_name]], cmap='gray')
            #plt.imsave(f"dataset/runss/case{img_name}_slice{slice_map[img_name]}.jpg", torch.argmax(val_outputs_copy, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='gray')
            #cv2.imwrite(f"dataset/runs/case{img_name}_slice{slice_map[img_name]}.jpg",val_outputs[0, :, :, slice_map[img_name]])


if __name__ == "__main__":
    main()
