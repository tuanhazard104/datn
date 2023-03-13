import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.eff_transunet import EffTransUNet
from networks.transunet import TransUNet
from networks.eff_transunet_3d import EffTransUNet3D
from networks.config.config2d.configs import get_EFNB7_config, get_r50_b16_config
from trainer2d import trainer_synapse as trainer_2d
from trainer3d import run_training as trainer_3d
parser = argparse.ArgumentParser()
parser.add_argument('--volume', action="store_true", help='add this argument to train with 3d data')
parser.add_argument('--model_name', type=str, default="EffTransUNet", help="model name: EffTransUNet or TransUNet")
parser.add_argument('--root_path', type=str,
                    default='datasets/data_2d/train_npz', help='root dir for data')
parser.add_argument('--output_dir', type=str,
                    default="runs")
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='datasets/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

parser.add_argument('--resume_training', action="store_true", help='using pretrained model')
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
args = parser.parse_args()

CUDA_VISIBLE_DEVICES=0

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    checkpoint = None
    if args.volume:
        args.output_dir = args.output_dir + "3d"
        net = EffTransUNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        img_size=(96,96,96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0).cuda()
        args.pretrained_model = ""
        checkpoint = torch.load(args.pretrained_model)
        net.load_state_dict(checkpoint["state_dict"])
        trainer_3d(
                    model=net,
                    acc_func=dice_acc,
                    args=args,
                    model_inferer=model_inferer,
                    scheduler=scheduler,
                    start_epoch=start_epoch,
                    post_label=post_label,
                    post_pred=post_pred)

        
    if args.model_name == "EffTransUNet":
        config_vit = get_EFNB7_config()
        net = EffTransUNet(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
        args.pretrained_model = "runs/epoch_70_transeffunet7_final.pth" 
        if args.resume_training:
            checkpoint=torch.load(args.pretrained_model)
            net.load_state_dict(checkpoint)
    
    else:
        config_vit = get_r50_b16_config()
        net = TransUNet(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer_2d(args, net, args.output_dir, checkpoint=checkpoint)




