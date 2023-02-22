import argparse
# hehe
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.axialnet import MedT
from networks.segformer import SegFormer
from trainer import trainer_synapse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="TransUNet", help="model name: TransUNet/MedT/SegFormer")
parser.add_argument('--root_path', type=str,
                    default='datasets/raw_data/train_npz', help='root dir for data')
parser.add_argument('--output_dir', type=str,
                    default="runs/transunet")
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
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default=r'R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
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
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    checkpoint = None
    if args.model_name == "TransUNet":
        print("pretrain path: ", config_vit.pretrained_path)
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))

    elif args.model_name == "SegFormer":
        # from aiplatform.segformer_lucid.segformer_pytorch.segformer_pytorch import Segformer
        # net = Segformer(
        #     dims = (32, 64, 160, 256),      # dimensions of each stage
        #     heads = (1, 2, 5, 8),           # heads of each stage
        #     ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
        #     reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        #     num_layers = 2,                 # num layers of each stage
        #     decoder_dim = 256,              # decoder dimension
        #     num_classes = args.num_classes                 # number of segmentation classes
        # )



        net = SegFormer(num_classes=args.num_classes,image_size=args.img_size).cuda()
        # from aiplatform.segformer_pytorch.nets.segformer import SegFormer
        # net = SegFormer(num_classes=args.num_classes, phi="b0", pretrained=False)
    elif args.model_name == "MISSFormer":
        from networks.MISSFormer import MISSFormer
        net = MISSFormer(num_classes=args.num_classes).cuda()
    elif args.model_name == "SwinUNet":
        from networks.swin_vision_transformer import SwinUnet
        from aiplatform.Swin_Unet.config import get_config
        args.cfg = "aiplatform/Swin_Unet/configs/swin_tiny_patch4_window7_224_lite.yaml"
        config = get_config()
        net = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    elif args.model_name == "pvtv2":
        from aiplatform.pvtv2.pvtv2 import PyramidVisionTransformerV2 as PVT
        net = PVT(num_classes=args.num_classes).cuda()
    elif args.model_name == "MyNetworks":
        from networks.my_networks import MyNetworks
        checkpoint = torch.load("runs/epoch_66.pth")
        print(checkpoint.keys())
        net = MyNetworks(num_classes=args.num_classes).cuda()
        net.load_state_dict(checkpoint)

    else:
        args.img_size = 128
        net = MedT(img_size = args.img_size, imgchan = 1, num_classes = args.num_classes).cuda()

    trainer = {'Synapse': trainer_synapse,}

    trainer[dataset_name](args, net, args.output_dir, checkpoint=checkpoint)




