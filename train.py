import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
CUDA_VISIBLE_DEVICES=0
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="transunet", help="transunet/segformer/medt")
parser.add_argument('--train_path', type=str, default="none", help='training data path')
parser.add_argument('--runs_path', type=str, default="none", help='log training/testing result')
parser.add_argument('--list_path', type=str, default='none', help='filename for train/test')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--num_epochs', type=int, default=5, help='number of training epoch')
parser.add_argument('--batch_size', type=int, default=1, help='batch size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='input size of network')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

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

