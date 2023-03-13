import sys
sys.path.append('/hdd/tuannca/datn/tuannca181816')
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from tester import test_single_volume

from networks.eff_transunet import EffTransUNet
from networks.transunet import TransUNet
from networks.config.config2d.configs import get_EFNB7_config, get_r50_b16_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="TransUNet", help="model name: TransUNet/MedT/SegFormer")
parser.add_argument('--volume_path', type=str,
                    default='datasets/raw_data/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--output_dir',type=str, default="runs/medt", help="save testing phase on this path")
parser.add_argument('--pretrained_model', type=str,
                    default='runs/epoch_70_transeffunet7_final.pth', help='path to pretrain model')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='datasets/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=4, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='EFN-B7', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='runs/transunet', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("len test loader: ", len(testloader))
 
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval().cuda()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image, label = image.cuda(), label.cuda()
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
   
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)

    if args.model_name == "EffTransUNet":
        config_vit = get_EFNB7_config()
        net = EffTransUNet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        args.pretrained_model = "runs/epoch_70_transeffunet7_final.pth" 
        net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    else:
        config_vit = get_r50_b16_config()
        net = TransUNet(config_vit, img_size=args.img_size, num_classes=args.num_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))



    args.exp=dataset_name + str(args.img_size)
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+args.model_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(args.model_name)

    if args.is_savenii:
        args.test_save_dir = args.test_save_dir+f"/prediction_{args.model_name}"
        test_save_path = os.path.join(args.test_save_dir, args.exp, args.model_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)