import os,sys
import glob
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import h5py
import nibabel as nib
from pathlib import Path
import torchvision.transforms as T
sys.path.append('/hdd/tuannca/datn/tuannca181816')
import matplotlib.cbook as cbook
# import matplotlib.image as image

cases = ["0002", "0008", "0008", "0008", "0022","0022","0022","0022","0025","0025","0025","0025","0029","0032"]
channels = [189,162,187,212,129,148,173,193,123,151,184,188,73,163]
aorta, gallbladder, left_kidney, right_kidney, liver, pancreas, spleen, stomach = 220,244,40,70,170,215,28,199
organ_list = [aorta, gallbladder, left_kidney, right_kidney, liver, pancreas, spleen, stomach]
# organ_color = [[0,0,255], [0,255,0], [255,0,0], [102,255,255], [255,51,255], [255,255,51], [0,153,153], [224,224,224]]
organ_color = [[255,0,0], [0,255,0], [0,0,255], [255,255,102], [255,51,255], [51,255,255], [153,153,0], [224,224,224]]

# case = "0008"
# slice = "164"
for m in range(len(cases)):
    case = cases[m]
    slice = channels[m]
    im0_path = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/transeffunet3d/research-contributions/UNETR/BTCV/dataset/runss/caseimg{case}.nii.gz_slice{slice}_img.jpg"
    gttt = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/transeffunet3d/research-contributions/UNETR/BTCV/dataset/runss/caseimg{case}.nii.gz_slice{slice}_label.jpg"
    path_3d = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/transeffunet3d/research-contributions/UNETR/BTCV/dataset/runss/caseimg{case}.nii.gz_slice{slice}_pred.jpg"
    # gt_2d = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/transunet/prediction_796/TU_Synapse224/TransUNet/converted/case{case[m]}_gt_channel_{channel[m]}.jpg"

    # Reading an image in default mode
    im0 = cv2.resize(cv2.imread(im0_path,0), (224,224))
    im1 = cv2.resize(cv2.imread(path_3d,0), (224,224))
    gt = cv2.resize(cv2.imread(gttt,0), (224,224))

    # Window name in which image is displayed
    window_name = 'Image'
    
    # Using cv2.flip() method
    # Use Flip code 0 to flip vertically
    im1 = cv2.flip(im1, 0)
    gt = cv2.flip(gt,0)
    im0 = cv2.flip(im0,0)

    im1 = cv2.rotate(im1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im0 = cv2.rotate(im0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    test_img_gt = im0.copy()

    H,W = 224,224
    thresh = 20 
    # for k in range(len(organ_list)):
    #     gray_lolor = organ_list[k]
    #     for i in range(H):
    #         for j in range(W):
    #             gt_pixel = gt[i][j]
    #             if gt_pixel[0] != 0:
    #                 test_img_gt[i][j] = np.array([int(gt_pixel[0]), int(gt_pixel[1]), int(gt_pixel[2]/2)])
    

    # fig1, ax1 = plt.subplots()
    # c = ax1.imshow(gt)
    # fig2, ax2 = plt.subplots()
    # # gt = np.array(gt,dtype=np.float32)
    # c = ax2.imshow(test_img_gt)

    # plt.show()

    stacked_img = np.hstack((im0, gt,im1))
    plt.imsave(f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/img_datn/3d/img{case}_slice{slice}.jpg", stacked_img)