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
import matplotlib.image as image

  
# case = ["0008", "0022", "0038", "0036", "0032", "0002", "0029","0003","0001","0004","0025","0035"]
# channel = ["124","65","74","152","111","93","79","100","102","94","69","68"]
case = ["0008", "0022", "0038", "0036", "0032", "0002", "0029"]
channel = ["164","186","160","225","183","177","85"]


for m in range(len(case)):
    pred_path = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/transunet/prediction_796/TU_Synapse224/TransUNet/converted/case{case[m]}_pred_channel_{channel[m]}.jpg"
    pred = cv2.imread(pred_path,0)

    gt_path = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/transunet/prediction_796/TU_Synapse224/TransUNet/converted/case{case[m]}_gt_channel_{channel[m]}.jpg"
    gt = cv2.imread(gt_path)
    img_path = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/transunet/prediction_796/TU_Synapse224/TransUNet/converted/case{case[m]}_img_channel_{channel[m]}.jpg"
    transunet_path = f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/transunet/predictionTransUNet_pth/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_20k_epo150_bs4_224/converted/case{case[m]}_pred_channel_{channel[m]}.jpg"
    with cbook.get_sample_data(gt_path) as image_file1:
        gt_plt = plt.imread(image_file1)
    with cbook.get_sample_data(pred_path) as image_file2:
        pred_plt = plt.imread(image_file2)
    with cbook.get_sample_data(img_path) as image_file3:
        img_plt = plt.imread(image_file3)
    with cbook.get_sample_data(transunet_path) as image_file4:
        trans_plt = plt.imread(image_file4)   
    img_plt = np.array(img_plt)
    test_img_pred = img_plt.copy()
    test_img_gt = img_plt.copy()
    test_img_trans = img_plt.copy()
    H,W,_ = gt_plt.shape
    # aorta, gallbladder, left_kidney, right_kidney, liver, pancreas, spleen, stomach = 32,60,96,128,160,192,224,255
    aorta, gallbladder, left_kidney, right_kidney, liver, pancreas, spleen, stomach = 220,244,40,70,170,215,28,199
    organ_list = [aorta, gallbladder, left_kidney, right_kidney, liver, pancreas, spleen, stomach]
    # organ_color = [[0,0,255], [0,255,0], [255,0,0], [102,255,255], [255,51,255], [255,255,51], [0,153,153], [224,224,224]]
    organ_color = [[255,0,0], [0,255,0], [0,0,255], [255,255,102], [255,51,255], [51,255,255], [153,153,0], [224,224,224]]

    thresh = 10


    for k in range(len(organ_list)):
        gray_color = organ_list[k]
        for i in range(H):
            for j in range(W):
                pixel_pred = pred_plt[i][j][0]
                pixel_gt = gt_plt[i][j][0]
                pixel_trans = trans_plt[i][j][0]
                if pixel_pred >= gray_color-thresh and pixel_pred <= gray_color+thresh:
                    test_img_pred[i][j] = np.array(organ_color[k])
                if pixel_gt >= gray_color-thresh and pixel_gt <= gray_color+thresh:
                    test_img_gt[i][j] = np.array(organ_color[k])  
                if pixel_trans >= gray_color-thresh and pixel_trans <= gray_color+thresh:
                    test_img_trans[i][j] = np.array(organ_color[k])   
    
    stacked_img = np.hstack((img_plt, test_img_gt,test_img_trans, test_img_pred))
    fig1, ax1 = plt.subplots()
    c = ax1.imshow(gt)
    plt.show()
    break
    # cv2.imwrite(f"E:/tai_lieu_hoc_tap/tdh/tuannca_datn/img_datn/case{case[m]}_slice{channel[m]}_stacked.jpg",stacked_img)





