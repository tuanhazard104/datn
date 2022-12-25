import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0003_img_channel_154.jpg"), (256,256,))
img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0003_pred_channel_154.jpg"), (256,256))
img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0003_pred_channel_154.jpg"), (256,256))
img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0003_gt_channel_154.jpg"), (256,256))
img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0003_pred_channel_154.jpg"), (256,256))
img_total1 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0022_img_channel_73.jpg"), (256,256,))
img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0022_pred_channel_73.jpg"), (256,256))
img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0022_pred_channel_73.jpg"), (256,256))
img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0022_gt_channel_73.jpg"), (256,256))
img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0022_pred_channel_73.jpg"), (256,256))
img_total2 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0008_img_channel_118.jpg"), (256,256,))
img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0008_pred_channel_118.jpg"), (256,256))
img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0008_pred_channel_118.jpg"), (256,256))
img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0008_gt_channel_118.jpg"), (256,256))
img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0008_pred_channel_118.jpg"), (256,256))
img_total3 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0036_img_channel_183.jpg"), (256,256,))
img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0036_pred_channel_183.jpg"), (256,256))
img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0036_pred_channel_183.jpg"), (256,256))
img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0036_gt_channel_183.jpg"), (256,256))
img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0036_pred_channel_183.jpg"), (256,256))
img_total4 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

img_total1234 = np.vstack((img_total1, img_total2, img_total3, img_total4))
cv2.imwrite(os.path.join("E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/combined", "combined.jpg"), img_total1234)