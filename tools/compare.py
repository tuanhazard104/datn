import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0003_img_channel_154.jpg"), (256,256,))
# img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0003_pred_channel_154.jpg"), (256,256))
# img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0003_pred_channel_154.jpg"), (256,256))
# img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0003_gt_channel_154.jpg"), (256,256))
# img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0003_pred_channel_154.jpg"), (256,256))
# img_total1 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

# img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0022_img_channel_73.jpg"), (256,256,))
# img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0022_pred_channel_73.jpg"), (256,256))
# img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0022_pred_channel_73.jpg"), (256,256))
# img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0022_gt_channel_73.jpg"), (256,256))
# img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0022_pred_channel_73.jpg"), (256,256))
# img_total2 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

# img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0008_img_channel_118.jpg"), (256,256,))
# img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0008_pred_channel_118.jpg"), (256,256))
# img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0008_pred_channel_118.jpg"), (256,256))
# img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0008_gt_channel_118.jpg"), (256,256))
# img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0008_pred_channel_118.jpg"), (256,256))
# img_total3 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

# img_raw = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0036_img_channel_183.jpg"), (256,256,))
# img_transunet = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0036_pred_channel_183.jpg"), (256,256))
# img_segformer = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_segformer\case0036_pred_channel_183.jpg"), (256,256))
# img_gt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_transunet\case0036_gt_channel_183.jpg"), (256,256))
# img_medt = cv2.resize(cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\img_medt\case0036_pred_channel_183.jpg"), (256,256))
# img_total4 = np.hstack((img_raw, img_gt, img_transunet, img_segformer, img_medt))

im1_img = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0008_img_channel_60.jpg")
im1_gt = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0008_gt_channel_60.jpg")
im1_pred = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0008_pred_channel_60.jpg")
print(im1_img.shape, im1_gt.shape, im1_pred.shape)
im1 = np.hstack((im1_img, im1_gt, im1_pred))

im2_img = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0022_img_channel_75.jpg")
im2_gt = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0022_gt_channel_75.jpg")
im2_pred = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0022_pred_channel_75.jpg")
im2 = np.hstack((im2_img, im2_gt, im2_pred))

im3_img = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0022_img_channel_70.jpg")
im3_gt = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0022_gt_channel_70.jpg")
im3_pred = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0022_pred_channel_70.jpg")
im3 = np.hstack((im3_img, im3_gt, im3_pred))

im4_img = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0038_img_channel_64.jpg")
im4_gt = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0038_gt_channel_64.jpg")
im4_pred = cv2.imread(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\converted\case0038_pred_channel_64.jpg")
im4 = np.hstack((im4_img, im4_gt, im4_pred))
img_total1234 = np.vstack((im1, im2, im3, im4))

cv2.imwrite(os.path.join("E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/combined", "combined12345.jpg"), img_total1234)