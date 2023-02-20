import sys
sys.path.append('/hdd/tuannca/datn/tuannca181816')
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import glob
import os
# plt.plot(epchs, loss_plt, marker="o", markersize=1)
event_count = 0
value_count = 0

# E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\transunet\transunet\log\events.out.tfevents.1676372802.aiserver
# files = glob.glob("model/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224/log/*.aiserver")
# for j in range(len(files)):
#     basename = os.path.basename(files[j])
#     print(basename)
#     imgname = basename+"total_loss.jpg"
#     summary = summary_iterator(files[j])
#     for i,event in enumerate(summary): # moi event co mot value
#         for value in event.summary.value:
#             if value.tag == "info/total_loss": # loss_ce
#                 if value.HasField('simple_value'):
#                     iters.append(i)
#                     loss.append(value.simple_value)
#     # iter = []
#     # for i in range(10500):
#     #     iter.append(i)
#     print("length of iters: ", len(iters))
#     print("length of loss: ", len(loss))

#     plt.figure()
#     plt.xlabel('Iters')
#     plt.ylabel('Total Loss')

#     plt.plot(iters, loss)
#     plt.savefig(imgname)
total_loss = []
ce_loss = []
dice_loss = []
tversky_loss = []
iters = []
file_name = "E:/tai_lieu_hoc_tap/tdh/tuannca_datn/runs/transunet/transunet/transunet/log/events.out.tfevents.1676372802.aiserver"
file_name1 = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\transunet\transunet\log\events.out.tfevents.1676365275.aiserver"
filenamey530 = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\events.out.tfevents.1675101858.Y530"
swin_file = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\aiplatform\Swin_Unet\outputs\log\swin.Y530"
filenname = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\abc.aiserver"
mynetlog = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\log\final_mynet.Y530"
colab_log1 = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\log\events.out.tfevents.1676844150.49efd7d13c53"
colab_log2 = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\log\events.out.tfevents.1676838941.49efd7d13c53"
colab_log3 = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\events.out.tfevents.16768colab69672.54ca5bfa2118"
summary = summary_iterator(colab_log3)
for i,event in enumerate(summary): # moi event co mot value
    for value in event.summary.value:
        if value.tag == "info/total_loss": # loss_ce
            if value.HasField('simple_value'):
                iters.append(i)
                total_loss.append(value.simple_value)
        elif value.tag == "info/loss_ce":
            if value.HasField('simple_value'):
                # iters.append(i)
                ce_loss.append(value.simple_value)
        elif value.tag == "info/loss_tversky":
            if value.HasField('simple_value'):
                # iters.append(i)
                tversky_loss.append(value.simple_value)
print("length of iters: ", len(iters))
print("length of total_loss: ", len(total_loss))
print("length of ce_loss:",len(ce_loss))
print("len of tversky loss:",len(tversky_loss))
plt.figure()
plt.xlabel('Iters')
plt.ylabel('Total Loss')
plt.plot(iters, total_loss)
plt.savefig(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\total_loss_colab3.jpg")

plt.figure()
plt.xlabel('Iters')
plt.ylabel('CE Loss')
plt.plot(iters, ce_loss)
plt.savefig(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\ce_loss_colab3.jpg")

plt.figure()
plt.xlabel('Iters')
plt.ylabel('Tversky Loss')
plt.plot(iters, tversky_loss)
plt.savefig(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\tversky_loss_colab3.jpg")