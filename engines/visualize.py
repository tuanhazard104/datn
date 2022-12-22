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
loss = []
iters = []

files = glob.glob("model/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_224/log/*.aiserver")
for j in range(len(files)):
    basename = os.path.basename(files[j])
    print(basename)
    imgname = basename+"total_loss.jpg"
    summary = summary_iterator(files[j])
    for i,event in enumerate(summary): # moi event co mot value
        
        for value in event.summary.value:
            if value.tag == "info/total_loss": # loss_ce
                if value.HasField('simple_value'):
                    iters.append(i)
                    loss.append(value.simple_value)
    # iter = []
    # for i in range(10500):
    #     iter.append(i)
    print("length of iters: ", len(iters))
    print("length of loss: ", len(loss))

    plt.figure()
    plt.xlabel('Iters')
    plt.ylabel('Total Loss')

    plt.plot(iters, loss)
    plt.savefig(imgname)
