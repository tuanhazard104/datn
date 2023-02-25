# import h5py
# import numpy as np
# from PIL import Image

# hdf = h5py.File("data/Synapse/test_vol_h5/case0001.npy.h5",'r')
# print(hdf)
# array = hdf["Photos/Image 1"][:]
# img = Image.fromarray(array.astype('uint8'), 'RGB')
# img.save("yourimage.thumbnail", "JPEG")
# img.show()

# import h5py
# import numpy as np
# import os
# from matplotlib import pyplot as plt
# f = h5py.File("data/Synapse/test_vol_h5/case0001.npy.h5","r")
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].shape)
import cv2
import h5py
import numpy as np
import os
from matplotlib import pyplot as plt
with h5py.File("data/Synapse/test_vol_h5/case0001.npy.h5","r") as f:
    for key in f.keys():
        print(f[key].name)
        print(f[key].shape)
f = h5py.File("data/Synapse/test_vol_h5/case0001.npy.h5","r")
imagedata = f["image"]
labeldata = f["label"]
i=146
imgsel = np.array(imagedata)[i,:,:]
print(imgsel.shape)
labelsel = np.array(labeldata)[i,:,:]

plt.imsave(f"imgsel2{i}.jpg", imgsel)
plt.imsave(f"labelsel2{i}.jpg", labelsel)

plt.subplot(121)
plt.imshow(imgsel)
plt.subplot(122)
plt.imshow(labelsel)
plt.show()
