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
class ConvertData():
    def __init__(self, input_dir, output_dir, sufflex="npz"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.files = glob.glob(os.path.join(input_dir,f"*.{sufflex}"))
        print("The length of the folder: ", len(self.files))
        
    def convert_h5(self):
        for i in range(len(self.files)):
            case = self.files[i][-11:-7]
    
            f = h5py.File(self.files[i]) # f.keys() = ["image", "label"]
            
            image = f["image"]
            label = f["label"]
            for j in range(image.shape[0]):
                img_arr = np.array(image)[j,:,:]
                label_arr = np.array(label)[j,:,:]
                plt.imsave(self.output_dir+"/img/"+f"case{case}_slice{j:03d}.jpg", img_arr, cmap='gray')
                plt.imsave(self.output_dir+"/labelcol/"+f"case{case}_slice{j:03d}.jpg", label_arr, cmap='gray')

                
    def convert_npz(self):
        for i in range(len(self.files)):
            basename = os.path.basename(self.files[i])[:-3]+"jpg"
            print(basename)
            f = np.load(self.files[i]) # f.keys() = ["image", "label"]
            
            image = f["image"]
            label = f["label"]
            
            plt.imsave(self.output_dir+"/img/"+basename, image, cmap='gray')
            plt.imsave(self.output_dir+"/labelcol/"+basename, label, cmap='gray')
            
    def convert_nii(self):
        for i in range(len(self.files)):
            filename_split_gz = Path(self.files[i]).stem
            filename_split_nii_gz = Path(filename_split_gz).stem
            filename_jpg = filename_split_nii_gz + ".jpg"
            
            data = nib.load(self.files[i]).get_fdata()
            *_, num_slices, num_channels = data.shape
            for channel in range(num_channels):
                volume = data[..., channel]
                volume = np.array(volume)
                # print(min(volume[1]))
                # print(volume.shape)
                slice_data = np.stack(3 * [volume], axis=2)
                channel_dir = os.path.join(self.output_dir , f'{filename_split_nii_gz}_channel_{channel}.jpg')
                plt.imsave(channel_dir, volume, cmap='gray')
                
    def resize(self):
        for i in range(len(self.files)):
            basename = os.path.basename(self.files[i])[:-3]+"jpg"
            img = cv2.imread(self.files[i])
            img_128 = cv2.resize(img, (128, 128))
            cv2.imwrite(os.path.join(self.output_dir, basename), img_128)

converter = ConvertData(input_dir=r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\prediction\TU_Synapse224\MyNetworks",
                        output_dir=r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\runs\transunet\prediction\TU_Synapse224\MyNetworks\converted55",
                        sufflex="nii.gz")

converter.convert_nii()