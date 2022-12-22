from pathlib import Path
import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
from skimage import io
import cv2

def to_uint8(data):
    data -= data.min()
    data /= data.max()
    data *= 255
    return data.astype(np.uint8)


def nii_to_jpgs(input_path, output_dir, rgb=False):
    output_dir = Path(output_dir)
    data = nib.load(input_path).get_fdata()
    *_, num_slices, num_channels = data.shape
    for channel in range(num_channels):
        volume = data[..., channel]
        volume = to_uint8(volume)
        channel_dir = output_dir / f'channel_{channel}'
        channel_dir.mkdir(exist_ok=True, parents=True)
        for slice in range(num_slices):
            slice_data = volume[..., slice]
            if rgb:
                slice_data = np.stack(3 * [slice_data], axis=2)
            output_path = channel_dir / f'channel_{channel}_slice_{slice}.jpg'
            plt.imsave(output_path, slice_data)

def nii_2_jpg(input_dir, output_dir=None, rgb=True):
    files = glob.glob(os.path.join(input_dir,"*.nii.gz"))
    print("len(files): ",len(files))
    for i in range(len(files)):
        # filename = os.path.basename(files[i])
        filename_split_gz = Path(files[i]).stem
        filename_split_nii_gz = Path(filename_split_gz).stem
        filename_jpg = filename_split_nii_gz + ".jpg"
        # data = np.load(files)
        data = nib.load(files[i]).get_fdata()
        *_, num_slices, num_channels = data.shape
        for channel in range(num_channels):
            volume = data[..., channel]
        
            volume = np.array(volume)
            # print(min(volume[1]))
            # print(volume.shape)
            slice_data = np.stack(3 * [volume], axis=2)
            channel_dir = os.path.join(output_dir , f'{filename_split_nii_gz}_channel_{channel}.jpg')
            plt.imsave(channel_dir, volume)
           # channel_dir.mkdir(exist_ok=True, parents=True)
            # for slice in range(num_slices):
            #     slice_data = volume[..., slice]
            #     if rgb:
            #         print(slice_data.shape)
            #         slice_data = np.stack(3 * [slice_data], axis=2)
            #     output_path = os.path.join(channel_dir , f'channel_{channel}_slice_{slice}.jpg')
            #     print(slice_data.shape)
            #     io.imsave(output_path, slice_data)
        print(f"{filename_split_nii_gz} -- {data.shape}")
        #plt.imsave(os.path.join(output_dir, filename_jpg), data)
        print(f"{100*i/len(files)}%")
        # if i ==10:
        #     break
def checking(output_dir):
    path = os.path.join(output_dir, "*.jpg")
    images_path = glob.glob(path)
    count_512=0
    count_other=0
    for i in range(len(images_path)):
        img=cv2.imread(images_path[i])
        if int(img.shape[0]) == 512:
            count_512+=1
        else:
            print(img.shape[0])
            count_other+=1
    print(count_512)

if __name__ == "__main__":
    input_dir = "TransUnet_kaggle/project_TransUNet/predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_20k_epo50_bs8_224"
    output_dir = "predictions\TU_Synapse224\converted"
    nii_2_jpg(input_dir, output_dir=output_dir)
    #checking(output_dir) # check xem trong folder anh co anh nao khac size 512x512 khong