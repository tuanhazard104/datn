import os,sys
import tqdm
import glob
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import h5py
import nibabel as nib
from pathlib import Path
import torchvision.transforms as T
import SimpleITK as sitk
sys.path.append(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn")
from datasets.dataset_synapse import Synapse_dataset
class ConvertData():
    def __init__(self, input_dir, output_dir, sufflex="npz"):
        # self.synapse = Synapse_dataset(base_dir="")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.files = glob.glob(os.path.join(input_dir,f"*.{sufflex}"))
        print("The length of the folder: ", len(self.files))
        
    def convert_h5(self):
        for i in range(len(self.files)):
            case = self.files[i][-11:-7]
    
            f = h5py.File(self.files[i]) # f.keys() = ["image", "label"]
            
            images = f["image"]
            labels = f["label"]
            images = np.array(images)
            labels = np.array(labels)
            print(case, images.shape, labels.shape)
            img_itk = sitk.GetImageFromArray(images.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(labels.astype(np.float32))
            img_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(img_itk, self.output_dir+"/imagesTr"+ f"/img{case}.nii.gz")
            sitk.WriteImage(lab_itk, self.output_dir+"/labelsTr"+ f"/label{case}.nii.gz")
            
            # for j in range(image.shape[0]):
            #     img_arr = np.array(image)[j,:,:]
            #     label_arr = np.array(label)[j,:,:]
            #     plt.imsave(self.output_dir+"/img/"+f"case{case}_slice{j:03d}.jpg", img_arr, cmap='gray')
            #     plt.imsave(self.output_dir+"/label/"+f"case{case}_slice{j:03d}.jpg", label_arr, cmap='gray')
            # nifti_file_img = nib.Nifti1Image(image , np.eye(4))
            # nifti_file_label = nib.Nifti1Image(label , np.eye(4))
            # nib.save(nifti_file_img, self.output_dir+"/images"+ f"/img{case}.nii.gz")
            # nib.save(nifti_file_label, self.output_dir+"/labels"+ f"/label{case}.nii.gz")

                
    def convert_npz(self):
        for i in range(len(self.files)):
       
            basename = os.path.basename(self.files[i])[:-3]+"jpg"
            print(basename)
            f = np.load(self.files[i]) # f.keys() = ["image", "label"]
            
            image = f["image"]
            label = f["label"]

            plt.imsave(self.output_dir+"/img/"+basename, image, cmap='gray')
            plt.imsave(self.output_dir+"/label/"+basename, label, cmap='gray')
            
    def convert_nii(self):
        for i in range(len(self.files)):
            filename_split_gz = Path(self.files[i]).stem
            filename_split_nii_gz = Path(filename_split_gz).stem
            filename_jpg = filename_split_nii_gz + ".jpg"
            
            data = nib.load(self.files[i]).get_fdata()
            *_, num_slices, num_channels = data.shape
            print(filename_split_nii_gz, data.shape)
            # for channel in range(num_channels):
            #     volume = data[..., channel]
            #     volume = np.array(volume)
            #     # print(min(volume[1]))
            #     # print(volume.shape)
            #     # slice_data = np.stack(3 * [volume], axis=2)
            #     channel_dir = os.path.join(self.output_dir , f'{filename_split_nii_gz}_channel_{channel}.jpg')
            #     plt.imsave(channel_dir, volume, cmap='gray')
            #     # cv2.imwrite(channel_dir,slice_data)
            # break
                
    def resize(self):
        for i in range(len(self.files)):
            basename = os.path.basename(self.files[i])[:-3]+"jpg"
            img = cv2.imread(self.files[i])
            img_128 = cv2.resize(img, (128, 128))
            cv2.imwrite(os.path.join(self.output_dir, basename), img_128)

    def npz2nii(self):
        case_list = []
        for i in range(len(self.files)):
            
            case = self.files[i][-17:-13]
            # slice = self.files[i][-7:-4]
            if not case in case_list:
                # print(case)
                case_list.append(case) 
        print(case_list, len(case_list))
        total = 0
        for casee in case_list:
            images = []
            labels = []
            slices = glob.glob(self.input_dir+"/"+f"case{casee}_slice*.npz")
            for slice in slices:
                f = np.load(slice)
                image = f["image"]
                label = f["label"]
                images.append(image)
                labels.append(label)
            images = np.array(images)
            labels = np.array(labels)
            print(images.shape, labels.shape)
            img_itk = sitk.GetImageFromArray(images.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(labels.astype(np.float32))
            img_itk.SetSpacing((1, 1, 1))
            lab_itk.SetSpacing((1, 1, 1))
            sitk.WriteImage(img_itk, self.output_dir+"/imagesTr"+ f"/img{casee}.nii.gz")
            sitk.WriteImage(lab_itk, self.output_dir+"/labelsTr"+ f"/label{casee}.nii.gz")
        #     print(casee, len(slices))
        #     total+=len(slices)
        # print(total)
            # f = np.load(self.files[i]) # f.keys() = ["image", "label"]
            
            # image = f["image"]
            # label = f["label"]
            # print(case, image.shape, label.shape)
            # # for j in range(image.shape[0]):
            # #     img_arr = np.array(image)[j,:,:]
            # #     label_arr = np.array(label)[j,:,:]

            # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
            # img_itk.SetSpacing((1, 1, 1))
            # lab_itk.SetSpacing((1, 1, 1))
            # sitk.WriteImage(img_itk, self.output_dir+"/imagesTr"+ f"/img{case}.nii.gz")
            # sitk.WriteImage(lab_itk, self.output_dir+"/labelsTr"+ f"/label{case}.nii.gz")
            # break

converter = ConvertData(input_dir=r"datasets\data_3d\imagesTr",
                        output_dir=r"datasets_new\imagesTr",
                        sufflex="nii.gz")

# converter = ConvertData(input_dir=r"datasets_new\imagesTr",
#                         output_dir=r"datasets_new\imagesTr",
#                         sufflex="nii.gz")

converter.convert_nii()