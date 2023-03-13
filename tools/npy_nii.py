import numpy as np
import nibabel as nib
import h5py
file_path = r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\datasets\raw_data\test_vol_h5\case0001.npy.h5"
f = h5py.File(file_path)
image = np.array(f["image"])
label = np.array(f["label"])

# img_array1 = np.load(r"E:\tai_lieu_hoc_tap\tdh\tuannca_datn\datasets\raw_data\test_vol_h5\case0001.npy.h5",allow_pickle=True) 
nifti_file = nib.Nifti1Image(image , np.eye(4))
nib.save(nifti_file, "dicom_volume_image.nii.gz")