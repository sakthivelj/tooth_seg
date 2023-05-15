import nibabel as nib
import numpy as np

# Define the cropping size
crop_size = (256, 256, 256)

# Load the 3D image and label
image_path = "img/1000813648_20180116.nii.gz"
label_path = "label/1000813648_20180116.nii.gz"

# Load the image and label using nibabel
image_nifti = nib.load(image_path)
label_nifti = nib.load(label_path)

# Get the image and label data arrays
image = image_nifti.get_fdata()
label = label_nifti.get_fdata()

# Crop the image and label
image_shape = image.shape
crop_start = [(image_shape[i] - crop_size[i]) // 2 for i in range(3)]
crop_end = [crop_start[i] + crop_size[i] for i in range(3)]
cropped_image = image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
cropped_label = label[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

# Save the cropped image and label using nibabel
cropped_image_nifti = nib.Nifti1Image(cropped_image, image_nifti.affine, image_nifti.header)
cropped_label_nifti = nib.Nifti1Image(cropped_label, label_nifti.affine, label_nifti.header)

cropped_image_path = "cropped_image.nii.gz"
cropped_label_path = "cropped_label.nii.gz"

nib.save(cropped_image_nifti, cropped_image_path)
nib.save(cropped_label_nifti, cropped_label_path)

print("Cropped images and labels saved successfully.")
