import os
import nibabel as nib
import numpy as np

# Define the cropping size
crop_size = (256, 256, 256)


def list_maker(dir, folder, ext):
    dir_path = os.path.join(dir, folder)
    print(dir_path)
    files = os.listdir(dir_path)
    files_list = [os.path.join(dir_path, f) for f in files if f.endswith(ext)]
    return sorted(files_list)
    

input_dir = '/media/iniyan/android/dataset/'

image_list = list_maker(input_dir, 'img', '.nii.gz')
label_list = list_maker(input_dir, 'label', '.nii.gz')  

for cropimg_lab in range(0, len(image_list)):

    # Load the 3D image and label
    image_path = image_list[cropimg_lab]
    label_path = label_list[cropimg_lab]

    print(image_path)
    print(label_path)

    # Load the image and label using nibabel
    image_nifti = nib.load(image_path)
    label_nifti = nib.load(label_path)

    # Get the image and label data arrays
    image = image_nifti.get_fdata()
    label = label_nifti.get_fdata()

    # Calculate center of ROI from the label mask
    mask = (label > 0).astype(np.int32)  # Assuming the mask has values > 0 where the ROI is
    mask_center = np.array(np.where(mask)).mean(axis=1, dtype=int)

    # Calculate the crop boundaries based on the center ROI and crop size
    crop_start = [mask_center[i] - crop_size[i] // 2 for i in range(3)]
    crop_end = [crop_start[i] + crop_size[i] for i in range(3)]

    # Ensure the crop boundaries are within the image dimensions
    crop_start = [max(start, 0) for start in crop_start]
    crop_end = [min(end, dim) for end, dim in zip(crop_end, image.shape)]

    # Crop the image and label
    cropped_image = image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
    cropped_label = label[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    # Save the cropped image and label using nibabel
    cropped_image_nifti = nib.Nifti1Image(cropped_image, image_nifti.affine, image_nifti.header)
    cropped_label_nifti = nib.Nifti1Image(cropped_label, label_nifti.affine, label_nifti.header)

    img_name = f"crop_256/img/image_{cropimg_lab}.nii.gz"
    label_name = f"crop_256/label/label_{cropimg_lab}.nii.gz"
    cropped_image_path = os.path.join(input_dir, img_name)
    cropped_label_path = os.path.join(input_dir, label_name)

    nib.save(cropped_image_nifti, cropped_image_path)
    nib.save(cropped_label_nifti, cropped_label_path)

    print("Cropped images and labels saved successfully.")
