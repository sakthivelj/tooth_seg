from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
import h5py
import os
from IPython.display import HTML


def list_maker(dir, folder, ext):
    dir_path = os.path.join(dir, folder)
    print(dir_path)
    files = os.listdir(dir_path)
    files_list = [os.path.join(dir_path, f) for f in files if f.endswith(ext)]
    return sorted(files_list)

input_dir = 'crop_256/'

image_list = list_maker(input_dir, 'img', '.nii.gz')
label_list = list_maker(input_dir, 'label', '.nii.gz')


for imglist in range(0, len(image_list)):
    data = nib.load(image_list[imglist])
    label = nib.load(label_list[imglist])

    ct = data.get_fdata()
    mask = label.get_fdata().astype(int) 

    print(nib.aff2axcodes(data.affine))


    fig = plt.figure()
    camera = Camera(fig)  # Create the camera object from celluloid

    for i in range(ct.shape[2]):  # Axial view
        plt.imshow(ct[:,:,i], cmap="bone")
        mask_ = np.ma.masked_where(mask[:,:,i]==0, mask[:,:,i])
        plt.imshow(mask_, alpha=0.5)
        # plt.axis("off")
        camera.snap()  # Store the current slice
    plt.tight_layout()
    animation = camera.animate()  # Create the animation

    print("Video")
    # HTML(animation.to_html5_video())

    video_location = f"video_{imglist}.mp4"
     # Store the video location

    animation.save(video_location)
