from pathlib import Path

import torchio as tio
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np

import os

from model import VNet

def list_maker(dir, folder, ext):
    dir_path = os.path.join(dir, folder)
    print(dir_path)
    files = os.listdir(dir_path)
    files_list = [os.path.join(dir_path, f) for f in files if f.endswith(ext)]
    return sorted(files_list)

input_dir = '/media/iniyan/android/dataset/'

image_list = list_maker(input_dir, 'img', '.nii.gz')
label_list = list_maker(input_dir, 'label', '.nii.gz')

subjects = []
for img_path, label_path in zip(image_list, label_list):
    subject = tio.Subject(
        {
            "CT": tio.ScalarImage(img_path),
            "Label": tio.LabelMap(label_path)
        }
    )
    subjects.append(subject)

for subject in subjects:
    assert subject["CT"].orientation == ("L", "P", "S")
    


class ClipIntensity(tio.Transform):
    def __init__(self, min_val=0, max_val=2500):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image_name, image in subject.get_images_dict().items():
            if isinstance(image, tio.ScalarImage):
                intensity = image.numpy()
                intensity[intensity < self.min_val] = self.min_val
                intensity[intensity > self.max_val] = self.max_val
                intensity = intensity.astype(float) / self.max_val
                subject[image_name] = tio.ScalarImage(tensor=torch.from_numpy(intensity).float())
        return subject

process = tio.Compose([
            ClipIntensity(),
            tio.CropOrPad((256, 256, 256)),
            ])


augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))

val_transform = process
train_transform = tio.Compose([process, augmentation])

train_dataset = tio.SubjectsDataset(subjects[:35], transform=train_transform)
val_dataset = tio.SubjectsDataset(subjects[35:], transform=val_transform)

sampler = tio.data.LabelSampler(patch_size=96, label_name="Label")
# sampler = tio.data.LabelSampler(patch_size=96, label_name="Label", label_probabilities={0:0.5, 1:0.5})

train_patches_queue = tio.Queue(
    train_dataset,
    max_length=10,
    samples_per_volume=5,
    sampler=sampler,
    num_workers=12,
)

val_patches_queue = tio.Queue(
    val_dataset,
    max_length=10,
    samples_per_volume=5,
    sampler=sampler,
    num_workers=12,
)

batch_size = 1

train_loader = torch.utils.data.DataLoader(train_patches_queue, batch_size=batch_size, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_patches_queue, batch_size=batch_size, num_workers=0)



class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = VNet()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train")
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Val Loss", loss)
        self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val")
        
        return loss

    
    def log_images(self, img, pred, mask, name):
        
        results = []
        pred = torch.argmax(pred, 1) # Take the output with the highest value
        axial_slice = 50  # Always plot slice 50 of the 96 slices
        
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][:,:,axial_slice]==0, mask[0][:,:,axial_slice])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")
        
        axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][:,:,axial_slice]==0, pred[0][:,:,axial_slice])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")

        self.logger.experiment.add_figure(f"{name} Prediction vs Label", fig, self.global_step)

            
    
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.optimizer]
    # Create the trainer

# Instanciate the model
model = Segmenter()

# Create the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='Val Loss',
    save_top_k=10,
    mode='min')




gpus = 1  

trainer = pl.Trainer(
    accelerator='gpu',  # Use 'ddp' for Distributed Data Parallel
    gpus=gpus,
    logger=TensorBoardLogger(save_dir="./logs"),
    log_every_n_steps=1,
    callbacks=checkpoint_callback,
    max_epochs=100
)

# Move your model to the GPU
model.to(device=torch.device('cuda'))

# Move your train and validation data to the GPU (if not already)
# train_loader = train_loader.to(device=torch.device('cuda'))
# val_loader = val_loader.to(device=torch.device('cuda'))

print("here")
trainer.fit(model, train_loader, val_loader)


# # Change the gpus parameter to the number of available gpus in your computer. Use 0 for CPU training

# gpus = 0
# trainer = pl.Trainer(accelerator=gpus,logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1,
#                      callbacks=checkpoint_callback,
#                      max_epochs=100)

# print("here")
# trainer.fit(model, train_loader, val_loader)