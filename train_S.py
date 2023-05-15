import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torchio as tio
from torch.utils.data import DataLoader

from unet import UNet  # Assuming you have a UNet module defined somewhere


def list_maker(dir, folder, ext):
    dir_path = os.path.join(dir, folder)
    files = os.listdir(dir_path)
    files_list = [os.path.join(dir_path, f) for f in files if f.endswith(ext)]
    return sorted(files_list)


# Image and label list creation as per your provided function
input_dir = '/media/iniyan/android/dataset/'
image_list = list_maker(input_dir, 'img', '.nii.gz')
label_list = list_maker(input_dir, 'label', '.nii.gz')  # Assuming you have labels in the same directory

# Dataset preparation using torchio
subjects = []
for img_path, lbl_path in zip(image_list, label_list):
    subject = tio.Subject(
        img=tio.ScalarImage(img_path),
        lbl=tio.LabelMap(lbl_path)
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)

# Use TorchIO to create a patches dataset
patch_size = 64
queue_length = 300
samples_per_volume = 8

sampler = tio.data.UniformSampler(patch_size)
patches_queue = tio.Queue(
    dataset,
    queue_length,
    samples_per_volume,
    sampler,
)

train_loader = DataLoader(patches_queue, batch_size=4)

# Model, optimizer and loss
model = UNet(n_channels=1, n_classes=2)  # Define according to your UNet module
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda')
model.to(device)

# Dice loss function
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=1)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2).sum(dim=1) + target.sum(dim=2).sum(dim=2).sum(dim=1) + smooth)))
    return loss.mean()

from tqdm import tqdm  # Import tqdm

# Training loop
for epoch in tqdm(range(100)):  # Use tqdm on the range of epochs
    print(f"\nEpoch {epoch+1}")
    epoch_loss = 0.0  # Initialize the epoch loss

    for i, batch in enumerate(train_loader):
        img = batch['img'][tio.DATA].float().to(device)  # Added .float() to convert ShortTensor to FloatTensor
        lbl = batch['lbl'][tio.DATA].to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = dice_loss(output, lbl)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # Accumulate the batch loss

        print(f"Batch {i+1}, Loss: {loss.item()}")

    # Print the average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Avg Loss for Epoch {epoch+1}: {avg_epoch_loss}")

# Save the model after training
torch.save(model.state_dict(), 'model.pth')
