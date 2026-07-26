import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.networks.nets import BasicUNet
from monai.losses.dice import GeneralizedDiceLoss
from torchvision.transforms import functional as TF
import random
import time

# Directories for the data
BRIGHTFIELD_DIR = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/brightfield'
MASK_DIR = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/masks'

class CellDataset(Dataset):
    def __init__(self, well_folders, brightfield_dir, mask_dir, transform=None, augment=False):
        """
        Initialize the dataset with a list of well folders.
        """
        self.brightfield_dir = brightfield_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment

        # Store all images and their corresponding well folders
        self.image_paths = []
        self.masks_paths = []

        for well_folder in well_folders:
            well_path = os.path.join(brightfield_dir, well_folder)
            # Get list of image files that match the criteria
            images = sorted([f for f in os.listdir(well_path) if f.endswith('z06c2_ORG.tif')])
            for img in images:
                image_path = os.path.join(well_path, img)

                # Extract sample prefix for mask lookup
                sample_prefix = image_path.split('_')[7].split('z')[0]  # Adjust based on your filename structure
                mask_filename = f"{well_folder}_{sample_prefix}z06c1_ORG_mask.tiff"
                mask_path_full = os.path.join(mask_dir, mask_filename)

                # Verify if the mask exists
                if not os.path.exists(mask_path_full):
                    mask_path_full = os.path.join(mask_dir, f"{mask_dir}/{well_folder}_{sample_prefix}z06c1_ORGmask.tiff")

                if os.path.exists(mask_path_full):
                    self.image_paths.append(image_path)
                    self.masks_paths.append(mask_path_full)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the single image
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert('L'))  # Load as grayscale
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Shape: (1, H, W)

        # Load the corresponding mask
        mask_path_full = self.masks_paths[idx]
        mask = np.array(Image.open(mask_path_full).convert('L'))
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Shape: (1, H, W)

        if self.augment:
            image, mask = self.apply_augmentations(image, mask)

        return image, mask

    def apply_augmentations(self, image, mask):
        # Apply the same random transformations to both image and mask
        if random.random() > 0.5:  # Horizontal flip
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:  # Vertical flip
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:  # Random rotation
            angle = random.uniform(-180, 180)  # Rotate within a range of -30 to 30 degrees
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        return image, mask

def get_dataloader(well_folders, batch_size, brightfield_dir, mask_dir, shuffle=False):
    dataset = CellDataset(well_folders, brightfield_dir, mask_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

print("..................this is unet with one image....................")   
batch_size = 16
num_epochs = 30
learning_rate = 0.001

# Define lists of folders for training, validation, and testing
train_folders = ['Alexa488_Fibroblasts_well3_200locations', 'Alexa488_Fibroblasts_well4_225locations','Alexa488_Fibroblasts_well5_225locations','Alexa488_Fibroblasts_well6_135locations','Alexa488_Fibroblasts_well7_135locations']
val_folders = ['Alexa488_Fibroblasts_well2_200locations']
test_folders = ['Alexa488_Fibroblasts_well1_50locations']


val_loader = get_dataloader(val_folders, batch_size, BRIGHTFIELD_DIR, MASK_DIR, shuffle=False)
test_loader = get_dataloader(test_folders, batch_size, BRIGHTFIELD_DIR, MASK_DIR, shuffle=False)

print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Testing dataset size: {len(test_loader.dataset)}")
# For training set, enable augmentation
train_dataset = CellDataset(train_folders, BRIGHTFIELD_DIR, MASK_DIR, augment=True)
print(f"Training dataset size: {len(train_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Testing samples: {len(test_loader.dataset)}")



# Print a sample batch shape
for images, masks in train_loader:
    print(f"Images Shape: {images.shape}")  # Expected: (batch_size, 11, H, W)
    print(f"Masks Shape: {masks.shape}")    # Expected: (batch_size, 1, H, W)
    break

num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

# Initialize model, loss function, and optimizer
model = BasicUNet(
    spatial_dims=2,  # Change to 2 if you're working with 2D images
    in_channels=1,
    out_channels=2,
    features = (32, 64, 128, 256, 512, 32),
    dropout=0.2
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adjust the learning rate as needed
GDL = GeneralizedDiceLoss(include_background=False,to_onehot_y=True,softmax=False)


avg_train_losses = []
avg_val_losses = [] 
train_losses = []
val_losses = []
epoch_losses =[]
val_epoch_losses = []

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for inputs, masks in train_loader:
        # Move data to device (e.g., GPU if available)
        inputs = inputs.to(device)

        masks = masks.to(device)
        
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.softmax(dim=1)

        # Calculate loss (using continuous output from sigmoid)
        loss = GDL(outputs, masks)
        train_losses.append(loss.item())
        epoch_loss += loss.item()
        batch_count += 1

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss for each epoch
    epoch_loss = epoch_loss / batch_count
    epoch_losses.append(epoch_loss)
    print("Epoch: {}, Train Loss: {}".format(epoch, epoch_loss))

    # Evaluation/Validation step (optional)
    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0.0
        val_batch_count = 0
        for val_inputs, val_masks in val_loader:  # Or use a separate validation loader
            val_inputs = val_inputs.to(device)
            val_masks = val_masks.to(device)


            prediction = model(val_inputs)
            prediction = prediction.softmax(dim=1)
            
            loss = GDL(prediction, val_masks)
            val_losses.append(loss.item())
            val_epoch_loss += loss.item()
            val_batch_count += 1

    val_epoch_loss = val_epoch_loss / val_batch_count
    val_epoch_losses.append(val_epoch_loss)
    print("Epoch: {},  Val Loss: {}".format(epoch, val_epoch_loss))
# Directory to save the loss plot
plots_dir = "/zhome/7b/3/168395/Desktop/DEEP_LEARNING/plots"

# Save loss plot
def save_loss_plot(train_losses, val_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(plots_dir, "loss_plot_unetwith1.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved as '{plot_path}'")   

save_loss_plot(epoch_losses,val_epoch_losses,num_epochs)
print("-----------now testing-----------")
# Testing
model.eval()
predictions_dir = "/zhome/7b/3/168395/Desktop/DEEP_LEARNING/predwith1"

def compute_iou(preds, targets):
    """
    Compute the Intersection over Union (IoU) score.

    Args:
        preds (torch.Tensor): Predicted masks (logits or probabilities).
        targets (torch.Tensor): Ground truth masks.

    Returns:
        float: IoU score.
    """
    # Binarize predictions based on the threshold
    # Ensure inputs are binary (optional, for safety)
    preds = preds.float()
    targets = targets.float()

    # Compute intersection and union
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds + targets) - intersection

    # Avoid division by zero
    if union == 0:
        return 1.0  # Perfect score if both are empty

    return intersection / union


test_loss = 0.0
iou_test = 0.0
total_samples = len(test_loader.dataset)

with torch.no_grad():
    for idx, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass through the model
        segmentation_output = model(images)
        segmentation_output = segmentation_output.softmax(dim=1)
        
        # Calculate the loss
        loss = GDL(segmentation_output, masks)
        test_loss += loss.item()
        
        preds = torch.argmax(segmentation_output, dim=1)

        # Compute IoU for the batch
        batch_iou = compute_iou(preds, masks)
        iou_test += batch_iou.item() * images.size(0)  # Scale by batch size
        for i in range(preds.shape[0]):
            plt.figure(figsize=(8, 8))
            # Move prediction to CPU before converting to NumPy
            plt.imshow(preds[i].cpu().numpy(), cmap='gray')
            plt.axis('off')

            prediction_path = os.path.join(predictions_dir, f"prediction_{idx * preds.shape[0] + i}.tiff")
            plt.savefig(prediction_path, bbox_inches='tight', pad_inches=0)
            plt.close()

end_time = time.time()

# Compute average test loss and IoU
test_loss /= len(test_loader)
average_iou = iou_test / total_samples
print(f"Test Loss: {test_loss:.4f}")
print(f"IoU: {average_iou:.4f}")

print(f"Predictions saved in '{predictions_dir}' folder.")
inference_time = end_time - start_time
print(f"Inference : {inference_time:.4f} seconds")