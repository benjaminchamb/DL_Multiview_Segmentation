import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from torchvision.transforms import functional as TF
from monai.losses.dice import GeneralizedDiceLoss
import random

torch.cuda.empty_cache()

print("Title: Enhanced CNN with FFT with GDL ")

# Directories for the data
BRIGHTFIELD_DIR = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/brightfield'
MASK_DIR = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/masks'
PLOTS_DIR = "/zhome/61/b/209122/Deep_Learing/Plots"
PREDICTIONS_DIR = "/zhome/61/b/209122/Deep_Learing/predictions"

# Custom Dataset class for loading images and masks
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

            # Get list of image files
            images = sorted([f for f in os.listdir(well_path) if f.endswith('ORG.tif')])
            num_samples = len(images) // 11  # Assuming 11 images per sample

            for idx in range(num_samples):
                # Get the 11 images for each sample
                sample_images = images[idx * 11:(idx + 1) * 11]
                sample_images = [os.path.join(well_path, img) for img in sample_images]

                # Extract sample prefix for mask lookup
                sample_prefix = sample_images[0].split('_')[7].split('z')[0]
                mask_filename = f"{well_folder}_{sample_prefix}z06c1_ORG_mask.tiff"
                mask_path_full = os.path.join(mask_dir, mask_filename)

                # Verify if the mask exists
                if not os.path.exists(mask_path_full):
                    mask_path_full = os.path.join(mask_dir, f"{mask_dir}/{well_folder}_{sample_prefix}z06c1_ORGmask.tiff")

                if os.path.exists(mask_path_full):
                    self.image_paths.append(sample_images)
                    self.masks_paths.append(mask_path_full)

    def __len__(self):
        return len(self.image_paths)
    
    def fft_filter_image(self, image_array, radius=60):
    # Take the 2D FFT of the image
        image_fft = fft2(image_array)
        shifted_image_fft = fftshift(image_fft)
        # Create a circular mask using vectorized operations
        rows, cols = image_array.shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2
        # Apply mask to the shifted FFT image
        filtered_fft = shifted_image_fft * mask
        # Inverse FFT to get the filtered image back in the spatial domain
        filtered_image = np.abs(ifft2(ifftshift(filtered_fft)))
        return filtered_image

    def __getitem__(self, idx):
        # Load and optionally apply FFT filtering to the 11 images
        image_paths = self.image_paths[idx]
        images = []
        for img_path in image_paths:
            img = np.array(Image.open(img_path).convert('L'))
            img = self.fft_filter_image(img)
            images.append(img)
        images = np.stack(images, axis=0)  # Shape: (11, H, W)
        images = torch.tensor(images, dtype=torch.float32) / 255.0

        # Load the corresponding mask
        mask_path_full = self.masks_paths[idx]
        mask = np.array(Image.open(mask_path_full).convert('L'))
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask = mask.unsqueeze(0)  # Shape: (1, H, W)

        if self.augment:
            images, mask = self.apply_augmentations(images, mask)

        return images, mask

    def apply_augmentations(self, images, mask):
    # Apply the same random transformations to both images and masks
        if random.random() > 0.5:  # Horizontal flip
            images = TF.hflip(images)
            mask = TF.hflip(mask)

        if random.random() > 0.5:  # Vertical flip
            images = TF.vflip(images)
            mask = TF.vflip(mask)

        if random.random() > 0.5:  # Random rotation
            angle = random.uniform(-180, 180)  # Rotate within a range of -30 to 30 degrees
            images = TF.rotate(images, angle)
            mask = TF.rotate(mask, angle)

        return images, mask

# Function to get DataLoader for all wells
def get_dataloader(well_folders, batch_size, brightfield_dir, mask_dir, shuffle=False):
    dataset = CellDataset(well_folders, brightfield_dir, mask_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#############################  DEFINE MODEL  #############################

class EnhancedCNN(nn.Module):
    def __init__(self, input_channels=1, input_size=(1024, 1024)):
        super(EnhancedCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First Block: 64 filters
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Downsample by 2x

            # Second Block: 128 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Third Block: 256 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()  # Binary segmentation output
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.dropout(x)  # Apply dropout to encoder output
        
        # Decoder
        segmentation_output = self.decoder(x)
        
        return segmentation_output

##########################################################################




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
    plot_path = os.path.join(PLOTS_DIR, "loss_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved as '{plot_path}'")


#############################  HYPERPARAMETERS  #############################
batch_size = 16 ## 16
num_epochs = 25
learning_rate = 0.001
#############################################################################



# List all well folders to include in the combined dataset
#############################  TRAINING DATASET  #############################
train_well_folders = [
    'Alexa488_Fibroblasts_well6_135locations',
    'Alexa488_Fibroblasts_well5_225locations',
    'Alexa488_Fibroblasts_well3_200locations',
    'Alexa488_Fibroblasts_well4_225locations',
    'Alexa488_Fibroblasts_well7_135locations' # All the others
]

# Load the training dataset and DataLoader
train_dataset = CellDataset(
    well_folders=train_well_folders,
    brightfield_dir=BRIGHTFIELD_DIR,
    mask_dir=MASK_DIR
)

print(f"Number of training samples: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
################################################################################


#############################  VALIDATION DATASET  #############################
val_well_folders = [
    'Alexa488_Fibroblasts_well2_200locations' # Well 2
]

# Load the validation dataset and DataLoader
val_dataset = CellDataset(
    well_folders=val_well_folders,  # Make sure to pass the well_folders as a list
    brightfield_dir=BRIGHTFIELD_DIR,
    mask_dir=MASK_DIR
)
print(f"Number of validation samples: {len(val_dataset)}")

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
################################################################################


#############################  TEST DATASET  #############################
test_well_folders = [
    'Alexa488_Fibroblasts_well1_50locations' # Well 1 
]

# Load the testing dataset and DataLoader
test_dataset = CellDataset(
    well_folders=test_well_folders,  # Make sure to pass the well_folders as a list
    brightfield_dir=BRIGHTFIELD_DIR,
    mask_dir=MASK_DIR
)
print(f"Number of testing samples: {len(test_dataset)}")

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
################################################################################


# Print a sample batch shape
for images, masks in train_loader:
    print(f"Images Shape: {images.shape}")  # Expected: (batch_size, 11, H, W)
    print(f"Masks Shape: {masks.shape}")    # Expected: (batch_size, 1, H, W)
    break

num_gpus = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")


############################## Initialize model, loss function, and optimizer #############################
model = EnhancedCNN(input_channels=11)
model.to(device)
#criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for segmentation
GDL = GeneralizedDiceLoss(include_background=False,to_onehot_y=True,softmax=False)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


print("------------start training-----------")

avg_train_losses = []
avg_val_losses = [] 
train_losses = []
val_losses = []
epoch_losses =[]
val_epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for inputs, masks in train_loader:
        # Move data to device (e.g., GPU if available)
        inputs = inputs.to(device)
        masks = masks.to(device)
        
        # Forward pass
        segmentation_output = model(inputs)

        ## Calculate segmentation loss
        seg_loss = GDL(segmentation_output, masks) ## Used GDL
        #seg_loss = criterion(segmentation_output, masks) ## Used BCE

        # Track losses
        train_losses.append(seg_loss.item())
        epoch_loss += seg_loss.item()
        batch_count += 1

        # Backward pass and optimization
        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()

    scheduler.step()
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
            # Forward pass
            val_seg_output = model(val_inputs)

            # Calculate segmentation loss
            val_seg_loss = GDL(val_seg_output, val_masks) ## Used GDL
            #val_seg_loss = criterion(val_seg_output, val_masks) ## Used BCE

            val_losses.append(val_seg_loss.item())
            val_epoch_loss += val_seg_loss.item()
            val_batch_count += 1

    val_epoch_loss = val_epoch_loss / val_batch_count
    val_epoch_losses.append(val_epoch_loss)
    print("Epoch: {},  Val Loss: {}".format(epoch, val_epoch_loss))

print("------------finished training-----------")

save_loss_plot(epoch_losses,val_epoch_losses,num_epochs)

print("-----------now testing-----------")
# Testing
model.eval()
test_loss = 0.0

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


with torch.no_grad():
    test_loss = 0.0
    iou_test = 0
    for idx, (images, masks) in enumerate(test_loader):
        # Move images and masks to the device (GPU/CPU)
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        segmentation_output = model(images)

        # Calculate loss
        seg_loss = GDL(segmentation_output, masks) # Used GDL
        #seg_loss = criterion(segmentation_output, masks) ## Used BCE
        test_loss += seg_loss.item()
        
        # Compute IoU for the batch
        batch_iou = compute_iou(segmentation_output, masks)
        iou_test += batch_iou.item()
        # Generate binary predictions (thresholding at 0.5)
        preds = (segmentation_output > 0.6).float().cpu().detach()  # Moved tensors to CPU here

        # Save each prediction
        for i in range(preds.shape[0]):
            plt.figure(figsize=(8, 8))
            plt.imshow(preds[i][0].numpy(), cmap='gray')
            plt.axis('off')

            # Generate a unique filename for each prediction
            prediction_path = os.path.join(PREDICTIONS_DIR, f"prediction_{idx * preds.shape[0] + i}.png")
            plt.savefig(prediction_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

print("-----------finished testing-----------")

############################# Compute average test loss #############################
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
average_iou = iou_test / len(test_loader)
print(f"IoU: {average_iou}")
print(f"Predictions saved in '{PREDICTIONS_DIR}' folder.")

