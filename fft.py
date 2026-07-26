import os
from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Directories for the original data and the preprocessed data
BRIGHTFIELD_DIR = '/zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data/brightfield'
PREPROCESSED_DIR = '/dtu/blackhole/0b/168395/fftt'

# Create the preprocessed directory if it doesn't exist
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

def fft_filter_image(image_array):
    # Apply FFT to the image
    image_fft = fft2(image_array)
    shifted_image_fft = fftshift(image_fft)

    # Create a mask to remove high-frequency components
    rows, cols = image_array.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)

    radius = 60  # Adjust this based on your data characteristics
    for i in range(rows):
        for j in range(cols):
            if (i - center_row)**2 + (j - center_col)**2 > radius**2:
                mask[i, j] = 0

    # Apply the mask to the FFT image
    filtered_fft = shifted_image_fft * mask

    # Perform inverse FFT to return to spatial domain
    filtered_image = np.abs(ifft2(ifftshift(filtered_fft)))
    return filtered_image

def preprocess_fft_images(brightfield_dir, preprocessed_dir):
    # Iterate through well folders
    for well_folder in os.listdir(brightfield_dir):
        well_path = os.path.join(brightfield_dir, well_folder)

        # Ensure it's a directory
        if not os.path.isdir(well_path):
            continue

        # Create corresponding folder in the preprocessed directory
        preprocessed_well_path = os.path.join(preprocessed_dir, well_folder)

        # Skip processing if the folder already exists
        if os.path.exists(preprocessed_well_path):
            print(f"Skipping {well_folder}, already processed.")
            continue

        os.makedirs(preprocessed_well_path, exist_ok=True)

        # Process each image in the well folder
        for image_file in sorted(os.listdir(well_path)):
            if image_file.endswith('ORG.tif'):
                # Load the image
                image_path = os.path.join(well_path, image_file)
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)

                # Apply FFT filter
                filtered_image = fft_filter_image(img_array)

                # Save the filtered image
                filtered_image = (filtered_image / np.max(filtered_image) * 255).astype(np.uint8)  # Normalize to 0-255
                filtered_image = Image.fromarray(filtered_image)

                save_path = os.path.join(preprocessed_well_path, image_file)
                filtered_image.save(save_path)

# Run the preprocessing
preprocess_fft_images(BRIGHTFIELD_DIR, PREPROCESSED_DIR)

print(f"All FFT-filtered images have been saved to '{PREPROCESSED_DIR}'!")
