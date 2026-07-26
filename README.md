# P22-Multiview-Segmentation  

This GitHub repository contains the code for our project on multifocal deep learning models for accurate cell segmentation in traction force microscopy. Please note that the dataset has not been included in this repository due to confidentiality agreements with the data providers.  

## Overview  

The project explores the development and evaluation of multiple neural networks for cell segmentation, including variations that utilize FFT (Fast Fourier Transform) filtering. The main models in this repository include:  
- `enhanced_cnn.py`: The Convolutional Neural Network (CNN) model.  
- `unet.py`: The U-Net model, designed specifically for segmentation tasks.  
- `unetwithonly1.py`: A U-Net model trained using only the 6th layer of data, which significantly reduces training time compared to using all 11 layers.  

### Data Requirements  

The models are designed to work with datasets consisting of 11 layers per mask, resulting in substantial training times. Variants of the models, such as those using FFT-filtered data, are also included for comparison.  

## Reproducing Results  

- **Trained Model Prediction**:  
  To reproduce results from the trained U-Net model, use the `TrainedModelUnet1InputPredict.ipynb` file. This notebook demonstrates the performance of the U-Net model with an example dataset and corresponding segmentation output. The actual model is too large to incluede in the repository, and was instead run from a private computer so the results are visible. 

- **Training and Testing Demo**:  
  The `ModelTrainTestPredictDemo.ipynb` notebook provides a simplified demonstration of the training, testing, and prediction processes. It uses just two images of data due to confidentiality and epochs and batch size are reduced to improve training time. The code is configured by default to work with the U-Net model. To test the CNN model instead, simply uncomment the relevant sections in the code.  

## Authors  

This project and all accompanying code were developed by:  
- Benjamin Chambaudet (s240414)  
- Frederik Lundgren (s214626)  
- Weinan Xiong (s214612)  

Confidential data for the project was provided under strict usage agreements.  

## Notes  

- All code involving large datasets was executed on a High-Performance Computing (HPC) system due to the significant computational resources required.  
- Variations of the models have been implemented to handle FFT-filtered data, enabling additional preprocessing workflows.  
