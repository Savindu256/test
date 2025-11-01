# Jute Pest Image Classification using Convolutional Neural Networks (CNN)

In this project, a **Convolutional Neural Network (CNN)** is implemented for image classification using a **custom Jute Pest dataset**. 

The process begins with environment setup and dataset preparation, where the Jute Pest dataset, containing 7,235 color images across 17 pest classes, is loaded and preprocessed. 
The dataset is organized into train, validation, and test sets. Images are resized to 128x128 pixels and normalized for efficient model training.

A CNN model is then created using TensorFlow/Keras. The architecture includes multiple convolutional and max-pooling layers followed by fully connected layers with dropout regularization to prevent overfitting. 
The model is trained for 20 epochs using the Adam optimizer and categorical crossentropy loss function, and its performance is evaluated on the testing dataset using metrics such as accuracy, precision, recall, and confusion matrix. 

Additionally, this project explores the effect of different **learning rates** (ranging from 0.1 to 0.0001) on model performance and identifies an optimal learning rate of 0.0003. 
The best model is saved for later use, and possible extensions with transfer learning are also discussed.

---

## The Jute Pest Dataset

![image](https://github.com/yourusername/Jute-Pest-CNN-Classification/assets/sample-dataset.png)

The **Jute Pest dataset** is a custom image dataset containing photos of 17 pest species that affect jute plants.  
Each pest category is stored in a separate folder, automatically labeled when loaded using TensorFlow’s `image_dataset_from_directory()`.

- **Total Images:** 7,235  
- **Training Set:** 6,443 images (89.1%)  
- **Validation Set:** 413 images (5.7%)  
- **Test Set:** 379 images (5.2%)  
- **Image Size:** 128x128 pixels  
- **Number of Classes:** 17  

This dataset enables the CNN model to recognize and classify pest species based on their visual features.

---

## Data Pre-processing

1. **Dataset Loading:**
   - The dataset is loaded using `tf.keras.utils.image_dataset_from_directory()`.
   - Training, validation, and testing sets are automatically created.

2. **Normalization:**
   - Pixel values are normalized to the [0,1] range for stable model training.

3. **Batching and Caching:**
   - Data pipelines are optimized using TensorFlow’s `cache()` and `prefetch()` for better GPU utilization.

4. **Visualization:**
   - Random images are plotted to verify dataset correctness and class balance.

---

## CNN Architecture Overview

![image](https://github.com/yourusername/Jute-Pest-CNN-Classification/assets/cnn-architecture.png)

The CNN model architecture includes three convolutional blocks and two fully connected layers. Each convolutional block consists of a Conv2D and MaxPooling2D layer pair.

1. **Convolutional Layers:**
   - Three Conv2D layers with 32, 64, and 128 filters respectively.
   - ReLU activation is applied after each convolution.
   - Each block is followed by a 2x2 MaxPooling layer for dimensionality reduction.

2. **Flattening Layer:**
   - Converts the 2D feature maps into a 1D feature vector.

3. **Fully Connected Layers:**
   - Dense(256, ReLU) + Dropout(0.5)
   - Dense(128, ReLU) + Dropout(0.5)

4. **Output Layer:**
   - Dense(17, Softmax) for multi-class classification.

### Model Parameters:
- Conv2D Layer 1: 896 parameters  
- Conv2D Layer 2: 18,496 parameters  
- Conv2D Layer 3: 73,856 parameters  
- Dense Layer 1: 6,422,784 parameters  
- Dense Layer 2: 32,896 parameters  
- Output Layer: 2,193 parameters  

**Total Parameters:** 6,551,121 (~25 MB)

---

## Model Training

### Training Configuration:

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Learning Rate:** 0.001 (optimized to 0.0003)  
- **Batch Size:** 32  
- **Epochs:** 20  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau  

### Interpretation:
- ReLU activation accelerates learning and prevents vanishing gradients.  
- Dropout layers (0.5) reduce overfitting.  
- Adam optimizer provides stable convergence with adaptive learning.  
- Validation and training metrics are tracked per epoch.

---

## Model Performance

### Initial Training (LR = 0.001)
- Training Accuracy: **53.9%**
- Validation Accuracy: **39.7%**
- Validation Loss: **2.61**

### Optimized Training (LR = 0.0003)
- Training Accuracy: **86.9%**
- Validation Accuracy: **60.5%**
- Test Accuracy: **59.6%**
- Test Loss: **1.65**

The best results were obtained at **LR = 0.0003**, balancing accuracy and generalization.

---

## Learning Rate Comparison

| Learning Rate | Train Accuracy | Validation Accuracy | Overfitting Gap |
|----------------|----------------|---------------------|-----------------|
| 0.1 | 9.9% | 17.2% | +0.07 |
| 0.01 | 10.4% | 17.1% | +0.06 |
| 0.005 | 11.8% | 18.9% | +0.07 |
| 0.001 | 63.8% | 50.8% | +0.13 |
| **0.0003** | **86.6%** | **66.1%** | **+0.20** |
| 0.0005 | 65.2% | 50.8% | +0.14 |
| 0.0001 | 10.4% | 17.1% | +0.06 |

**Conclusion:** The CNN performed best at a learning rate of **0.0003**, yielding the highest validation accuracy and smooth loss convergence.

---

## Visualization

### Training and Validation Accuracy
![image](https://github.com/yourusername/Jute-Pest-CNN-Classification/assets/accuracy-curve.png)

### Training and Validation Loss
![image](https://github.com/yourusername/Jute-Pest-CNN-Classification/assets/loss-curve.png)

---

## Model Evaluation

### Confusion Matrix
![image](https://github.com/yourusername/Jute-Pest-CNN-Classification/assets/confusion-matrix.png)

- **True Positive (TP):** Correctly identified pest species.  
- **False Positive (FP):** Incorrect pest predictions.  
- **False Negative (FN):** Missed correct pest class.  
- **True Negative (TN):** Correctly rejected wrong class.  

The confusion matrix visualizes how well the model distinguishes between different pest species.

---

## With State-of-the-Art Networks (Future Work)

- Apply **transfer learning** using pre-trained models such as **ResNet50**, **VGG16**, or **MobileNetV2** for improved feature extraction.  
- Compare performance metrics (accuracy, loss, training time) between the custom CNN and transfer learning models.  
- Evaluate improvements in generalization and inference speed.

---

## Repository Structure

