# ME555-Project  
**Enhancing Robotic Task Prediction with Multi-Modal Inputs: Integrating Voice Commands and Mechanical Measurements**

---

## Overview  
This project focuses on enhancing robotic task prediction by integrating multi-modal inputs, including voice commands and mechanical measurements. It involves preprocessing data, applying data augmentation techniques, voice command feature extraction, and training a predictive model using 1D CNN.

---

## Code Structure  

### 1. **Data Preprocessing**  
The `data_preprocessing` module handles RH20T dataset preparation:  
- **Step 1**: Extract relevant data files (e.g., `tcp.npy`, `force_torque.npy`) from filtered task folders.  
- **Step 2**: Perform signal downsampling and low-pass filtering.  
- **Step 3**: Feature engineering to generate statistical features.

---

### 2. **Data Augmentation**  
The `data_augmentation` module enhances data robustness with the following techniques:  
- **Variational Autoencoder (VAE)**  
- **Unconditioned Gaussian Noise Augmentation**  
- **t-SNE Visualization** for data distribution analysis.

---

### 3. **Voice Command Embedding**  
The `voice_embedding` module extracts features from voice commands:  
- **Step 1**: Generate voice command features based on task number.  
- **Step 2**: Use SentenceTransformer to generate text embeddings for voice commands.  
- **Step 3**: Apply PCA to filter out non-primary components in the voice command embeddings.  
- **Step 4**: Perform t-SNE visualization to verify data distribution.

---

### 4. **1D CNN Predictive Model**  
The `1DCNN` module implements and evaluates a predictive model using TensorFlow/Keras:  
- **Step 1**: Model implementation for task prediction.  
- **Step 2**: Model training and testing on an unseen data split.  
- **Step 3**: Conduct a paired t-test for statistical significance testing.

---

## Dataset  
The dataset is available for download on the **RH20T** website.  
- **Task Descriptions**: Provided in `task_description.json`.  
- **Data Files**: Includes relevant files such as `tcp.npy` and `force_torque.npy`.

---

## Requirements  
To run the project, install the following dependencies:  
```bash
pip install tensorflow torch scikit-learn matplotlib sentence-transformers
