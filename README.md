#  AI-Based Multi-Crop Disease Detection with End-to-End MLOps

An AI-powered web application that detects plant diseases from uploaded images using **MobileNetV2, TensorFlow/Keras, and Flask**. The application predicts the plant category and displays the prediction confidence score.

<img width="956" height="506" alt="crop" src="https://github.com/user-attachments/assets/1226d0e0-2334-465c-bf21-a37803e2745d" />


<img width="886" height="479" alt="crop2" src="https://github.com/user-attachments/assets/62d1983f-084d-4fe1-a1fb-aa8cb79e25e9" />


<img width="908" height="458" alt="crop3" src="https://github.com/user-attachments/assets/9b3fffc0-3454-4fd4-af95-a5d7a7fbd85a" />



##  Project Overview

Plant diseases can affect crop health and agricultural productivity. This project uses deep learning to classify cotton and corn/maize images into different disease and healthy categories.

The project demonstrates the machine learning lifecycle, including dataset preparation, preprocessing, model training, evaluation, and web application deployment.

##  Features

-  Image-based plant disease classification
-  MobileNetV2 transfer learning
-  Image preprocessing and data augmentation
-  Model evaluation using accuracy, precision, recall, and F1-score
-  Flask backend for model inference
-  User-friendly web interface
-  JSON-based prediction API

##  Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Pre-trained Model | MobileNetV2 |
| Image Processing | Pillow, NumPy |
| Model Evaluation | Scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |

##  Dataset

The final model classifies images into six categories.

**Cotton**
1. Diseased Cotton Leaf
2. Diseased Cotton Plant
3. Fresh Cotton Leaf
4. Fresh Cotton Plant

**Corn/Maize**
5. Blight
6. Healthy

The dataset contains images organized into class-specific folders. Dataset images are not included in this repository.

<img width="933" height="506" alt="crop4" src="https://github.com/user-attachments/assets/bf264a30-18c8-47a9-9319-06dd6ad32ec4" />


### Dataset Structure

```text
dataset/
├── cotton_disease/
│   ├── diseased_cotton_leaf/
│   ├── diseased_cotton_plant/
│   ├── fresh_cotton_leaf/
│   └── fresh_cotton_plant/
└── data/
    ├── Blight/
    └── Healthy/
```

##  Data Preprocessing

The following preprocessing techniques are used:

- **Image Resizing:** Images are resized to 224 × 224 pixels.
- **Image Preprocessing:** MobileNetV2's `preprocess_input()` prepares the image data.
- **Data Augmentation:** Random horizontal flipping, rotation, and zoom help the model learn variations.
- **Dataset Splitting:** 70% training, 15% validation, and 15% testing.
- **Random Seed:** 42 for reproducibility.

##  Model Development

The project uses **MobileNetV2 with transfer learning**. The pre-trained convolutional base extracts image features, while additional layers perform classification.

### Model Architecture

1. MobileNetV2 feature extraction
2. Global Average Pooling
3. Dropout layer with a rate of 0.25
4. Dense output layer with softmax activation

### Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Sparse categorical cross-entropy
- **Batch Size:** 32
- **Maximum Epochs:** 10
- **Callbacks:** ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

##  Model Evaluation

The final six-class model achieved the following results on the held-out test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 95.22% |
| Precision | 95.29% |
| Recall | 95.22% |
| F1-score | 95.23% |

These results represent performance on the test dataset. Actual performance may vary on images collected under different real-world conditions.

##  Project Structure

```text
AI_Crop_Disease_Detection_Project/
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── dataset/
│   ├── cotton_disease/
│   └── data/
├── model/
│   ├── crop_disease_model.keras
│   ├── class_names.json
│   └── metrics.json
├── templates/
│   └── index.html
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

*Note: The dataset and trained model files may need to be generated or added separately.*

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the required packages, such as TensorFlow, Flask, NumPy, Pillow, and scikit-learn.

### 4. Prepare the Dataset

Place the images into their respective class folders following the dataset structure described above.

### 5. Train the Model

```bash
python train.py
```

This runs the training pipeline and generates the model and supporting metadata according to the training script.

### 6. Run the Web Application

After the required model files have been generated, start the Flask server:

```bash
python app.py
```

Open the application in your browser:


http://127.0.0.1:5000


Upload a supported plant image to receive a prediction and confidence score.

The `/predict` endpoint accepts supported image formats, including JPG, JPEG, and PNG, and returns the result in JSON format.

##  MLOps Workflow

The implemented workflow follows these stages:


Dataset Preparation
        ↓
Image Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Flask Deployment
        ↓
Web-Based Prediction


This project demonstrates key stages of the machine learning lifecycle. Automated monitoring, scheduled retraining, and advanced MLOps integrations are possible future enhancements and are not currently implemented.

##  Future Enhancements

- Support additional crops and plant diseases
- Expand the dataset with diverse real-world images
- Improve classification performance
- Provide disease management recommendations
- Implement automated model monitoring and retraining
- Deploy the application on a cloud platform

##  Limitations

- The model supports only the six classes used during training.
- Image quality, lighting, and background variations may affect predictions.
- Predictions should be treated as decision-support information rather than a definitive agricultural diagnosis.

## 👥 Project Team

- Rashmi Raut
- Shambhavi Krishnan
- Tanushri Sawalikar

**Developed as an academic project** to demonstrate deep learning, plant disease classification, and Flask-based deployment.
