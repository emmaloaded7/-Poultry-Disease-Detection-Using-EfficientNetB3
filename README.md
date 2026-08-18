# Poultry Disease Detection Using EfficientNetB3

## Overview

This project develops a deep learning image classification model for the automated detection of 
poultry diseases using **EfficientNetB3** and transfer learning.

The model classifies poultry images into four categories:

* **Coccidiosis**
* **Healthy**
* **New Castle Disease**
* **Salmonella**

The project compares an initial EfficientNetB3 transfer-learning model with a fine-tuned version using 
test-set performance metrics including accuracy, precision, recall, F1-score, and classification reports.

---

## Objectives

The main objectives of this project were to:

1. Develop an image-based poultry disease classification model.
2. Apply transfer learning using the pretrained EfficientNetB3 architecture.
3. Compare the performance of an initial model and a fine-tuned model.
4. Evaluate classification performance using multiple metrics.
5. Demonstrate prediction on previously unseen poultry images.

---

## Dataset

The project uses the **Poultry Diseases** image dataset available on Kaggle.

The dataset contains images belonging to four classes:

| Class              | Description                           |
| ------------------ | ------------------------------------- |
| Coccidiosis        | Poultry affected by coccidiosis       |
| Healthy            | Healthy poultry                       |
| New Castle Disease | Poultry affected by Newcastle disease |
| Salmonella         | Poultry affected by salmonellosis     |

For the training experiment, **5,000 images were randomly selected from each class**, resulting in a balanced
training subset of **20,000 images**.

The original validation and test datasets were retained separately.

### Dataset Source

The dataset was obtained from Kaggle:

**Poultry Diseases Dataset — chandrashekarnatesh/poultry-diseases**

Users should obtain the dataset using their own Kaggle account and API credentials.

> **Security note:** `kaggle.json` is not included in this repository because it contains private API credentials.

---

## Model Architecture

The project uses **EfficientNetB3 pretrained on ImageNet** as the feature extraction backbone.

### Architecture

```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
EfficientNetB3
(ImageNet pretrained)
    │
    ▼
Global Max Pooling
    │
    ▼
Batch Normalization
    │
    ▼
Dense Layer
256 neurons
ReLU activation
    │
    ▼
Dropout
0.40
    │
    ▼
Dense Layer
4 neurons
Softmax activation
    │
    ▼
Disease Classification
```

The initial transfer-learning model uses a frozen EfficientNetB3 backbone, followed by a custom classification head.

A second model was fine-tuned to improve classification performance.

---

## Image Processing

Images were resized to:

```text
224 × 224 pixels
```

A batch size of:

```text
32
```

was used.

### Training augmentation

The training images were augmented using:

* Horizontal flipping
* Random rotation up to 10 degrees
* Zoom augmentation up to 10%

Validation and test images were not augmented.

---

## Training Configuration

The classifier uses:

* **Optimizer:** Adam
* **Initial learning rate:** 0.001
* **Loss function:** Categorical cross-entropy
* **Output activation:** Softmax
* **Number of classes:** 4
* **Dropout:** 0.40
* **Dense layer:** 256 neurons
* **Backbone:** EfficientNetB3
* **Pretrained weights:** ImageNet

---

## Evaluation

The models are evaluated on the independent test dataset using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report

Macro-averaged precision, recall, and F1-score are also calculated to provide an overall assessment across the four disease classes.

The project also compares the initial and fine-tuned EfficientNetB3 models using visual performance comparisons.

---

## Prediction

The trained fine-tuned model can be used to classify an individual poultry image.

For each uploaded image, the model returns:

* Predicted disease class
* Prediction confidence
* Probability for each of the four classes

Example:

```text
Prediction: Coccidiosis
Confidence: 94.32%

Coccidiosis: 94.32%
Healthy: 1.21%
New Castle Disease: 2.84%
Salmonella: 1.63%
```

The values above are illustrative examples only.

---

## Project Structure

```text
Poultry-Disease-Detection/
│
├── poultry_disease_detection.ipynb
├── class_names.json
├── requirements.txt
├── README.md
│
└── results/
    ├── accuracy_comparison.png
    ├── loss_comparison.png
    └── precision_recall_f1_comparison.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Poultry-Disease-Detection.git
cd Poultry-Disease-Detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

The project was developed using Google Colab.

Open:

```text
poultry_disease_detection.ipynb
```

in Google Colab or Jupyter Notebook.

Before running the dataset download section, configure your own Kaggle API credentials.

The trained model files are not included in the repository because of their relatively large size. 
The notebook can be adapted to load locally stored model files or models hosted separately.

---

## Supporting Files

### `class_names.json`

This file stores the class labels used by the model:

```json
[
    "Coccidiosis",
    "Healthy",
    "New Castle Disease",
    "Salmonella"
]
```

Keeping the class mapping separately helps maintain consistency between model outputs and disease names during inference.

---

## Results

The project compares the performance of the initial EfficientNetB3 model with the fine-tuned model.

The evaluation includes:

* Test accuracy
* Test loss
* Macro precision
* Macro recall
* Macro F1-score
* Per-class precision, recall, and F1-score

The numerical results reported in the repository should correspond directly to the outputs generated by the notebook.

### Visual Results

The repository includes performance comparison plots:

* `accuracy_comparison.png`
* `loss_comparison.png`
* `precision_recall_f1_comparison.png`

---

## Reproducibility

Random seeds were set during the experiment to improve reproducibility:

```python
random.seed(42)
np.random.seed(0)
tf.random.set_seed(0)
```

However, exact reproduction of deep learning results may vary depending on the TensorFlow version, hardware,
GPU configuration, and other environmental factors.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB3
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Pillow
* Google Colab
* Kaggle

---

## Limitations

This project is intended as a machine learning research and demonstration project.

Model predictions should **not be treated as a replacement for veterinary diagnosis**. 
Performance may vary when images differ substantially from those represented in the training and test datasets.

Additional validation using images from different farms, geographical locations, poultry breeds, 
camera conditions, and disease stages would be required before considering practical deployment.

---

## Future Improvements

Potential improvements include:

* Training on the complete available dataset.
* Further hyperparameter optimization.
* More extensive fine-tuning of EfficientNetB3.
* Evaluation on external datasets.
* Cross-validation.
* Addition of confusion matrix analysis.
* Testing other architectures such as EfficientNetB0, EfficientNetB4, ResNet, and DenseNet.
* Model explainability using Grad-CAM.
* Collection of field images from real poultry farms.
* Development of a lightweight deployment model for mobile or edge devices.

---

## Author

**Ometoro Emmanuel**

Animal Production | Data Science | Machine Learning | Agricultural AI

This project demonstrates the application of deep learning and computer vision to agricultural and poultry health problems.

---

## License

This project is intended for educational and research purposes. Please check the original dataset's licensing terms before
redistributing dataset images or derivative materials.
