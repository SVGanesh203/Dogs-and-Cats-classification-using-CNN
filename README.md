# 🐱🐶 Cats & Dogs Image Classifier using CNN (Transfer Learning)

A deep learning project that classifies **37 pet breeds** (cats and dogs) using **MobileNetV2** transfer learning on the **Oxford-IIIT Pet Dataset**.

---

## 📌 Project Overview

This project builds an image classification pipeline that:
- Loads and preprocesses the Oxford-IIIT Pet Dataset
- Applies data augmentation to improve generalization
- Uses **MobileNetV2** (pretrained on ImageNet) as the backbone
- Fine-tunes the model for higher accuracy
- Evaluates performance using accuracy, loss curves, confusion matrix, and classification report
- Accepts custom images and predicts the pet breed with confidence score

---

## 📂 Dataset

**Oxford-IIIT Pet Dataset** loaded via `tensorflow_datasets`

| Property        | Details                        |
|----------------|-------------------------------|
| Number of Classes | 37 pet breeds               |
| Species        | Cats & Dogs                   |
| Label Type     | Integer (0–36)                |
| Image Quality  | High quality, labeled         |
| Balance        | Balanced across breeds        |

---

## 🏗️ Model Architecture

```
Input Image (224×224×3)
        ↓
Data Augmentation (Flip, Rotation, Zoom)
        ↓
MobileNetV2 (Pretrained on ImageNet, frozen)
        ↓
Global Average Pooling 2D
        ↓
Dropout (0.3)
        ↓
Dense (128, ReLU)
        ↓
Dense (37, Softmax)
        ↓
Breed Prediction
```

**Why MobileNetV2?**
- Lightweight and efficient
- High accuracy on image tasks
- Pretrained on ImageNet (1.2M images)
- Optimized for 224×224 input images

---

## ⚙️ Training Configuration

| Parameter         | Value                          |
|------------------|-------------------------------|
| Image Size        | 224 × 224                     |
| Batch Size        | 32                            |
| Epochs            | 10 (with early stopping)      |
| Optimizer         | Adam (lr = 0.0001)            |
| Loss Function     | Sparse Categorical Crossentropy |
| Metric            | Accuracy                      |
| Early Stopping    | Patience = 3 (monitor val_loss) |
| LR Reduction      | Factor 0.2, Patience = 2      |

---

## 🔧 Fine-Tuning

After initial training, the last **20 layers** of MobileNetV2 are unfrozen and the model is fine-tuned with a lower learning rate (`1e-5`) for 5 additional epochs to boost accuracy.

---

## 📊 Evaluation

The model is evaluated using:
- ✅ **Accuracy & Loss curves** (train vs. validation)
- ✅ **Confusion Matrix** (heatmap via Seaborn)
- ✅ **Classification Report** (precision, recall, F1-score per breed)

---

## 🖼️ Custom Image Prediction

The `predict_image()` function accepts any image path and returns:
- **Breed name** (from 37 classes)
- **Species** (Cat or Dog)
- **Confidence score** (%)

```python
predict_image("your_pet_image.jpg", model)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn scikit-learn
```

### Run in Google Colab

1. Open the notebook `Cats_and_Dogs_Image_using_CNN.ipynb` in Google Colab
2. Run all cells sequentially
3. Upload your own pet image when prompted to test predictions

### Save the Model

```python
model.save("pet_breed_classifier.h5")
```

---

## 🛠️ Tech Stack

| Tool / Library         | Purpose                        |
|-----------------------|-------------------------------|
| TensorFlow / Keras    | Model building & training     |
| TensorFlow Datasets   | Dataset loading               |
| MobileNetV2           | Pretrained CNN backbone       |
| NumPy                 | Numerical operations          |
| Matplotlib            | Plotting accuracy/loss curves |
| Seaborn               | Confusion matrix heatmap      |
| Scikit-learn          | Classification report         |

---

## 📁 Project Structure

```
├── Cats_and_Dogs_Image_using_CNN.ipynb   # Main notebook
├── pet_breed_classifier.h5               # Saved model (after training)
└── README.md                             # Project documentation
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) by Visual Geometry Group
- [MobileNetV2](https://arxiv.org/abs/1801.04381) by Google
- [TensorFlow](https://www.tensorflow.org/) and [TensorFlow Datasets](https://www.tensorflow.org/datasets)
