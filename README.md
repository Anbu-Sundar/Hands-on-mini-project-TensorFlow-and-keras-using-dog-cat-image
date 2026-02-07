# Hands-on-mini-project-TensorFlow-and-keras-using-dog-cat-image
Build a simple model to classify images of cats and dogs with predicting new images

Project Overview
This project implements image classification on the Cats vs Dogs dataset using TensorFlow and Keras.
It leverages transfer learning with MobileNetV2 as the base model, combined with data augmentation, fine-tuning, and callbacks to achieve high accuracy.

Features
- Preprocessing pipeline with image resizing and normalization
- Data augmentation (flip, rotation, zoom, contrast)
- Transfer learning using MobileNetV2 pretrained on ImageNet
- Early stopping and learning rate scheduling for efficient training
- Fine-tuning step to boost accuracy further
- Model evaluation and saving for reuse

Requirements
- Python 3.8+
- TensorFlow 2.x
- TensorFlow Datasets
Install dependencies
pip install tensorflow tensorflow-datasets


Optional (for GPU acceleration):
pip install tensorflow-gpu



Dataset
- Dataset: Cats vs Dogs (from TensorFlow Datasets)
- Automatically downloaded when running the script:
(ds_train, ds_test), ds_info = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)



Usage
- Run the script in Google Colab or VS Code with Python.
- Training will proceed in two phases:
- Initial training with frozen MobileNetV2 base
- Fine-tuning with base model unfrozen
- After training, the model is saved as:
  cat_dog_model_boosted.h5



Evaluation
- Metrics: Accuracy
- Output:
- Final Accuracy: 95.20%
- Model saved as cat_dog_model_boosted.h5



Notes
- Adjust IMG_SIZE and BATCH_SIZE for different hardware setups.
- Fine-tuning learning rate is set lower (1e-5) to avoid overfitting.
- You can extend evaluation with confusion matrix, ROC-AUC, or PR-AUC if needed.
