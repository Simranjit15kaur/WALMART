# 🛒 Walmart Grocery Product Classifier & Smart Cart System

An AI-powered smart grocery cart system that uses deep learning to classify grocery items from images or live webcam input. The system fetches real-time product prices from Walmart's dataset using fuzzy matching and dynamically builds a shopping cart with total pricing.

---

## 📌 Features

- 🔍 **Image Classification**  
   Predicts grocery items (Apple, Milk, Banana, etc.) using a fine-tuned MobileNetV2 model.
- 🎥 **Live Camera Detection**  
   Allows real-time grocery item recognition using your device’s webcam.
- 🏷️ **Price Mapping**  
   Matches the predicted item with Walmart product titles using fuzzy matching (`rapidfuzz`) and fetches accurate prices.
- 🛒 **Dynamic Shopping Cart**  
   Automatically adds recognized products to a virtual cart, tracks quantity, and computes the total cost.
- 🖼️ **Colab/Image Upload Support**  
   Use Google Colab to upload images or access the camera via OpenCV.

---

## 📁 Dataset

- **Image Data**: Grocery product images organized into 43 categories.
- **Pricing Data**: Kaggle dataset – [`product-prices-and-sizes-from-walmart-grocery`](https://www.kaggle.com/datasets/thedevastator/product-prices-and-sizes-from-walmart-grocery)

---

## 🧠 Model Overview

- **Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Training Size**: 2128 training images, 512 validation images
- **Input Shape**: 224x224 RGB
- **Performance**: ~90% validation accuracy after 10 epochs
- **Model File**: `product_classifier.keras`

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Simranjit15kaur/WALMART.git
   cd WALMART
