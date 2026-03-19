# 🪙 Gold Price Prediction using ANN

ANN-based regression model for predicting gold prices using historical financial data. The project includes data preprocessing, feature scaling, model building with TensorFlow/Keras, and performance evaluation. Demonstrates how deep learning can capture complex market patterns for accurate price prediction.

This project focuses on predicting gold prices using an **Artificial Neural Network (ANN)** regression model. Gold price movements are influenced by multiple economic and financial factors, making them highly non-linear and complex. Traditional regression models often fail to capture these patterns effectively.

In this project, a deep learning approach is applied to model these complexities. The ANN is trained on historical financial data to learn underlying relationships between input features and gold prices. The workflow includes data preprocessing, feature scaling, model building, training, and evaluation.

The model is implemented using **TensorFlow/Keras**, where multiple dense layers are used to improve learning capability. The performance of the model is evaluated by comparing predicted values with actual gold prices, demonstrating how well the model generalizes to unseen data.

This project highlights the practical application of neural networks in financial forecasting and serves as a strong foundation for more advanced models such as time-series forecasting using LSTM.

---

## 📌 Features

* End-to-end implementation of ANN for regression
* Gold price prediction using historical financial data
* Data preprocessing including handling missing values
* Feature scaling to improve model performance
* Deep learning model built with TensorFlow/Keras
* Model training, validation, and evaluation
* Visualization of predicted vs actual values

---

## 📊 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn
* TensorFlow / Keras

---

## 🏗️ Model Overview

* Input Layer: Based on selected features
* Hidden Layers: Fully connected dense layers with ReLU activation
* Output Layer: Single neuron for continuous value prediction

**Configuration:**

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Metrics: Mean Absolute Error (MAE)

---

## 📊 Workflow

1. Data Collection and Loading
2. Data Preprocessing and Cleaning
3. Feature Selection and Scaling
4. ANN Model Design
5. Model Training and Validation
6. Performance Evaluation

---

## 📈 Results

The ANN model successfully learns patterns in gold price data and provides accurate predictions. The comparison between actual and predicted values shows that the model captures trends effectively, although further tuning can enhance performance.

---

## 🚀 How to Run

1. Install requirements:
   pip install -r requirements.txt

2. Run the notebook:
   regression_ANN_Project.ipynb

