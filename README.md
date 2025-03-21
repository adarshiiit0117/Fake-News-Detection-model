# Fake News Detection

## Overview
This project aims to detect fake news using machine learning techniques. The model analyzes text features and applies classification algorithms to distinguish between real and fake news. The project was developed as part of the SERVE Smart Hackathon at **IIT BHU**.

## Features
- **Preprocessing:** Tokenization, stop-word removal, and text vectorization using **TF-IDF**.
- **Feature Engineering:** Includes title length, text length, keyword density, sentiment analysis, and readability.
- **Model Training:** 
  - **Machine Learning Model:** Random Forest Classifier
  - **Deep Learning Model:** Sequential Neural Network with Dense layers
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Methodology
### 1. Data Preprocessing
- Tokenization and removal of stop words
- Text vectorization using **TF-IDF**
- Sentiment analysis for emotional bias detection
- Feature extraction:
  - **Title Length:** Detect exaggerated headlines
  - **Text Length:** Analyze verbosity and content reliability
  - **Keyword Density:** Identify frequent use of sensational keywords
  - **Linguistic Features:** Grammar quality, punctuation, readability

### 2. Model Training
#### **Random Forest Classifier**
- **Dataset Split:** 80% training, 20% testing
- **Hyperparameters:**
  - Number of Trees: **100**
  - Max Depth: Tuned using cross-validation

#### **Deep Learning Model**
- **Architecture:** Sequential model with Dense layers
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Epochs:** 20
- **Batch Size:** 32

### 3. Model Evaluation
- **Accuracy:** **97.43%**
- **AUC-ROC Score:** **0.986**
- **Precision, Recall, F1-Score:** Evaluated for model performance

## Results
- The **Random Forest Classifier** was highly effective in fake news detection.
- Deep learning trials showed promising results but required more computational power.
- Feature engineering significantly improved model performance.

## Future Scope
- Expand dataset diversity to improve model generalizability.
- Experiment with advanced NLP models like **BERT** for better contextual understanding.
- Implement real-time fake news detection capabilities.

## Installation & Usage
### 1. Clone Repository
```bash
 git clone https://github.com/your-repo/fake-news-detection.git
 cd fake-news-detection
```
### 2. Install Dependencies
```bash
 pip install -r requirements.txt
```
### 3. Train and Evaluate Model
```bash
 python train.py
```
### 4. Run Predictions
```bash
 python predict.py --text "Your news headline here"
```

## Contributors
- **Adarsh Dubey**
- **Ansh Singh**
- **Aditya Karn**
  
**Developed for SERVE Smart Hackathon - IIT BHU** 🚀
