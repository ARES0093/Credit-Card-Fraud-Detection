# Credit-Card-Fraud-Detection
A machine learning project to detect fraudulent transactions using XGBoost, SMOTE for class balancing, SHAP for model explainability.


✨ Overview
This project focuses on building a machine learning model capable of detecting fraudulent credit card transactions.
It uses structured transaction data and applies advanced ML techniques to maximize fraud detection while minimizing false positives.


📚 Dataset
FraudTrain.csv
FraudTest.csv
Sourced from Kaggle Dataset


🛠️ Key Steps
Data Cleaning and Preprocessing
Feature Engineering
Handling Class Imbalance using SMOTE
Model Training using XGBoostClassifier
Model Evaluation using Precision, Recall, F1 Score, ROC-AUC
Model Explainability using SHAP visualizations


🧠 Technologies Used
Python 3.9+
Pandas, NumPy, Matplotlib, Seaborn
scikit-learn, XGBoost, imbalanced-learn
SHAP, Joblib
Streamlit (for web app deployment)


📈 Results
High ROC-AUC Score achieved
Clear separation between fraudulent and legitimate transactions
SHAP plots highlight the key transactional attributes influencing predictions



🚀 How to Run Locally
1. Clone this repository:  git clone https://github.com/ARES0093/Credit-Card-Fraud-Detection.git
2. Install dependencies:   pip install -r requirements.txt
3. Run the main script:    python main.py


📸 Visual Outputs
ROC Curve (See /images/Figure_1 Credit Card Fraud Detection.png)

SHAP Summary Plot (See /image/Figure_2 Credit Card Fraud Detection.png)
