# Telecom-Churn-Prediction-Overview
Telecom Churn Prediction - README

Project Overview

This project aims to predict customer churn in the telecom industry using machine learning techniques. The objective is to identify high-value customers at risk of churn and provide actionable insights to improve customer retention strategies.

Dataset

The dataset consists of customer usage patterns over four months:

Months 6 & 7: Good phase (regular usage behavior)

Month 8: Action phase (potential churn indicators appear)

Month 9: Churn phase (customers who stopped using services)

Key Features

total_rech_amt_*: Recharge amounts for different months

total_og_mou_*: Outgoing call minutes

total_ic_mou_*: Incoming call minutes

vol_2g_mb_*, vol_3g_mb_*: Data usage (MB)

arpu_*: Average revenue per user

aon: Age on network (in days)

Project Steps

Data Preprocessing

Filter high-value customers (top 30% based on recharge amount)

Identify and tag churners

Handle missing values and remove churn-phase attributes

Normalize numerical data using StandardScaler

Feature Engineering

Aggregate recharge, call, and data usage trends

Compute feature importance for churn prediction

Handling Class Imbalance

Since churners are only 5-10% of total customers, we use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

Model Training

Logistic Regression (Baseline model)

Random Forest Classifier (Feature importance analysis)

XGBoost Classifier (High-performance model)

Model Evaluation

Confusion Matrix, Precision-Recall Curve, ROC-AUC Score

Feature Importance Analysis using SHAP values

Installation and Requirements

Prerequisites

Ensure you have the following installed:

Python 3.8+

Jupyter Notebook (optional)

Required libraries:

pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

How to Run the Project

Clone the repository:

git clone https://github.com/your-repo/telecom-churn-prediction.git
cd telecom-churn-prediction

Run the preprocessing script:

python preprocess_data.py

Train and evaluate models:

python train_model.py

Results and Insights

Top churn indicators: Drop in recharge amount, reduction in call minutes, and decline in data usage.

Random Forest performed well with an AUC score of 0.85.

Business impact: The model helps predict churn early, allowing companies to intervene with discounts or loyalty programs.

Future Improvements

Optimize hyperparameters using GridSearchCV.

Deploy model using Flask API or FastAPI.

Implement real-time churn monitoring system.

Contributing

Contributions are welcome! Feel free to submit pull requests.

License

This project is licensed under the MIT License.

