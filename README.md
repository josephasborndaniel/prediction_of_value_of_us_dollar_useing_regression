# USD to INR Exchange Rate Prediction

This project predicts the USD to INR exchange rate for the year 2026 using historical data from 1960 to 2025. It leverages polynomial regression with Ridge regularization and uses `GridSearchCV` for hyperparameter tuning.

## 📌 Features

- Polynomial regression with Ridge regularization.
- Hyperparameter tuning using GridSearchCV.
- Visualizes historical data and future prediction.
- Predicts the USD to INR exchange rate for the year 2026.

## 🧠 Technologies Used

- Python
- pandas
- numpy
- matplotlib
- scikit-learn

## 🚀 How It Works

1. Historical exchange rate data is loaded into a DataFrame.
2. Polynomial features are generated.
3. Ridge regression is applied.
4. A pipeline is used for seamless transformation and modeling.
5. Hyperparameters (degree of polynomial and alpha) are tuned using `GridSearchCV`.
6. The best model is selected and used to predict the exchange rate for 2026.
7. The result is visualized using matplotlib.
   
## 📦 Dependencies
Install the required Python packages:
pip install pandas numpy matplotlib scikit-learn

## ▶️ Run the Script
python usd_inr_prediction.py


## 📈 Example Output
## 📊 Visualization Output

![USD to INR Prediction](usd_inr_prediction.png.png)

