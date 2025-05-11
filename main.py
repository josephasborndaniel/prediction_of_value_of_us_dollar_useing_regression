# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Step 1: Prepare the data
data = {
    'Year': [1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'USD_to_INR': [4.76, 4.76, 4.76, 4.76, 4.76, 4.76, 7.50, 7.50, 7.50, 7.50, 7.50, 7.50, 7.59, 7.74, 8.10, 8.39, 8.96, 8.74, 8.19, 8.13, 7.86, 8.66, 9.46, 10.10, 11.36, 12.38, 12.61, 12.96, 13.92, 16.23, 17.50, 22.74, 25.92, 30.49, 31.37, 32.43, 35.43, 36.31, 41.26, 43.06, 44.94, 47.19, 48.61, 46.58, 45.32, 44.10, 45.31, 41.35, 43.51, 48.41, 45.73, 46.67, 53.44, 56.57, 62.33, 62.97, 66.46, 67.79, 70.09, 70.39, 76.38, 74.57, 81.35, 81.94, 84.83, 86.83]
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Prepare features and labels
X = df['Year'].values.reshape(-1, 1)  # Features
y = df['USD_to_INR'].values  # Labels

# Step 4: Set up pipeline and hyperparameter tuning
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('ridge', Ridge())
])

param_grid = {
    'poly__degree': [1, 2, 3, 4, 5],  # Testing different polynomial degrees
    'ridge__alpha':[0.01, 0.1, 1, 10, 100]  # Testing different regularization strengths
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Step 5: Use the best model found
best_model = grid_search.best_estimator_

# Step 6: Predict for 2026
X_2026 = np.array([[2026]])
predicted_value_2026 = best_model.predict(X_2026)

print(f"The predicted USD to INR exchange rate for 2026 is: {predicted_value_2026[0]:.2f}")
print(f"Best Degree Chosen: {grid_search.best_params_['poly__degree']}")
print(f"Best Alpha Chosen: {grid_search.best_params_['ridge__alpha']}")

# Step 7: Visualize the data and prediction
X_full = np.arange(1960, 2027).reshape(-1, 1)  # for smooth curve from 1960 to 2026
y_full_pred = best_model.predict(X_full)

plt.scatter(X, y, color='blue', label='Historical Data')
plt.plot(X_full, y_full_pred, color='red', label='Best Model Prediction Line')
plt.scatter(2026, predicted_value_2026, color='green', label='Prediction for 2026', zorder=5)
plt.xlabel('Year')
plt.ylabel('USD to INR')
plt.title('USD to INR Prediction (with Hyperparameter Tuning)')
plt.legend()
plt.grid(True)
plt.show()
