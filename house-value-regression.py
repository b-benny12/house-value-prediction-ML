
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


housing = pd.read_csv('housing.csv')

# Selecting relevant features and target variable
X = housing[['total_bedrooms', 'population', 'median_income']]
y = housing['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


total_bedrooms = float(input("Enter the total number of bedrooms sqmt: "))
population = float(input("Enter the population: "))
median_income = float(input("Enter the median income($10k): "))

# Predicting median_house_value for the user input values
input_data = np.array([[total_bedrooms, population, median_income]])
predicted_value = model.predict(input_data)

print("Predicted median_house_value:", predicted_value)


