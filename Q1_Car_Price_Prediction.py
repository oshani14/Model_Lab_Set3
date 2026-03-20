import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("car.csv")   # You can use any sample dataset

# First 5 rows
print(df.head())

# Basic stats
print(df.describe())

# Data types
print(df.dtypes)

# Handle null values
df.fillna(df.mode().iloc[0], inplace=True)

# Heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Convert categorical to numeric
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
