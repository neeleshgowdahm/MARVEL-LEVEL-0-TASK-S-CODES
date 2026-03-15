import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# 1. Extract ZIP Dataset
# -----------------------------
zip_file = "archive.zip"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("Files in dataset:", os.listdir("dataset"))


# -----------------------------
# 2. Load Dataset
# -----------------------------
df = pd.read_csv("dataset/housing.csv")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)


# -----------------------------
# 3. Split Features and Target
# -----------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]


# -----------------------------
# 4. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 5. Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# 6. Linear Regression From Scratch
# -----------------------------
class LinearRegressionScratch:

    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, y):

        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        y = np.array(y)

        for i in range(self.iterations):

            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# -----------------------------
# 7. Train Custom Model
# -----------------------------
custom_model = LinearRegressionScratch()

custom_model.fit(X_train, y_train)

y_pred_custom = custom_model.predict(X_test)


# -----------------------------
# 8. Train Sklearn Model
# -----------------------------
sk_model = LinearRegression()

sk_model.fit(X_train, y_train)

y_pred_sklearn = sk_model.predict(X_test)


# -----------------------------
# 9. Performance Metrics
# -----------------------------
print("\nCustom Linear Regression")

print("MSE:", mean_squared_error(y_test, y_pred_custom))
print("MAE:", mean_absolute_error(y_test, y_pred_custom))
print("R2:", r2_score(y_test, y_pred_custom))


print("\nSklearn Linear Regression")

print("MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("MAE:", mean_absolute_error(y_test, y_pred_sklearn))
print("R2:", r2_score(y_test, y_pred_sklearn))


# -----------------------------
# 10. Model Comparison Graph
# -----------------------------
plt.figure()

plt.scatter(y_test, y_pred_custom, alpha=0.5, label="Custom Model")
plt.scatter(y_test, y_pred_sklearn, alpha=0.5, color="green", label="Sklearn Model")

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression Model Comparison")

plt.legend()

plt.show()


# -----------------------------
# 11. Proper Linear Regression Line
# -----------------------------

feature = "median_income"

X_single = df[[feature]]
y_single = df["median_house_value"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single, y_single, test_size=0.2, random_state=42
)

line_model = LinearRegression()
line_model.fit(X_train_s, y_train_s)

y_line_pred = line_model.predict(X_test_s)

# sort for smooth line
sorted_idx = np.argsort(X_test_s.values.flatten())

X_sorted = X_test_s.values.flatten()[sorted_idx]
y_sorted = y_line_pred[sorted_idx]

plt.figure()

plt.scatter(X_test_s, y_test_s, alpha=0.5, label="Data Points")

plt.plot(X_sorted, y_sorted, color="red", linewidth=3, label="Best Fit Line")

plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Linear Regression Best Fit Line")

plt.legend()

plt.show()