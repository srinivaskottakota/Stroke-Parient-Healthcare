import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Ridge Regression class
class RidgeRegression:
    def __init__(self, lr=0.01, n_iters=1000, alpha=0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha  # Regularization strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Compute predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y)) + (self.alpha / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Data preparation
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1XyhVIZaKYZczlM2alun_fofilqTBq_9c")
df = df.dropna()

# Encode categorical columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col, le in label_encoders.items():
    df[col] = le.fit_transform(df[col])

# Feature and target variable separation
X = df.drop(columns='stroke').values  # Features
y = df['stroke'].values  # Target variable

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge Regression model
ridge_model = RidgeRegression(lr=0.01, n_iters=1000, alpha=0.5)
ridge_model.fit(X_train_scaled, y_train)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(ridge_model, file)

print("Model trained and saved successfully!")
