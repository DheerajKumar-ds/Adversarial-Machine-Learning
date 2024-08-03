import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch as th
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('heart_disease_uci.csv')
df.drop('id', axis=1, inplace=True)

# Define the new column names
new_columns = {
    'age': 'age',
    'dataset': 'place_of_study',
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol': 'serum_cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_ecg_results',
    'thalach': 'max_heart_rate_achieved',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression',
    'slope': 'st_slope',
    'ca': 'num_major_vessels',
    'thal': 'thalassemia',
    'num': 'heart_disease_presence'
}
df = df.rename(columns=new_columns)

# Handle missing values
is_null = df.isnull().sum()
numerical_dtypes = [int, float]
categorical_dtypes = [object, bool]
for column in df.columns:
    if is_null[column] >= 0.3 * len(df):
        df.drop(column, axis=1, inplace=True)
    elif is_null[column] > 0:
        if df[column].dtypes in categorical_dtypes:
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].dtypes in numerical_dtypes:
            df[column].fillna(df[column].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
encoded_columns = {}
for column in df.columns:
    if df[column].dtype in categorical_dtypes:
        encoded_column = le.fit_transform(df[column])
        encoded_columns[column] = encoded_column

for column in encoded_columns:
    df[column] = encoded_columns[column]

# Split the dataset into features and target
X = df.drop('heart_disease_presence', axis=1)
y = df['heart_disease_presence']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Original model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Original Model - Mean Squared Error:", mse)
print("Original Model - R-squared:", r2)

# Adversarial perturbations
def generate_adversarial_examples(model, X_test, y_test):
    epsilon = 0.1  # Perturbation strength
    X_test_perturbed = X_test + epsilon * np.sign(model.coef_)
    return X_test_perturbed

X_test_perturbed = generate_adversarial_examples(model, X_test, y_test)
print(X_test_perturbed)

# Evaluate the model on perturbed data
y_pred_perturbed = model.predict(X_test_perturbed)
mse_perturbed = mean_squared_error(y_test, y_pred_perturbed)
r2_perturbed = r2_score(y_test, y_pred_perturbed)
print("Model with Adversarial Perturbations - Mean Squared Error:", mse_perturbed)
print("Model with Adversarial Perturbations - R-squared:", r2_perturbed)

# Adversarial Training
X_adv_train = generate_adversarial_examples(model, X_train, y_train)
model.fit(X_adv_train, y_train)

# Evaluate the adversarially trained model
y_pred_adv_train = model.predict(X_test)
mse_adv_train = mean_squared_error(y_test, y_pred_adv_train)
r2_adv_train = r2_score(y_test, y_pred_adv_train)
print("Adversarially Trained Model - Mean Squared Error:", mse_adv_train)
print("Adversarially Trained Model - R-squared:", r2_adv_train)
