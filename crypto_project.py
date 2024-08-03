import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch as th
from sklearn.preprocessing import LabelEncoder



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

is_null=df.isnull().sum()
numerical_dtypes=[int,float]
categorical_dtypes=[object,bool]

for column in df.columns:
  if is_null[column]>=.3*len(df):
    df.drop(column, axis=1, inplace=True)
  elif is_null[column]>0:
    if df[column].dtypes in categorical_dtypes:
      df[column].fillna(df[column].mode()[0],inplace=True)
    elif df[column].dtypes in numerical_dtypes:
      df[column].fillna(df[column].median(),inplace=True)

df.isnull().sum()

predicted_column=df.columns[-1]
numerical_columns = [col for col in df.columns if df[col].dtype in numerical_dtypes and col!=predicted_column]
categorical_columns = [col for col in df.columns if df[col].dtype in categorical_dtypes]


le = LabelEncoder()  # Create a single LabelEncoder instance

encoded_columns = {}  # Dictionary to store encoded data

for column in categorical_columns:
    column_list = df[column]
    print(f"Encoding column: {column}")  # Informative message

    encoded_column = le.fit_transform(column_list)
    encoded_columns[column] = encoded_column


# Assuming we have encoded_columns

for column in encoded_columns:
    df[column] = encoded_columns[column]

print(df)  # View the DataFrame with encoded values


tensor = th.tensor(df.values)


def homomorphic_encrypt(data):
    # Placeholder for actual homomorphic encryption logic
    modified_data=data*4
    modified_data=modified_data+9
    modified_data=modified_data*7
    return modified_data

def homomorphic_decrypt(encrypted_data):
    # Placeholder for actual homomorphic decryption logic
    modified_data=encrypted_data/4
    modified_data=modified_data-9
    modified_data=modified_data/7
    return modified_data  # Simple non-secure illustration

# Create data
data = th.tensor(df.values)
print(data)
# Simulate homomorphic encryption (not secure!)
encrypted_data = homomorphic_encrypt(data)

# Perform homomorphic addition (not secure!)
result = encrypted_data
print(result)

# Simulate homomorphic decryption (not secure!)
decrypted_result = homomorphic_decrypt(result)

print(decrypted_result)

df_encrypt=pd.DataFrame(result.numpy(),columns=df.columns)
df_encrypt

column1=df.columns[0]
column2=df.columns[-1]

# Training a model using the original dataframe
X_train, X_test, y_train, y_test = train_test_split(df[column1], df[column2], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

X_train_df = X_train.to_frame(name='independent_variable')  # Assuming feature name
y_train_df = y_train.to_frame(name='dependent_variable')  # Assuming target name
X_test_df = X_test.to_frame(name='independent_variable')  # Assuming feature name
y_test_df = y_test.to_frame(name='dependent_variable')  # Assuming target name

model.fit(X_train_df, y_train_df)

y_pred = model.predict(X_test_df)
mse = mean_squared_error(y_test_df, y_pred)
r2 = r2_score(y_test_df, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Training a model using the encrypted dataframe

X_train, X_test, y_train, y_test = train_test_split(df_encrypt[column1], df_encrypt[column2], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

X_train_df = X_train.to_frame(name='independent_variable')  # Assuming feature name
y_train_df = y_train.to_frame(name='dependent_variable')  # Assuming target name
X_test_df = X_test.to_frame(name='independent_variable')  # Assuming feature name
y_test_df = y_test.to_frame(name='dependent_variable')  # Assuming target name

model.fit(X_train_df, y_train_df)

y_pred = model.predict(X_test_df)
mse = mean_squared_error(y_test_df, y_pred)
r2 = r2_score(y_test_df, y_pred)
print("Mean Squared Error of Encrypted Model:", mse)
print("R-squared of Encrypted Model:", r2)

