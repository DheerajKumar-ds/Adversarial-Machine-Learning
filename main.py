# main.py

from encryption_functions import encrypt_data, decrypt_data
from ml_functions import train_model, make_predictions
from phe import paillier

# Load dataset
data_path = "heart_disease_uci.csv"

# Encrypt dataset
encrypted_data = encrypt_data(data_path)

# Train machine learning model
model = train_model(encrypted_data)

# Make predictions
predictions = make_predictions(model, encrypted_data)

# Decrypt predictions
private_key = paillier.PaillierPrivateKey(2048)
decrypted_predictions = decrypt_data(predictions, private_key)

# Perform further analysis or tasks with decrypted predictions
