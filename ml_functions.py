# machine_learning/ml_functions.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    # Split data into features and target
    X = data.drop(columns=['target']) 
    y = data['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train machine learning model (example using RandomForestClassifier)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model

def make_predictions(model, data):
    # Make predictions using the trained model
    predictions = model.predict(data.drop(columns=['target']))
    return predictions
