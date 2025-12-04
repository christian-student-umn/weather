import joblib
import numpy as np

def predict(data):
    # Load the model we just created
    clf = joblib.load("model_churn.sav")
    
    # Reshape data for a single prediction
    return clf.predict(np.array(data).reshape(1, -1))