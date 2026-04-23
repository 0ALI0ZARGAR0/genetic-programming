import pickle

import numpy as np

# Load the model and preprocessor
try:
    model = pickle.load(open('../models/gp_anomaly_detector_enhanced.pkl', 'rb'))
    preprocessor = pickle.load(open('../models/gp_anomaly_detector_enhanced_preprocessor.pkl', 'rb'))
except FileNotFoundError:
    # Fallback to simple model if enhanced model not found
    model = pickle.load(open('../models/gp_anomaly_detector.pkl', 'rb'))
    preprocessor = None  # Simple model doesn't have separate preprocessor

def detect_anomaly(features, threshold=0.5):
    '''
    Detect anomalies using the trained GP model
    
    Parameters:
    -----------
    features : dict or pandas DataFrame row
        Network traffic features
    threshold : float, default=0.5
        Probability threshold for anomaly detection
        
    Returns:
    --------
    is_anomaly : bool
        True if the sample is an anomaly, False otherwise
    probability : float
        Probability of being an anomaly
    '''
    # Preprocess the features if preprocessor is available
    if preprocessor is not None:
        X = preprocessor.transform([features])
    else:
        # For simple model, convert features to array directly
        import pandas as pd
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
            X = features_df.values
        else:
            X = [features]
    
    # Make prediction
    prob = model.predict_proba(X)[0, 1]
    is_anomaly = prob >= threshold
    
    return bool(is_anomaly), float(prob)

# Example usage:
# sample = {
#    'dur': 0.121478,
#    'proto': 'tcp',
#    ...
# }
# is_anomaly, prob = detect_anomaly(sample)
# print(f"Anomaly: {is_anomaly}, Probability: {prob:.4f}")
