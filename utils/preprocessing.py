import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df):
    """Simple utility to normalize features except the label."""
    scaler = StandardScaler()
    features = df.drop(columns=['label'])
    label = df['label']
    
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled['label'] = label.values
    return df_scaled