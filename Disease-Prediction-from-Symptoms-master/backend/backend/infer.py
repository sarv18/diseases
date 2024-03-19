import pandas as pd
import numpy as np
from joblib import load

def get_prediction(symptoms):
    # Prepare Test Data
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))

    # Load pre-trained model
    clf = load("C:\\Users\\SARVESH\\Downloads\\Disease-Prediction-from-Symptoms-master\\Disease-Prediction-from-Symptoms-master\\backend\\backend\\random_forest.joblib")  # corrected path here
    result = clf.predict(df_test)
    print(f"Predicted Disease: {result}")

    return result
