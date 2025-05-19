import pickle
import numpy as np
import pandas as pd
# Load the trained model (example for Random Forest)
with open("/teamspace/studios/this_studio/MLops-maya/titanic_model.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Load the preprocessor (example)
with open("/teamspace/studios/this_studio/MLops-maya/pipline_feature_trans.pkl", "rb") as f:
    processing = pickle.load(f)

# Example input data
input_data = {
'Age': 30,  'Pclass': 1,'Sex': "male",'SibSp':1,'Parch':0,'Fare':7.5,'Embarked':"S"
}
data = pd.DataFrame.from_dict(input_data)
print(data)
# Preprocess the data
x = np.array([list(input_data.values())])  # Convert to numpy array

# Apply preprocessing (if necessary)
x_processed = processing.transform(x)  # Assuming `processing` is a transformer

# Make prediction
prediction = model_rf.predict(x_processed)

print("Prediction:", prediction)
