import pickle
import pandas as pd
import litserve as ls

class ColumnTransformAPI(ls.LitAPI):
    def setup(self, device):
        # Load trained model
        with open("saved_models/model_logreg.pkl", "rb") as f:
            self.model_rf = pickle.load(f)
        
        # Load the preprocessor
        with open("pipline_feature_trans.pkl", "rb") as f:
            self.processing = pickle.load(f)

    def decode_request(self, request):
        # Expecting a dict with column names matching training data
        return pd.DataFrame([request["input"]])

    def predict(self, X):
        # Preprocess the input data
        X_processed = self.processing.transform(X)
        
        # Predict using the loaded model
        return self.model_rf.predict(X_processed).tolist()

    def encode_response(self, output):
        return {"output": output}

if __name__ == "__main__":
    api = ColumnTransformAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000)
