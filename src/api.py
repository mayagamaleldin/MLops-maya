import os
import pickle

import litserve as ls
import pandas as pd
from pydantic import ValidationError


class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        with open("/teamspace/studios/this_studio/MLops-maya/titanic_model.pkl", "rb") as pkl:
            self._model = pickle.load(pkl)
        with open("/teamspace/studios/this_studio/MLops-maya/pipline_feature_trans.pkl", "rb") as pkl:
            self._prep = pickle.load(pkl)
            

    def decode_request(self, request):
        try:
            print(request)
            columns = request['columns']
            rows = request['rows']
            df = pd.DataFrame(rows, columns=columns)
            print(df)
            return df
        except Exception:
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(self._prep.transform(x))
        else:
            return None

    def encode_response(self, output):
        print(output, 9 * "*")
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist(),
        }
        return response