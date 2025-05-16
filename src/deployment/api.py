import numpy as np
import litserve as ls
import pickle

from src.deployment.online.requests import InferenceRequest

class InferenceAPI(ls.LitAPI):
    def setup(self, device = "cpu"):
        with open("models/fake/final_model.pkl", "rb") as pkl:
            self._model = pickle.load(pkl)
        with open("models/fake/model_target_translator.pkl", "rb") as pkl:
            self._encoder = pickle.load(pkl)

    def decode_request(self, request):
        try:
            InferenceRequest(**request["input"])
            data = [val for val in request["input"].values()]
            x = np.asarray(data)
            x = np.expand_dims(x, 0)
            return x
        except:
            return None

    def predict(self, x):
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        return {
            "message": message,
            "prediction": [self._encoder['decoder'][val] for val in output]
        }