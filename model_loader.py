import joblib
import os


class ModelLoader(object):
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)

    def predict(self, input):
        prediction_value = self.clf.predict(input)[0]
        probability = self.clf.predict_proba(input)[0]
        return '{"prediction":' + str(prediction_value) + ', "probability":' + str(probability[0]) + "}"
