import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(object):
    def __init__(self, file_):
        data_set = pd.read_csv(file_, index_col=0)
        self.classifier = GaussianNB()
        features = np.array(data_set[["AGE"]])
        target = np.array(data_set["is_obesity"]).reshape(-1, 1).ravel()
        self.classifier.fit(features, target)

    def predict(self, value):
        return {'prediction': self.classifier.predict(np.array([value]).reshape(1, 1))[0] }
