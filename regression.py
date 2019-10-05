from sklearn.linear_model import LinearRegression


class Regression(object):
    def __init__(self):
        self.classifier = LinearRegression().fit(X, y)

    def predict(self, input):
        pass
