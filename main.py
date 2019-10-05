from flask import Flask, request
from gevent.pywsgi import WSGIServer

from bayes_classifier import NaiveBayes
from model_loader import ModelLoader
import numpy as np

app = Flask(__name__)

naive = {'male': NaiveBayes("male_data_0.csv"), 'female': NaiveBayes("female_data_0.csv")}
model01 = ModelLoader("best_model_01.joblib")


@app.route('/predict', methods=['POST'])
def hello():
    print('Request {} {}'.format(request.values, request.get_json()))
    sex = request.values.get('sex')
    model = request.values.get('model')
    if not model or not sex:
        raise Exception('Param undefined!')
    if model == 'naive':
        classifier = naive[sex]
        age = request.values.get('age')
        if not age:
            raise Exception('Age param undefined!')
        print('Age: ' + age)
        prediction = str(classifier.predict(int(age)))
        print('Result: ' + prediction)
        return prediction
    if model == 'model01':
        data = ["1", "1"]
        text = "GG,GA,GG,TT,AA,AA,GG,TC,AA,GG,TT,AA,00,GG,GG,GG,GA,AG,CC,CC,TC,TC,TC,TT,TC,CC,AG,AG,GG,CC,GG,AA,CC,TG,GG,CC,TT,TC,CA,GA,GT,TC,AG,GG,AA,GG,TT,AG,CT,CT,GG,TG,TC,AG,GG,AG,TT,CC,AG,00,AA,AA,AA,GG,TT,AG,TT,CC,TC,TT,GG,TC,CC,CT,CT,AA,GA,AA,AA,GG,GG,AA,GA,GG,AA,AA,CC,AA,AA,CC,TT,CC,GG,GG,AA,AA,CC,AA,GG,AA,TT,00,GG,TT,TT,CC,AA,CC,TT,TT,TT,TT,TT,AA,CT,AG,AA,GT,AA,AA,CC,CT,TC,AA,TT,TT,GG,CC,CC,CC,AA,CC,GG,GG,AG,GG,CC,GG,AG,GA,CC,GG,CC,TT,AG,TT,GG,TT,AA,CT,TC,GG,GG,GG,AA,GG,TT,GG,CC,CC,GG,AA,TT,CC,AA,GG,CC,GG,GA,CT,TT,CT,GA,AG,GA,GG,GA,CA,GG,GG,TT,AA,TT,GT,CC,AA,AG,TC,GA,CC,AA,TT,AA,TT,GG,CC,AC,GG,TC,CT,CC,TT,GG,GA,TC,GG,AA,AC,AG,CC,GG,GG,AA,AG,AG,AA,AA,TT,CC,TT,CC,CT,CC,TC,AG,AA,AG,CC,CT,TC,GG,TG,CC,AA,GG,AA,AC,CC,GG,TT,TT,GG,CC,GA,CT,CT,CC,CC,TG,TC,AA,CA,AG,AG,GG,GA,CC,TC,GA,CC,AG,AC,GA,AA,TG,TT,CT,CC,AG,CC,GG,AA,TT,GG,TT,TT,GG,AA,AA,CC,TG,TC,GT,AG,GA,CC,GG,TT,TG,GA,CT,CC,AA,TT,TC,GG,CT,CT,GA,AG,TT,TT,AA,CC,GG,TT,TT,GG,TT,GG,GG,TC,AG,CA,AA,AA,TG,AG,CC,GG,TT,GG,GG,GG"
        data = data + text.split(",")
        model01.predict(np.array(data))
    return 'Ignored...'


if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
