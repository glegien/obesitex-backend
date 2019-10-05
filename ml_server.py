from flask import Flask, request
from gevent.pywsgi import WSGIServer

from bayes_classifier import NaiveBayes

app = Flask(__name__)

naive = {'male': NaiveBayes("male_data_0.csv"), 'female': NaiveBayes("female_data_0.csv")}


@app.route('/predict', methods=['POST'])
def hello():
    print('Request {}'.format(request.values))
    sex = request.values.get('sex')
    model = request.values.get('model')
    if not model or not sex:
        raise Exception('Param undefined!')
    if model == 'naive':
        classifier = naive[sex]
        age = request.values.get('age')
        if not age:
            raise Exception('Age param undefined!')
        print 'Age: ' + age
        prediction = str(classifier.predict(int(age)))
        print('Result: ' + prediction)
        return prediction
    return 'Ignored...'


if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
