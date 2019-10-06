from flask import Flask, request
from gevent.pywsgi import WSGIServer

from bayes_classifier import NaiveBayes
from model_loader import ModelLoader
from regression import Regression
import numpy as np
import pandas as pd

app = Flask(__name__)

naive = {'male': NaiveBayes("male_data_0.csv"), 'female': NaiveBayes("female_data_0.csv")}
model01 = ModelLoader("best_model_01.joblib")
model02 = Regression("female_data_0.csv")


@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/predict', methods=['POST'])
def hello():
    print('Request {} {}'.format(request.values, request.get_json()))
    sex = request.get_json()['sex']
    model = request.get_json()['model']
    if not model or not sex:
        raise Exception('Param undefined!')
    if model == 'naive':
        classifier = naive[sex]
        age = request.get_json()['age']
        if not age:
            raise Exception('Age param undefined!')
        print('Age: ' + str(age))
        prediction = str(classifier.predict(int(age)))
        print('Result: ' + prediction)
        return prediction
    else:
        # wybrac model w zaleznosci od plci
        age = request.get_json()['age']
        text = request.get_json()['genome']
        #data = {}
        #data['AGE'] = age
        columns = "rs12620338,rs7559271,rs2234675,rs6436302,rs12053273,rs1430657,rs16863576,rs7589708,rs4674639,rs10932949,rs12995399,rs9768991,rs7809325,rs17879130,rs6964358,rs4724821,rs2410612,rs3816246,rs61734430,rs2651364,rs7963401,rs2733682,rs2651374,rs7132461,rs10771951,rs4931631,rs7299495,rs10844219,rs7311935,rs7963397,rs7295095,rs10844227,rs7977101,rs7966856,rs7967302,rs2088656,rs4931635,rs904582,rs10771966,rs6488068,rs7962152,rs4135048,rs4135060,rs3751209,rs140436257,rs4135113,rs4135126,rs2888805,rs2041794,rs2908792,rs12930428,rs2160290,rs4784311,rs13332406,rs76818213,rs1131220,rs3809634,rs3095631,rs17194040,rs1861556,rs16952304,rs7193898,rs1362572,rs12599436,rs1946155,rs4784320,rs12443767,rs3213758,rs17214955,rs8050354,rs139974543,rs2111119,rs2302677,rs9934800,rs5005161,rs7205986,rs1421084,rs7203521,rs6499640,rs4396532,rs1861868,rs1075440,rs13334933,rs9930333,rs9939973,rs9940128,rs1421085,rs16952520,rs1558902,rs10852521,rs1121980,rs7193144,rs17817449,rs11075987,rs8050136,rs9935401,rs9936385,rs9926289,rs76804286,rs9939609,rs9941349,rs7190492,rs9930506,rs9922708,rs9922619,rs8044769,rs12149832,rs10852523,rs3826169,rs10521307,rs17819033,rs7205009,rs2160481,rs4784329,rs7191718,rs9934504,rs9929152,rs12232391,rs9924072,rs12933996,rs17224310,rs17823199,rs7194907,rs6499662,rs12596210,rs8046658,rs7200972,rs9925908,rs12931859,rs7194243,rs4784351,rs2540781,rs856973,rs2003583,rs16953002,rs708258,rs1008400,rs11646512,rs11863548,rs2665271,rs2689264,rs8053279,rs8063722,rs879679,rs1610237,rs8054310,rs2542674,rs2689258,rs1033046,rs2010410,rs17835974,rs4783830,rs8060235,rs16953241,rs16953243,rs7200222,rs8049962,rs10521300,rs16953283,rs1126960,rs1868689,rs17176417,rs1079368,rs1004299,rs1004930,rs12930159,rs729633,rs8056104,rs2388632,rs7193399,rs11076030,rs12932839,rs7191827,rs8050506,rs11639567,rs17257349,rs7203944,rs1420303,rs1530793,rs4784379,rs7189231,rs9972796,rs1420285,rs4784390,rs12931301,rs12447674,rs9921518,rs4783845,rs17200070,rs11640012,rs12929998,rs733017,rs716083,rs751214,rs1362437,rs749622,rs8059628,rs1211435,rs1201336,rs1186817,rs1874025,rs8045161,rs8051442,rs1882591,rs1151277,rs11861365,rs2388773,rs1493897,rs8044756,rs1861532,rs11639521,rs17205999,rs16953856,rs1420562,rs2388807,rs1420553,rs1861538,rs4784415,rs12444481,rs1548912,rs7499390,rs4622506,rs4257585,rs4440156,rs7198507,rs9924618,rs11076057,rs4591143,rs6499720,rs4435250,rs4383140,rs4784429,rs4555155,rs9932117,rs11076060,rs12447300,rs13336114,rs1133611,rs11076063,rs11076064,rs8060082,rs4238773,rs12927600,rs4238775,rs13331158,rs4783863,rs8055853,rs4784467,rs6499743,rs16954195,rs4784474,rs1352191,rs7197624,rs11076070,rs8050248,rs1825730,rs16954308,rs11076076,rs4270172,rs8060698,rs12917822,rs8064192,rs1486735,rs1552426,rs7187108,rs8054239,rs11076081,rs2200537,rs9922031,rs1486733,rs12934198,rs2588996,rs2171262,rs17291845,rs7204268,rs2397376,rs9928598,rs12050985,rs4784510,rs1437449,rs16954658,rs991057,rs30922,rs30923,rs11860394,rs31045,rs31046,rs6499755,rs893263,rs31064,rs4784523,rs31103,rs31104,rs360774,rs30905,rs12918370,rs7199709,rs1370385,rs9926841,rs1610101,rs1420227,rs8045690,rs2540707,rs2576542,rs11643666,rs7184310,rs9936365,rs837537,rs7187242,rs7187258,rs11859163,rs17301608,rs2287074,rs7201,rs837550,rs2287072,rs112426189,rs3744374,rs12602590,rs11654604,rs200805689,rs117651561,rs79742527,rs143040759"
        #text = "GG,GA,GG,TT,AA,AA,GG,TC,AA,GG,TT,AA,00,GG,GG,GG,GA,AG,CC,CC,TC,TC,TC,TT,TC,CC,AG,AG,GG,CC,GG,AA,CC,TG,GG,CC,TT,TC,CA,GA,GT,TC,AG,GG,AA,GG,TT,AG,CT,CT,GG,TG,TC,AG,GG,AG,TT,CC,AG,00,AA,AA,AA,GG,TT,AG,TT,CC,TC,TT,GG,TC,CC,CT,CT,AA,GA,AA,AA,GG,GG,AA,GA,GG,AA,AA,CC,AA,AA,CC,TT,CC,GG,GG,AA,AA,CC,AA,GG,AA,TT,00,GG,TT,TT,CC,AA,CC,TT,TT,TT,TT,TT,AA,CT,AG,AA,GT,AA,AA,CC,CT,TC,AA,TT,TT,GG,CC,CC,CC,AA,CC,GG,GG,AG,GG,CC,GG,AG,GA,CC,GG,CC,TT,AG,TT,GG,TT,AA,CT,TC,GG,GG,GG,AA,GG,TT,GG,CC,CC,GG,AA,TT,CC,AA,GG,CC,GG,GA,CT,TT,CT,GA,AG,GA,GG,GA,CA,GG,GG,TT,AA,TT,GT,CC,AA,AG,TC,GA,CC,AA,TT,AA,TT,GG,CC,AC,GG,TC,CT,CC,TT,GG,GA,TC,GG,AA,AC,AG,CC,GG,GG,AA,AG,AG,AA,AA,TT,CC,TT,CC,CT,CC,TC,AG,AA,AG,CC,CT,TC,GG,TG,CC,AA,GG,AA,AC,CC,GG,TT,TT,GG,CC,GA,CT,CT,CC,CC,TG,TC,AA,CA,AG,AG,GG,GA,CC,TC,GA,CC,AG,AC,GA,AA,TG,TT,CT,CC,AG,CC,GG,AA,TT,GG,TT,TT,GG,AA,AA,CC,TG,TC,GT,AG,GA,CC,GG,TT,TG,GA,CT,CC,AA,TT,TC,GG,CT,CT,GA,AG,TT,TT,AA,CC,GG,TT,TT,GG,TT,GG,GG,TC,AG,CA,AA,AA,TG,AG,CC,GG,TT,GG,GG,GG"
        #match = zip(columns.split(','), text.split(','))
       # for (x, y) in match:
       #     data[x] = y
       # print(data)
        input = pd.DataFrame(columns=['AGE'] + columns.split(','), data=[[int(age)] + text.split(',')])
        print(input.info())
        print(input)
        if model == 'model01':
            return model01.predict(input)
        if model == 'model02':
            return model02.predict(input)
    return '{"error":"Ignored..."}'


if __name__ == '__main__':
    http_server = WSGIServer(('', 8081), app)
    http_server.serve_forever()
