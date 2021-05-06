from flask import Flask, Response, jsonify
import json
import pickle
import pandas as pd
import pmdarima as pm
import os
from statsmodels.tsa.arima_model import ARIMA
import time
from datetime import datetime, timedelta

import arima_dao
import middleware
from zipfile import ZipFile



app = Flask(__name__)
app.wsgi_app = middleware.LoggerMiddleware(app.wsgi_app)




def predict_weather(interval):

    with ZipFile('./modelos/model__temp__sarimax.p.zip', 'r') as myzip:
        myzip.extractall('./')

    with ZipFile('./modelos/model__hum__sarimax.p.zip', 'r') as myzip:
        myzip.extractall('./')
    

    arima_temp = pickle.load( open( './modelos/model__temp__sarimax.p', "rb" ) )
    predicc_temp = arima_temp.predict(n_periods=interval, return_conf_int=True)

    aria_hum = pickle.load( open( './modelos/model__hum__sarimax.p', "rb" ) )
    predicc_hum = aria_hum.predict(n_periods=interval, return_conf_int=True)

    primera_fecha = datetime.now() + timedelta(hours=3)
    rango_fechas = pd.date_range(primera_fecha.replace(second=0, microsecond=0), periods=interval, freq='H')
    prediction_response = []

    for tiempo, temp, hum in zip(rango_fechas, predicc_temp, predicc_hum):
        tiempo_unix = time.mktime(tiempo.timetuple())
        prediction_response.append(
            {'hour': datetime.utcfromtimestamp(tiempo_unix).strftime('%d-%m %H:%M'),
            'temp': temp,  
             'hum': hum
            })
    return prediction_response

@app.route("/servicio/v2/prediccion/test",methods=['GET'])
def test():
    response = Response("Test Api V2", status=200)
    response.headers['Content-Type']='application/json'
    return response

@app.route("/servicio/v2/prediccion/24horas",methods=['GET'])
def hours_24():
    response = Response(json.dumps(predict_weather(24)), status=200)
    return response


@app.route("/servicio/v2/prediccion/48horas",methods=['GET'])
def hours_48():
    response = Response(json.dumps(predict_weather(48)), status=200)
    return response

@app.route("/servicio/v2/prediccion/72horas",methods=['GET'])
def hours_72():
    response = Response(json.dumps(predict_weather(72)), status=200)
    return response

@app.after_request
def after(response):
    response.headers['Content-Type']='application/json'
    return response