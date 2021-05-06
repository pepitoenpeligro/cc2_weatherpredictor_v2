from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import pmdarima as pm
import pymongo
import pickle
from datetime import datetime, timedelta
import requests
import os
import zipfile
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from zipfile import ZipFile

class Modelo:
    def __init__(self):
        client = pymongo.MongoClient(
            "mongodb+srv://"+"pepitoenpeligro"+":"+"QDGzSuG1Wv9QtQ6Z"+"@cluster0.xoro9.mongodb.net/test?retryWrites=true&w=majority")
        self.data = client.p2Airflow['sanfrancisco']

    def create_export_model(self):
        dataframe = pd.DataFrame(list(self.data.find()))
        print(dataframe)
        dataframe = dataframe.dropna()
        #print(dataframe.columns)

        model_temp = SARIMAX(dataframe['TEMP'].to_numpy(dtype='float', na_value=np.nan) ,
                                order=(0,1,1),
                                enforce_invertibility=False,
                                enforce_stationarity=False)
        model_temp = model_temp.fit()

        # predicciones = model_temp.predict(nperiods=24, return_conf_int=True)
        # print("Predicciones")
        # print(predicciones)
        pickle.dump(model_temp, open("./modelos/model__temp__sarimax.p", "wb" ) )

        model_hum = SARIMAX(dataframe['HUM'].to_numpy(dtype='float', na_value=np.nan) ,
                            order=(0,1,1),
                            enforce_invertibility=False,
                            enforce_stationarity=False)
        model_hum = model_hum.fit()

        pickle.dump(model_hum, open("./modelos/model__hum__sarimax.p", "wb" ) )


    def compress(self):
        with ZipFile('./modelos/model__temp__sarimax.p.zip', 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.write('./modelos/model__temp__sarimax.p')

        with ZipFile('./modelos/model__hum__sarimax.p.zip', 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.write('./modelos/model__hum__sarimax.p')



if __name__ == "__main__":
    m = Modelo()
    m.create_export_model()
    m.compress()