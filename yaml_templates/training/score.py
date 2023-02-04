# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""

import os
import sys
import numpy as np
import joblib
#from sklearn.externals import joblib

import math
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
import json
import re
import traceback
import logging
from sklearn.tree import DecisionTreeClassifier

'''
Inference script for Classification:

'''


  
# returns JSON object as 
# a dictionary


def init():
    '''
    Initialize required models:
        Get the config["model_name"] Model from Model Registry and load
    '''
    global prediction_dc
    global model
    
    prediction_dc = ModelDataCollector(config["model_name"], designation="predictions", feature_names=list(config["sample_data"].keys()) + ["Predicted_" + list(config["target_data"].keys())[0]])

    model_path = Model.get_model_path(config["model_name"])
    model = joblib.load(model_path+"/"+config["dataset_name"]+"_model.pkl")
    print('{} Model Loaded. '.format(config["model_name"]))

# def init():
#     '''
#     Initialize required models:
#         Get the IRIS Model from Model Registry and load
#     '''
#     global prediction_dc
#     global model
#     prediction_dc = ModelDataCollector("IRIS", designation="predictions", feature_names=["SepalLengthCm","SepalWidthCm", "PetalLengthCm","PetalWidthCm","Predicted_Species"])

#     model_path = Model.get_model_path('IRIS')
#     model = joblib.load(model_path+"/"+"iris_model.pkl")
#     print('IRIS model loaded...')

def create_response(predicted_lbl):
    '''
    Create the Response object
    Arguments :
        predicted_label : Predicted config["model_name"] Training output
    Returns :
        Response JSON object
    '''
    resp_dict = {}
    print("Predicted {} : ".format(list(config["target_data"].keys())[0]),predicted_lbl)
    name = ("Predicted_" + list(config["target_data"].keys())[0]).lower()
    resp_dict[name] = str(predicted_lbl)
    return json.loads(json.dumps({"output" : resp_dict}))

def run(raw_data):
    '''
    Get the inputs and predict the IRIS Species
    Arguments : 
        raw_data : config["sample_data"]
    Returns :
        Predicted config["model_name"] Output
    '''
    try:
        data = json.loads(raw_data)
        columns = [data[col] for col in list(data.keys()) if not col in list(config["target_data"].keys())]
        predicted_species = model.predict([columns])[0]
        prediction_dc.collect(columns+[predicted_species])
        return create_response(predicted_species)
    except Exception as err:
        traceback.print_exc()