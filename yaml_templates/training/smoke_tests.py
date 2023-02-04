# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""

import requests
import json


req_sample = config["sample_data"]

def test_ml_service(scoreurl, scorekey):
    """
    Function used to prepare the response headers, and test if the deployed model 
    is returning a request.
    Args :
        scoreurl : endpoint url
        scorekey : endpoint access key
    Returns :
        None : 
    """
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    assert resp.status_code == requests.codes["ok"]
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0

#TEST
def test_prediction(scoreurl, scorekey):
    """
    Function used to prepare the response headers, and test the output of the deployed
    Args :
        scoreurl : endpoint url
        scorekey : endpoint access key
    Returns :
        None : 
    """
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    resp_json = json.loads(resp.text)
    assert resp_json['output']['predicted_'+list(config["target_data"].keys())[0].lower()] == str(list(config["target_data"].values())[0])