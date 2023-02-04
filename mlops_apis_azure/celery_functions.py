# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from django.conf import settings
import os
from .models import Dataset

def generate_visualization(file_path, dataset_id, upload_id):
    """
    This function is used to generate visualizations, using 
    Pandas Profiler and and uploads the visualization in 
    HTML format

    Args :
        file_path : path of the input dataset in CSV format
        dataset_id : id of the dataset
        upload_id : correponds to the celery task id (folder with the same name as task id, is created in the azure data storage)
    Returns :
        None : 
    """

    _, file_extension = os.path.splitext(file_path)
    if file_extension==".csv":
      df = pd.read_csv(os.path.join(settings.BASE_DIR,file_path))
    elif file_extension==".xlsx":
      df = pd.read_excel(os.path.join(settings.BASE_DIR,file_path), engine='openpyxl')
    profile = ProfileReport(df, title="Pandas Profiling Report")
    viz_url = os.path.join(settings.VISUALIZATION_REPORTS_PATH,upload_id,"report.html")
    profile.to_file(viz_url)
    dataset = Dataset.objects.get(pk=dataset_id)
    dataset.visualize_url = viz_url
    dataset.save()