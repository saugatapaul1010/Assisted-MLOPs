# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""


from celery.decorators import task
from celery.utils.log import get_task_logger

from .celery_functions import generate_visualization
from mlops_apis_azure import api_methods
from mlops_apis.settings import input_params


logger = get_task_logger(__name__)

# @task(name="generate_dataset_visualization")
# def generate_viz_task(file_path,dataset_id,upload_id):
#     """generates visualiation after dataset is uploaded successfully"""
#     logger.info("generated dataset visualization")
#     generate_visualization(file_path,dataset_id,upload_id)

@task(name="azure_data_upload_in_chunks")
def upload_data_azure_task(file_path, upload_id):
    """
    Used for Azure blob upload
    """
    logger.info("generated dataset visualization")
    input_params['input_file_path'] = file_path
    input_params['upload_id'] = upload_id
    api_methods.upload_data_azure_in_chunks(input_params)


@task(name="commit_file_in_yaml_to_azure_repo")
def commit_file_in_yaml_to_azure_repo_task(input_params):
    """
    Used for commiting file ato Azure Repo
    """
    logger.info("function to commit yaml file to azure repo")
    result = api_methods.commit_file_in_yaml_to_azure_repo(input_params)
    logger.info(result)

@task(name="execute_current_pipeline")
def execute_current_pipeline_task(input_params):
    """
    Used for executing the pipeline
    """
    logger.info("function to execute the pipeline")
    result_execute_pipeline = api_methods.execute_current_pipeline(input_params)
    logger.info(result_execute_pipeline)
    return result_execute_pipeline
    