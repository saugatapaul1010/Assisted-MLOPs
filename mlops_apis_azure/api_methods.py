# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""

import subprocess
from mlops_apis.settings import URI
import os
from azureml.core.dataset import Dataset
import azureml.core
from azureml.core import Workspace, Datastore
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import os
import string
import random
from datetime import datetime, timedelta
from urllib.parse import quote
import json
import ruamel.yaml
from mlops_apis.settings import BASE_DIR
import shutil
from msrest.authentication import BasicAuthentication
from azure.devops.v6_0.pipelines.pipelines_client import PipelinesClient
import shutil
from azure.devops.v6_0.pipelines.pipelines_client import PipelinesClient
from azure.devops.v6_0.build.build_client import BuildClient
from msrest.authentication import BasicAuthentication
import yaml
import requests
import base64
import json
import pandas as pd
from datetime import datetime as dt
import textwrap
from ruamel.yaml.scalarstring import PreservedScalarString
import yaml
from inspect import getmembers, isfunction, getsourcelines, getsource
from mlops_apis_azure import functions
import imp
import io
from datetime import datetime

def login_azure(input_params):
    """
    This function is used to login to the azure machine learning 
    portal using PAT, using the azure cli.

    Args :
        input_params : input parameter configurations
    Returns :
        None : Authenticated to Azure DevOps

    """
    cmd = "echo {} | az devops login --organization {}".format(input_params['PAT'], input_params['ORGNIZATION_URL'])
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print("Authenticated to Azure DevOps.")


def sas_key_generation(sa_container, sa_blob, input_params):
    """
    This function us used to genrate a SAS key and use that SAS key
    to generate a SAS URL, that is used to access the files in the
    azure blob containers. This is done using azure sdk.

    Args :
        sa_container = container name follwed by the folder name where the csv files is present. 
        sa_blob = name of the csv file inside the folder, inside the container
        input_params : input parameter configurations
    Returns :
        b_url : SAS URL of the CSV file. This is used to access the CSV file present inside the container 
                from anywhere across the web.
        
    """
    blob_service_client = BlobServiceClient.from_connection_string(
        input_params['MY_CONNECTION_STRING'])
    sas = generate_blob_sas(account_name=blob_service_client.account_name,
                            account_key=blob_service_client.credential.account_key,
                            container_name=sa_container, #azureml-blobstore-fa9d2517-040e-4c2e-be65-676d5c708710/irisdata
                            blob_name=sa_blob, #Iris.csv
                            permission=BlobSasPermissions(read=True, write=True),
                            expiry=datetime.utcnow() + timedelta(weeks=4)) #Change timedelta if token expires.
    enocoded_part = quote(sa_container+"/"+sa_blob)
    b_url = "https://"+blob_service_client.account_name + ".blob.core.windows.net/"+enocoded_part+'?'+sas
    return(b_url)



def get_files_from_datastore(self, container_name, file_name):
    """
    Get the input CSV file from workspace's default data store
    Args :
        container_name : name of the container to look for input CSV
        file_name : input CSV file name inside the container
    Returns :
        data_ds : Azure ML Dataset object
    """
    datastore_paths = [(self.datastore, os.path.join(container_name, file_name))]
    data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    dataset_name = self.args.dataset_name
    if dataset_name not in self.workspace.datasets:
        data_ds = data_ds.register(workspace=self.workspace,
                                    name=dataset_name,
                                    description=self.args.dataset_desc,
                                    tags={'format': 'CSV'},
                                    create_new_version=True)
    else:
        print('Dataset {} already in workspace '.format(dataset_name))
    return data_ds


def create_use_workspace_azure(input_params):
    """
    Function used to create a new workspace in azure, using azure CLI tool.

    Args :
        input_params : input parameter configurations
    Returns :
        None : Workspace created.
    """
    cmd = "az ml workspace create -g {} -w {} -l {} --exist-ok --yes".format(input_params['resourceGroup'],
                                                                             input_params['workspace'],
                                                                             input_params['region'])
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print("Workspace created.")


def create_use_computetarget_azure(input_params):
    """
    Function used to create a new computer cluster in azure, using azure CLI tool.
    This is where the model will get trained.

    Args :
        input_params : input parameter configurations
    Returns :
        None : Computer Target Created.
    """
    cmd = 'az ml computetarget create amlcompute -g {} -w {} -n {} -s {} --min-nodes {} --max-nodes {} --idle-seconds-before-scaledown {}'.format(
            input_params['resourceGroup'],
            input_params['workspace'],
            input_params['computeName'],
            input_params['computeVMSize'],
            input_params['computeMinNodes'],
            input_params['computeMaxNodes'],
            input_params['computeIdleSecs'])
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print("Computer Target Created.")


# TODO : Monitor the Md5 values for each chunks for later enhancements.
def upload_data_viz_azure(input_params):
    """
    This function is used to upload the data visualizations in a specific location
    in the container, and returns the path of the data visualization file in the azure blob
    storage containers.

    Args :
        input_params : input parameter configurations
    Returns :
        blob_url : SAS URL of the visualization file. This is used to access the file present inside the container 
                   from anywhere across the web.
    """

    input_file_name = os.path.basename(input_params['data_viz_file_path'])
    upload_folder_name = input_params['data_viz_upload_id']
    output_blob_name = "{}/{}".format(upload_folder_name, input_file_name)

    blob_service_client = BlobServiceClient.from_connection_string(
        input_params['MY_CONNECTION_STRING'])
    blob_service_client.MAX_CHUNK_GET_SIZE = 4 * 1024 * 1024  # Chunks of size 4MB
    blob_client = blob_service_client.get_blob_client(container=input_params['CONTAINER_NAME'], blob=output_blob_name)

    with open(input_params['data_viz_file_path'], "rb") as data:
        blob_client.upload_blob(data, overwrite=True, validate_content=True)

    blob_base_name = os.path.basename(input_params['data_viz_file_path'])
    container_path = input_params['CONTAINER_NAME'] + "/" + upload_folder_name

    blob_url = sas_key_generation(container_path, blob_base_name, input_params)
    print("Data uploaded to Azure.")
    return blob_url

## TODO : Monitor the Md5 values for each chunks for later enhancements.


def upload_data_azure_in_chunks(input_params):
    """
    This function is used to upload the data in chunks of size 4 MB, to any specific location 
    in the container, and returns the path of the data visualization file in the 
    azure blob storage containers.

    Args :
        input_params : input parameter configurations
    Returns :
        blob_url : SAS URL of the CSV file. This is used to access the CSV file present inside the container 
                   from anywhere across the web.
    """

    input_file_name = os.path.basename(input_params['input_file_path'])
    upload_folder_name = input_params['upload_id']
    output_blob_name = "{}/{}".format(upload_folder_name, input_file_name)

    blob_service_client = BlobServiceClient.from_connection_string(input_params['MY_CONNECTION_STRING'])
    blob_service_client.MAX_CHUNK_GET_SIZE = 4 * 1024 * 1024  # Chunks of size 4MB
    blob_client = blob_service_client.get_blob_client(container=input_params['CONTAINER_NAME'], blob=output_blob_name)

    with open(input_params['input_file_path'], "rb") as data:
        blob_client.upload_blob(data, overwrite=True, validate_content=True)

    blob_base_name = os.path.basename(input_params['input_file_path'])
    container_path = input_params['CONTAINER_NAME'] +  "/" + blob_base_name.split('.')[0]
    blob_url = sas_key_generation(container_path, blob_base_name, input_params)
    print("Data uploaded to Azure.")
    return blob_url

def create_bucket_api(input_params):
    """
    This function is used to create a new bucket.

    Args :
        input_params : input parameter configurations
    Returns :
        create_container.primary_endpoint : if successful, it will return the endpoint of the newly created bucket
        primary_endpoint : if exception, it will return the primary endpoint of the input bucket.
                           the function only gets inside the exception block if the bucket is already
                           present in azure storage.
    """

    blob_service_client = BlobServiceClient.from_connection_string(input_params['MY_CONNECTION_STRING'])
    try:
        create_container = blob_service_client.create_container(input_params['name'])
        response_message = 'Container {} is created'.format(input_params['name'])
        return (True, create_container.primary_endpoint)
    except:
        response_message = 'Container {} already exists'.format(input_params['name'])
        base_url = "https://mlopsws0storageb2304ea4d.blob.core.windows.net/"
        primary_endpoint = base_url + input_params['name']
        return (False, primary_endpoint)

def list_bucket_api(input_params):
    """
    This function is used to list down the name of all the buckets present in the azure datastore

    Args :
        input_params : input parameter configurations
    Returns :
        container_names : list of all the buckets present in the azure datastore.
    """
    # List Bucket API
    container_names = []
    blob_service_client = BlobServiceClient.from_connection_string(input_params['MY_CONNECTION_STRING'])
    containers = blob_service_client.list_containers()
    for container in containers:
        container_names.append(container.name)
    return container_names


def list_dataset_api(input_params):
    """
    This function is used to list down the names of all the datasets present in the azure datastore
    across all the azure blobs.

    Args :
        input_params : input parameter configurations
    Returns :
        dataset_list : list of all the datasets present in all the blobs.
    """
    # List datasets
    blob_service_client = BlobServiceClient.from_connection_string(input_params['MY_CONNECTION_STRING'])
    container_client = blob_service_client.get_container_client(input_params['CONTAINER_NAME'])

    dataset_list = []

    blob_list = container_client.list_blobs(marker=None)
    for blob in blob_list:
        response_dict = dict()
        if(blob.name.endswith("csv")):
            blob_relative_dir = blob.name
            blob_base_name = os.path.basename(blob_relative_dir)
            file_path = "/".join(blob_relative_dir.split("/")[:-1])
            container_path = input_params['CONTAINER_NAME'] + "/" + file_path
            blob_url = sas_key_generation(container_path, blob_base_name, input_params)

            response_dict['dataset_name'] = os.path.basename(blob.name)
            response_dict['id'] = blob.etag
            response_dict['created_at'] = str(blob.creation_time)
            response_dict['updated_at'] = str(blob.last_modified)
            response_dict['url'] = blob_url
            response_dict['version'] = str(blob.version_id)
            response_dict['dataset_size'] = str(blob.size)
            response_dict['cloud_provider'] = "AZURE"
            response_dict['filename'] = os.path.basename(blob.name)
            dataset_list.append(response_dict)
    return dataset_list


def create_model_artifacts_folder(input_params):
    """
    This function is used to create the model artifacts directory in the azure pipelines/self hosted agent, using the azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        None : Artifacts folder created.
    """
    cmd = "mkdir metadata && mkdir models"
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print("Artifacts folder created.")


def start_training(input_params):
    """
    This function is used to start the training process in the azure pipelines/self hosted agent, using the azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        None : Started Training.
    """
    cmd = "az ml run submit-script -g {} -w {} -e {} --ct {} -c {} --source-directory {} --path environment_setup -t ./metadata/run.json {} --container_name {} --input_csv {} --model_path {} --artifact_loc {} --dataset_name {} --dataset_desc {}".format(
        input_params['resourceGroup'], input_params['workspace'], input_params['experimentName'],
        input_params['computeName'], input_params['run_configuration_name'], input_params['source_directory'],
        input_params['train_script_name'], input_params['container_name'], input_params['input_csv'],
        input_params['model_path'], input_params['artifact_loc'],
        input_params['dataset_folder_name'], input_params['dataset_desc'])
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print("Started Training.")


def create_new_repository(input_params):
    """
    This function is used to create a new repository in the azure portal, using the azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        result_create_repo : json dictionary containing success parameters
    """
    login_azure(input_params)
    cmd = "az repos create --name {} --project {} --org {}".format(input_params['repo_name'],input_params['PROJECT_NAME'],input_params['ORGNIZATION_URL'])
    result_create_repo = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result_create_repo = json.loads(result_create_repo.stdout)
    print("Created New Repository.")
    return result_create_repo


def clone_existing_repo_to_new_repo(input_params):
    """
    This function is used to clone an existing repository and create a new one, using the azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        result_clone_repo : json dictionary containing success parameters, and the clone repo URL
    """
    login_azure(input_params)
    cmd = "az repos import create --git-source-url https://SyntbotsAI-RnD:{}@dev.azure.com/SyntbotsAI-RnD/Assisted-MLOPs/_git/Assisted-MLOPs --project {} --org {} --repository {}".format(input_params['PAT'],input_params['PROJECT_NAME'],input_params['ORGNIZATION_URL'],input_params['repo_name'])
    result_clone_repo = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result_clone_repo = json.loads(result_clone_repo.stdout)
    print("Cloned Existing Repository.")
    return result_clone_repo


def register_model(input_params):
    """
    This function is used to register the mdoels after training, using the azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        None : Model Registered.
    """
    cmd = "az ml model register -g {} -w {} -n {} --asset-path {} -d {} --tag 'model'='Decision Tree' --model-framework Custom -f ./metadata/run.json -t metadata/model.json".format(
        input_params['resourceGroup'], input_params['workspace'], input_params['model_registry_name'],
        input_params['model_asset_path'], input_params['model_description'])
    proc = subprocess.Popen(
        cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print("Model Registered.")


def start_pipeline(input_params):
    """
    This function is used to create and start a pipeline, using the azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        None : 
    """
    cmd = "az pipelines create --name NEW_IRIS_USING_CMD_Tuesday_2 --description NEW_IRIS_USING_CMD_Tuesday_2 --repository Assisted-MLOPs --branch master --repository-type tfsgit --yaml-path ./azure-pipelines.yml --project Assisted-MLOPs --subscription b53cd405-74a5-4714-9549-88af4dc84f66"
    proc = subprocess.Popen(
        cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = proc.communicate()


def wrapped(s, width=2000):
    """
    This function is used to wrap the text that should be displayed in the 
    azure YAML file output. It takes a string of any length and outputs the 
    string in the YAML in the required format. Here, width = 2000 means, if 
    we input a string of length 5000 characters, in the YAML it will be displayed 
    as 2000 characters each in the first two lines, and 1000 characters in the 
    3rd line. This is doneto make sure that whatever YAML commands we are displaying 
    in the YAML file, they should and must be displayed in a single line, to avoid 
    formatting issue, that prevents code execution.

    Args :
        s : YAML input string
        width : length of the string till which formatting needs to be applied. (Change this according to need)
    Returns :
        PreservedScalarString : formatted output string for YAML
    """   
    return PreservedScalarString('\n'.join(textwrap.wrap(s, width=width)))

def generate_yaml_file(input_params):
    """
    This function is used to generate the YAML file dynamically. It loads the 
    default template for azure YAML creation, and replaces the variables with 
    the ones present in the input_params dictionary. It further modifies the 
    content of the YAML file for each experiment, based on the inputs. The code
    section is commented to make it more readable in terms of what's happening 
    step by step, during the entire process of generation of the YAML file.

    Args :
        input_params : input parameter configurations
    Returns :
        save_file_location : directory path in local, of the final YAML file.
                             this needs to be pushed to the azure repos.
        azure_yaml : modified YAML file contents in file format. 
    """

    #Read the YAML
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(input_params['yaml_template_location']) as fp:
        azure_yaml = yaml.load(fp)

    #Update the variables here
    for variables in azure_yaml['variables']:
        try:
            if(variables['name'] == 'ml.computeIdleSecs'):
                variables['value'] = input_params['computeIdleSecs']
                
            elif(variables['name'] == 'ml.computeMaxNodes'):
                variables['value'] = input_params['computeMaxNodes']
                
            elif(variables['name'] == 'ml.computeMinNodes'):
                variables['value'] = input_params['computeMinNodes']
                
            elif(variables['name'] == 'ml.computeName'):
                variables['value'] = input_params['computeName']
                
            elif(variables['name'] == 'ml.computeVMSize'):
                variables['value'] = input_params['computeVMSize']
                
            elif(variables['name'] == 'ml.region'):
                variables['value'] = input_params['region']
                
            elif(variables['name'] == 'ml.resourceGroup'):
                variables['value'] = input_params['resourceGroup']
                
            elif(variables['name'] == 'ml.workspace'):
                variables['value'] = input_params['workspace']
                
            elif(variables['name'] == 'ml.experimentName'):
                variables['value'] = input_params['EXPERIMENT_NAME']
                
            elif(variables['name'] == 'system.debug'):
                variables['value'] = input_params['system_debug']
                
            elif(variables['name'] == 'aks.clusterName'):
                variables['value'] = input_params['aks_clusterName']
                
            elif(variables['name'] == 'aks.vmSize'):
                variables['value'] = input_params['aks_vmSize']
                
            elif(variables['name'] == 'aks_service_name'):
                variables['value'] = input_params['aks_service_name']
                
            elif(variables['name'] == 'aks.aksLocation'):
                variables['value'] = input_params['aks_Location']
                
            elif(variables['name'] == 'aks.clusterPurpose'):
                variables['value'] = input_params['aks_clusterPurpose']
                
            elif(variables['name'] == 'aks.agentCount'):
                variables['value'] = input_params['aks_agentCount']

            elif(variables['name'] == 'agent.name'):
                variables['value'] = input_params['agent_name']
                
            elif(variables['name'] == 'run.traininfrasetup'):
                variables['value'] = input_params['run_traininfrasetup']
                
            elif(variables['name'] == 'run.preprocess'):
                variables['value'] = input_params['run_preprocess']
                
            elif(variables['name'] == 'run.train'):
                variables['value'] = input_params['run_train']
                
            elif(variables['name'] == 'run.deployinfrasetup'):
                variables['value'] = input_params['run_deployinfrasetup']
                
            elif(variables['name'] == 'run.deploytoaks'):
                variables['value'] = input_params['run_deploytoaks']
                
            elif(variables['name'] == 'run.publishendpoint'):
                variables['value'] = input_params['run_publishendpoint']
                
            elif(variables['name'] == 'run.deltraincluster'):
                variables['value'] = input_params['run_deltraincluster']
                
            elif(variables['name'] == 'run.delinfcluster'):
                variables['value'] = input_params['run_delinfcluster']
                 
        except:
            continue
        
    for idx, stage in enumerate(azure_yaml['stages']):
        print(str(stage['stage']))
        stage_name = str(stage['stage'])
        stage_object = azure_yaml['stages'][idx]

        #Training Infra Setup
        if(stage_name == "Training_Infra_Setup_Stage"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Azure CLI ML Installation"):          
                        cmd = "az extension add -n azure-cli-ml"
                        step['inputs']['inlineScript'] = cmd
                        
                    elif(step['displayName'] == "Create/Use Workspace"):          
                        cmd = "az ml workspace create -g $(ml.resourceGroup) -w $(ml.workspace) -l $(ml.region) --exist-ok --yes"
                        step['inputs']['inlineScript'] = wrapped(cmd)
                
                    elif(step['displayName'] == "Create/Use Compute Target"):          
                        cmd = "az ml computetarget create amlcompute -g $(ml.resourceGroup) -w $(ml.workspace) -n $(ml.computeName) -s $(ml.computeVMSize) --min-nodes $(ml.computeMinNodes) --max-nodes $(ml.computeMaxNodes) --idle-seconds-before-scaledown $(ml.computeIdleSecs) --location $(ml.region)"
                        step['inputs']['inlineScript'] = wrapped(cmd)
                except:
                    pass
                        
        #Data Preprocessing Stage
        elif(stage_name == "Data_Preprocessing_Stage"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Data Pre-processing Step"):   
                        cmd = "--container_name {} --input_csv {} --dataset_name {} --dataset_desc '{}' --training_columns '{}' --target_column '{}' --processed_file_path $(data.path)".format(input_params['train_stage_dataset_container_name'], input_params['train_stage_input_csv'], input_params['train_stage_dataset_name'], input_params['train_stage_dataset_desc'], input_params['train_columns'], input_params['target_columns'])
                        step['inputs']['arguments'] = wrapped(cmd)

                except:
                    pass
                
        #Training Stage
        elif(stage_name == "Training_Stage"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Create Metadata/Model/Artifcats Folders"):
                        cmd = "mkdir metadata && mkdir models"
                        step['inputs']['script'] = cmd
                        
                    elif(step['displayName'] == "Training Stage"):
                        cmd = "az ml run submit-script -g $(ml.resourceGroup) -w $(ml.workspace) -e $(ml.experimentName) --ct $(ml.computeName) -c {} --source-directory {} --path {} -t ./metadata/run.json {} --container_name {} --input_csv {} --model_path {} --artifact_loc {} --dataset_name {} --dataset_desc '{}' --training_columns '{}' --target_column '{}' --train_size {} --tag_name {} --processed_file_path $(data.path)".format(
                                                                                                    input_params['train_stage_run_configuration_name'], 
                                                                                                    input_params['train_stage_source_directory'],
                                                                                                    input_params['train_stage_environment_setup_path'],
                                                                                                    input_params['train_stage_training_script_name'], 
                                                                                                    input_params['train_stage_dataset_container_name'], 
                                                                                                    input_params['train_stage_input_csv'], 
                                                                                                    input_params['train_stage_model_path'], 
                                                                                                    input_params['train_stage_artifact_loc'],
                                                                                                    input_params['train_stage_dataset_name'], 
                                                                                                    input_params['train_stage_dataset_desc'],
                                                                                                    input_params['train_columns'], 
                                                                                                    input_params['target_columns'],
                                                                                                    input_params['train_size'],
                                                                                                    input_params['tagname'])
                        step['inputs']['inlineScript'] = wrapped(cmd)
                        print(step['inputs']['inlineScript'])

                    elif(step['displayName'] == "Register Model in Model Registry"):
                        cmd = "az ml model register -g $(ml.resourceGroup) -w $(ml.workspace) -n {} --asset-path {} -d \"{}\" --tag \"model\"=\"{}\" --model-framework Custom -f ./metadata/run.json -t metadata/model.json".format(input_params['register_model_name'], 
                                                                                                                                                                                                                                        input_params['register_model_model_asset_path'], 
                                                                                                                                                                                                                                        input_params['register_model_algo_description'],
                                                                                                                                                                                                                                        input_params['register_model_tag'])
                        step['inputs']['inlineScript'] = wrapped(cmd)
                        
                    elif(step['displayName'] == "Publish Pipeline Artifact"):
                        step['inputs']['artifactName'] = input_params['publish_model_artifactName']
                except:
                    pass
                        
        #Deployment_Infra_Setup_Stage
        elif(stage_name == "Deployment_Infra_Setup_Stage"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Install Dependencies"):          
                        cmd = "/home/azureuser/AMLops_Agent/myagent/_work/1/a/_Assisted_MLOPS/{}/a/environment_setup/install-requirements.sh".format(input_params['publish_model_artifactName'])
                        step['inputs']['filePath'] = cmd
                    elif(step['displayName'] == "Create AKS"):          
                        cmd = "az ml computetarget create aks -g $(ml.resourceGroup) -w $(ml.workspace) -n $(aks.clusterName) -s $(aks.vmSize) -a $(aks.agentCount) --cluster-purpose $(aks.clusterPurpose) --location $(aks.aksLocation)"
                        step['inputs']['inlineScript'] = wrapped(cmd)
                    elif(step['displayName'] == "Downloading Pipeline Artifacts"):          
                        step['inputs']['artifactName'] = input_params['publish_model_artifactName']
                except:
                    pass     
    
        #Model Deployment To AKS
        elif(stage_name == "Model_Deployment_To_AKS"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Install Dependencies"):          
                        cmd = "environment_setup/install-requirements.shs"
                        step['inputs']['filePath'] = cmd
                    elif(step['displayName'] == "Deploy ML Model To AKS"):
                        desc_string = '{} Classification Model Deployed to AKS'.format(str(input_params['base_data_name']).capitalize())
                        cmd1 = "az ml model deploy -g $(ml.resourceGroup) -w $(ml.workspace) -n $(aks_service_name) -f metadata/model.json --dc deployment/aksDeploymentConfig.yml --ic deployment/inferenceConfig.yml --ct $(aks.clusterName) --description \'{}\' --overwrite".format(desc_string)
                        step['inputs']['inlineScript'] = wrapped(cmd1)
                except:
                    pass     
        #Run Integration Test And Publish Endpoint
        elif(stage_name == "Run_Integration_Test_And_Publish_Endpoint"):
            for step in stage['jobs'][0]['steps']:
                try:   
                    if(step['displayName'] == "Run Integration Test-AKS"):    
                        cmd1 = 'pytest smoke_tests.py --doctest-modules --junitxml=junit/test-results.xml --cov=integration_test --cov-report=xml --cov-report=html --scoreurl $(az ml service show -g $(ml.resourceGroup) -w $(ml.workspace) -n $(aks_service_name) --query scoringUri -o tsv) --scorekey $(az ml service get-keys -g $(ml.resourceGroup) -w $(ml.workspace) -n $(aks_service_name) --query primaryKey -o tsv)'
                        step['inputs']['inlineScript'] = wrapped(cmd1)
                except:
                    pass       
                
        #Delete Training Cluster
        elif(stage_name == "Delete_Training_Cluster"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Delete Training Cluster"):          
                        cmd = "az ml computetarget delete -n $(ml.computeName) -g $(ml.resourceGroup) -w $(ml.workspace)"
                        step['inputs']['inlineScript'] = wrapped(cmd)
                except:
                    pass       
                
        #Delete Inference Cluster
        elif(stage_name == "Delete_Inference_Cluster"):
            for step in stage['jobs'][0]['steps']:
                try:
                    if(step['displayName'] == "Delete Inference Cluster"):          
                        cmd = "az ml computetarget delete -n $(aks.clusterName) -g $(ml.resourceGroup) -w $(ml.workspace)"
                        step['inputs']['inlineScript'] = wrapped(cmd)
                except:
                    pass       
        
        
    #Save the YAML output in a local/VM
    file_name = os.path.basename(input_params['yaml_template_location'])
    save_file_location = os.path.join(input_params['yaml_output_folder_location'] ,file_name)
    with open(save_file_location, 'w') as fp:
        yaml.dump(azure_yaml, fp)
        
    print("YAML file generated!")
    return save_file_location, azure_yaml

def generate_runconfig_file(input_params):
    """
    This function is used to generate the Runconfig file dynamically. It loads the 
    default template for Runconfig, and replaces the variables with the ones present 
    in the input_params dictionary. 

    Args :
        input_params : input parameter configurations
    Returns :
        save_file_location : directory path in local, of the final Runconfig file.
                             this needs to be pushed to the azure repos. 
        config_file : modified Runconfig file contents in file format. 
    """

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(input_params['train_config_location']) as fp:
        config_file = yaml.load(fp)
        
    for variable_name in config_file['dataReferences']['workspaceblobstore']:
        if(variable_name == 'pathOnDataStore'):
            config_file['dataReferences']['workspaceblobstore'][variable_name] = input_params['train_stage_dataset_container_name']
            
    save_file_location = os.path.join(input_params['train_config_save_location'], "{}_training.runconfig".format(input_params['base_data_name']))
           
    with open(save_file_location, 'w') as fp:
        yaml.dump(config_file, fp)

    print("Runconfig Generated")
    return save_file_location, config_file

def generate_conda_config_file(input_params):
    """
    This function is used to generate the Conda config file dynamically. It loads the 
    default template for Conda config, and replaces the variables with the ones present 
    in the input_params dictionary. 

    Args :
        input_params : input parameter configurations
    Returns :
        save_file_location : directory path in local, of the final Conda config file.
                             this needs to be pushed to the azure repos. 
        config_file : modified Conda config file contents in file format.
    """
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(input_params['conda_config_location']) as fp:
        config_file = yaml.load(fp)
        
    config_file['name'] = "{}_scoring".format(input_params['base_data_name'])
            
    save_file_location = os.path.join(input_params['conda_config_save_location'], "conda_dependencies.yml")
           
    with open(save_file_location, 'w') as fp:
        yaml.dump(config_file, fp)
    print("Conda config Generated")
    return save_file_location, config_file


def script_builder(input_params):
    """
    This function is used to generate the preprocess script and training script dynamically. All the training configurations
    are loaded from input_params dictionary. The training.py file gets created in the local and
    then later, should be pushed to the azure repos. In order to generate the final training.py file, 
    aruguments.py file is used.

    Args :
        input_params : input parameter configurations
    Returns :
        save_file_location : directory path in local, of the final Conda config file.
                             this needs to be pushed to the azure repos. 
        None : Generated Training.py file on the go.
    """

    user_input_preprocess = input_params['preprocess_config']['missing_value_treatment']
    user_input_training = input_params['user_input_training']
    
    ##Code of preprocess builder
    if user_input_preprocess == True:
        config = input_params['preprocess_config']
        imp.reload(functions)
        
        class CodeGenerator:
            def __init__(self, indentation='    '):
                self.indentation = indentation
                self.level = 0
                self.code = ''

            def indent(self):
                self.level += 1

            def dedent(self):
                if self.level > 0:
                    self.level -= 1

            def __add__(self, value):
                temp = CodeGenerator(indentation=self.indentation)
                temp.level = self.level
                temp.code = str(self) + ''.join([self.indentation for i in range(0, self.level)]) + str(value)
                return temp
                
            def __str__(self):
                return str(self.code)
            
        imports = ["os",
                "azureml.core[Workspace, Datastore, Dataset",
                "argparse",
                "numpy as np",
                ]
        L = []
        
        for importlib in imports:
            if importlib.find("[") != -1:
                L.append("from "+importlib.split("[")[0]+ " import "+importlib.split("[")[1]+"\n")
            else:
                L.append("import "+importlib+"\n")
                
        L.append("os.environ['AZURE_DEVOPS_EXT_GITHUB_PAT'] = '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'\n")
        L.append("os.environ['AZURE_DEVOPS_EXT_PAT'] = 'l2ckn6zjch4hsfjcjtwczznuoozrqafgyaff7cya5m5ru3nwmfuq'\n")

        classification_functions = ["__init__","get_files_from_datastore","upload_processed_file_to_datastore"]
        # regression_functions = ["__init__","get_files_from_datastore","create_regression_pipeline","create_confusion_matrix","create_outputs","regression_validate"]
        
        for key,value in config.items():
            if value:
                classification_functions.append(key)
        
        funcs = getmembers(functions.PreprocessFunctions, isfunction)
        # print(funcs)
        ind = CodeGenerator()
        ind += "".join(L)
        ind += "\n\n"
        ind+="class AzurePreprocessing():\n"
        functions_list = classification_functions

        for each in functions_list:
            for func in funcs:
                if func[0]==each:
                    ind+="\n"
                    # ind.indent()
                    flines = [line.rstrip()+"\n" for line in getsourcelines(func[1])[0]]
                    for fline in flines:
                        ind+= fline
                    # ind.dedent()
                    
        from mlops_apis_azure.arguments import arguments_function
        argument_list = arguments_function(input_params)
        argument_list['arguments_preprocess']
        ind.dedent()
        ind+='\nif __name__ == "__main__":\n'
        ind.indent()
        
        # Data balancing technique,preprocess_remove_whitespace applicable only when univariate and text classification is True. 
        argparams = argument_list['arguments_preprocess']
        
        for line in argparams:
            ind+=line+"\n"
        ind += "preprocessor.__init__(args)\n"
        
        for key,value in config.items():
            if value:
                ind += "preprocessor."+key+"()\n"        

        # writing to file
        file1 = open(input_params['preprocess_file_location'], 'w')
        file1.writelines(str(ind))
        file1.close()
        
    if user_input_training == True:
        ##Code of training builder
        
        config = input_params['model_config']
        print(config)

        imp.reload(functions)

        class CodeGenerator:
            def __init__(self, indentation='    '):
                self.indentation = indentation
                self.level = 0
                self.code = ''

            def indent(self):
                self.level += 1

            def dedent(self):
                if self.level > 0:
                    self.level -= 1

            def __add__(self, value):
                temp = CodeGenerator(indentation=self.indentation)
                temp.level = self.level
                temp.code = str(self) + ''.join([self.indentation for i in range(0, self.level)]) + str(value)
                return temp
                
            def __str__(self):
                return str(self.code)
            
        imports = ["os",
                "azureml.core[Workspace, Datastore, Dataset",
                "azureml.core.run[Run",
                "argparse",
                "logging",
                "sklearn.model_selection[train_test_split",
                "sklearn.ensemble[RandomForestClassifier,ExtraTreesClassifier",
                "sklearn.svm[LinearSVC",
                "sklearn.naive_bayes[MultinomialNB",
                "sklearn.linear_model[LogisticRegression",
                "sklearn.tree[DecisionTreeClassifier",
                "sklearn.model_selection[cross_val_score",
                "sklearn.metrics[classification_report, confusion_matrix, precision_score, recall_score, accuracy_score",
                "pandas as pd",
                "numpy as np",
                "re",
                "seaborn as sn",
                "matplotlib.pyplot as plt",
                "joblib",
                ]
        L = []
        
        for importlib in imports:
            if importlib.find("[") != -1:
                L.append("from "+importlib.split("[")[0]+ " import "+importlib.split("[")[1]+"\n")
            else:
                L.append("import "+importlib+"\n")
                
        classification_functions = ["__init__","get_files_from_datastore","create_confusion_matrix","create_outputs","validate"]
        # regression_functions = ["__init__","get_files_from_datastore","create_regression_pipeline","create_confusion_matrix","create_outputs","regression_validate"]
        
        for key,value in config.items():
            if value:
                classification_functions.append(key)
                

        funcs = getmembers(functions.TrainingFunctions, isfunction)
        # print(funcs)
        ind = CodeGenerator()
        ind += "".join(L)
        ind+="class AzureClassification():\n"
        functions_list = classification_functions

        for each in functions_list:
            for func in funcs:
                if func[0]==each:
                    ind+="\n"
                    # ind.indent()
                    flines = [line.rstrip()+"\n" for line in getsourcelines(func[1])[0]]
                    for fline in flines:
                        ind+= fline
                    # ind.dedent()   
                    
        from mlops_apis_azure.arguments import arguments_function
        argument_list = arguments_function(input_params)
        ind.dedent()
        ind+='\nif __name__ == "__main__":\n'
        ind.indent()
        
        # Data balancing technique,preprocess_remove_whitespace applicable only when univariate and text classification is True. 
        argparams = argument_list['arguments_training']
        
        for line in argparams:
            ind+=line+"\n"
        ind += "classifier.__init__(args)\n"
        
        for key,value in config.items():
            if value:
                ind += "classifier."+key+"()\n" 
                

        # writing to file
        file1 = open(input_params['training_file_location'], 'w')
        file1.writelines(str(ind))
        file1.close()
        print("Generated Training.py file on the go.")

def commit_file_in_yaml_to_azure_repo(input_params):
    """
    This function is used to commit all the file changes to the remote 
    azure git repositry. This function will clone an existing repository,
    delete the target files from local repository path, copy all the modified 
    files to their respective local directories, push the changes to remote 
    repo, and finally delete the cloned repo folder from local (or VM).For 
    detailed understanding, refer to the comments and print statements line by line. 

    Disclaimer: Sometimes, due to a bug in Python cache management system, the cloned repository
    folder might not get deleted. In that case, you have to manually delete the 
    clone folder from your local machine. Otherwise, the code will not run.

    Args :
        input_params : input parameter configurations
    Returns :
        True : Success Case
        False: Failure Case
    """
    try:
        print("Commiting File to Azure Repo..................")

        folder_name_for_repo = input_params['repo_name']
        save_file_name = os.path.basename(input_params['save_file_location'])


        if(os.path.isdir(os.path.join(BASE_DIR, folder_name_for_repo))):
            os.remove(os.path.join(BASE_DIR, folder_name_for_repo))
            print("Existing repo folder deleted")

        cmd = "git clone https://SyntbotsAI-RnD:{}@dev.azure.com/SyntbotsAI-RnD/Assisted-MLOPs/_git/{} --branch {}".format(input_params["PAT"],input_params['repo_name'],input_params["repo_branch"])
        #AUTHETICATION - COMMAND ----------------------------------------------------------------------------------------
        print(cmd)
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = proc.communicate()  

        print("Head currently pointed at --> ", os.getcwd())

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, save_file_name)):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, save_file_name))
            print("Deleteing the YAML file from repo")

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['training_file_location']))):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['training_file_location'])))
            print("Deleteing the training script file from repo")

        if os.path.isfile(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['preprocess_file_location']))):  
            os.remove(os.path.join(BASE_DIR,folder_name_for_repo, os.path.basename(input_params['preprocess_file_location'])))
            print("Deleteing the preprocess script file from repo")

        #Copy all the files to their desired locations (for the repo)
        shutil.copy(input_params['save_file_location'], folder_name_for_repo)
        shutil.copy(input_params['preprocess_file_location'], folder_name_for_repo)
        shutil.copy(input_params['training_file_location'], folder_name_for_repo)
        shutil.copy(input_params['train_config_save_location'], os.path.join(folder_name_for_repo, "environment_setup", ".azureml"))
        shutil.copy(input_params['score_file_save_location'], os.path.join(folder_name_for_repo, "inference"))
        shutil.copy(input_params['smoke_test_file_save_location'], os.path.join(folder_name_for_repo, "tests", "smoke"))
        shutil.copy(input_params['conda_config_save_location'], os.path.join(folder_name_for_repo, "deployment"))

        #Change the current working head.
        os.chdir(os.path.join(BASE_DIR,folder_name_for_repo))
        print("Changing head to -->", os.getcwd())

        #Standard GIT commands to push all the files to the remote repository
        cmd = "git add ." 
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = proc.communicate()  
        
        cmd = "git commit -m \"Updated Azure Pipeline for API run\""
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = proc.communicate()  
            
        cmd = "git push" 
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = proc.communicate()  

        print("Updated the YAML file in Azure Repo. Starting Pipeline run.")
        
        os.chdir(os.path.join(BASE_DIR))
        print("Changing head to project working directory-->", os.getcwd())

        shutil.rmtree(folder_name_for_repo, ignore_errors=True)
        print("------------------------ Repository Deleted from Project Directory ------------------------")
        return True
    except:
        return False

def create_new_pipeline(input_params):
    """
    This function is used to create a new pipeline, using azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        result : json dictionary containing pipeline creation response.
    """
    login_azure(input_params)
    cmd = "az pipelines create --name \'{}\' --description \'{}\' --repository \'{}\' --branch \'{}\' --repository-type tfsgit --project \'{}\' --organization \'{}\' --yml-path \'{}\' --service-connection \'{}\' --subscription \'{}\' --skip-first-run true".format(input_params["pipeline_name"], input_params["pipeline_description"], input_params["new_pipeline_commit_repo"], input_params['repo_branch'], input_params['project_name'], input_params['ORGNIZATION_URL'], input_params['yaml_path'], input_params['ServiceConnection_NAME'], input_params['SUBCRIPTION_ID'])
    result = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = json.loads(result.stdout)
    result['pipeline_creation_status'] = True
    return result

def execute_current_pipeline(input_params):
    """
    This function is used to execute a pipeline, using azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        result_execute_pipeline : json dictionary containing pipeline execution response.
    """
    login_azure(input_params)
    cmd_set_acc = "az account set -s {}".format(input_params['SUBCRIPTION_ID'])
    subprocess.run(cmd_set_acc, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    cmd = "az pipelines build queue --branch {} --org {} --project {} --definition-id {} --subscription {}".format(input_params['repo_branch'],input_params['ORGNIZATION_URL'],input_params['PROJECT_NAME'],input_params['PIPELINE_ID'],input_params['SUBCRIPTION_ID'])
    result_execute_pipeline = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result_execute_pipeline = json.loads(result_execute_pipeline.stdout)
    print("Started Pipeline execution.")
    return result_execute_pipeline

def get_pipeline_details(input_params):
    """
    This function is used to get details of a pipeline using it's name, using azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        result[0] : json dictionary containing pipeline details as response. return only the zero'eth index for the latest run
    """
    login_azure(input_params)
    cmd = "az pipelines list --name \'{}\' --organization \'{}\'  --project \'{}\' ".format(input_params["pipeline_name"], input_params['ORGNIZATION_URL'], input_params['project_name'])  
    result = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = json.loads(result.stdout)
    return result[0]

def check_if_pipeline_exists(input_params):
    """
    This function is used to check whether a given pipeline exists, using azure cli tool.

    Args :
        input_params : input parameter configurations
    Returns :
        pipeline_exists : Boolean. True if pipeline is already present, False otherwise.
        pipeline_dict : Blank if new pipeline is not created, output response json dictionary ff pipeline gets created 
    """
    cmd = "az pipelines list --org {} --project {}".format(input_params['ORGNIZATION_URL'], input_params['PROJECT_NAME'])
    result = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = json.loads(result.stdout)
    num_of_existing_pipelines = len(result)
    pipeline_dict = dict()
    for pipe in result:
        pipeline_dict[pipe['name']] = pipe['id']
        
    if(input_params['pipeline_name'] in list(pipeline_dict.keys())):
        pipeline_exists = True
        return pipeline_exists, pipeline_dict
    else:
        pipeline_exists = False
    return pipeline_exists, dict()

def get_experiment_name(input_params, build, run_id):
    """
    This function is used to get the experiment name based on run_id and build_id

    Args :
        input_params : input parameter configurations
        build : build id of the experiment
        run_id : run if of the experiment
    Returns :
        experiment_name : experiment name based on the run_id and build id.
    """
    get_log_url = build.get_build_logs(input_params['PROJECT_NAME'], run_id)[0].url
   
    data = [
     {
     "op": "add",
     "path": "/fields/System.Title",
     "value": "Sample task"
     }
    ]

    response = requests.get(get_log_url, 
        headers={'Content-Type': 'application/json-patch+json'},
        auth=('', input_params['PAT']))

    yaml_file = yaml.safe_load(response.text)
    experiment_name = [x['value'] for x in yaml_file['variables'] if x['name']=='ml.experimentName'][0]
    
    return experiment_name

def get_latest_experiment_list(input_params, experiment_list):
    """
    This function is used to get the list of all the latest build for each experiments.

    Args :
        input_params : input parameter configurations
        experiment_list : list of all experiments.
    Returns :
        final_experiment_list : list of all the latest build for each experiments.
    """

    df = pd.DataFrame(experiment_list, columns = list(experiment_list[0].keys()))
    new_df = df.drop_duplicates(subset=['experiment_name'], keep='first')
    new_df = new_df.sort_values(by = 'last_updated')
    final_experiment_list = []
    filtered_experiment_dic = new_df.T.to_dict()
    
    for exp_dic in filtered_experiment_dic.keys():
        final_experiment_list.append(filtered_experiment_dic[exp_dic])
        
    return final_experiment_list

    
def calculate_duration(start_time, end_time):
    """
    Calculate difference between two timestamps
    Args :
        start_time : starting time stamp
        end_time : ending time stamp
    Returns :
        time delta : time difference 
    """
    return (end_time - start_time)


def pipelines_status(input_params, definition_id_list):
    """
    This function is used to get the details of all the pipelines,
    based on the input definition id lists.

    Args :
        input_params : input parameter configurations
        definition_id_list : list of all pipeline ids.
    Returns :
        latest_run_of_each_pipeline_list : list object, containing details for each pipeline run
    """
    login_azure(input_params)
    latest_run_of_each_pipeline_list = []

    #Keep this import statement here, to avoid any errors. Do not remove it from here and put it at the top.
    from mlops_apis.settings import input_params

    for pipeline_id in definition_id_list:
        pipe_dict = dict()
        cmd = "az pipelines runs list --project {} --branch {} --pipeline-ids {} --org {}".format(input_params['project_name'],input_params['repo_branch'],pipeline_id,input_params['ORGNIZATION_URL'])
        result_pipeline_runs = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        result_pipeline_runs = json.loads(result_pipeline_runs.stdout)
        filtered_result_pipeline_runs = sorted(result_pipeline_runs, key=lambda d: d['lastChangedDate'])
        
        if(len(filtered_result_pipeline_runs) != 0):
        
            build_no = filtered_result_pipeline_runs[0]['buildNumber'].split(".")[0]
            date_created = build_no[0:4] + "-" + build_no[4:6] + "-" + build_no[6:]
            filtered_result_pipeline_runs = filtered_result_pipeline_runs[-1]
            pipe_dict['pipeline_name'] = filtered_result_pipeline_runs['definition']['name']
            pipe_dict['pipeline_id'] = filtered_result_pipeline_runs['definition']['id']
            pipe_dict['build_url'] = filtered_result_pipeline_runs['url']
            pipe_dict['pipeline_revision_number'] = filtered_result_pipeline_runs['definition']['revision']
            
            try:
                pipe_dict['startTime'] = str(datetime.fromisoformat(filtered_result_pipeline_runs['startTime']))
            except:
                pipe_dict['startTime'] = date_created
    
            try:
                pipe_dict['finishTime'] = str(datetime.fromisoformat(filtered_result_pipeline_runs['finishTime']))
                pipe_dict['duration'] = str(calculate_duration(datetime.fromisoformat(filtered_result_pipeline_runs['startTime']),datetime.fromisoformat(filtered_result_pipeline_runs['finishTime'])))
            except:
                pipe_dict['finishTime'] = 0
                pipe_dict['duration'] = 0
    
            pipe_dict['date_created'] = date_created
            pipe_dict['last_updated_time'] = filtered_result_pipeline_runs['definition']['project']['lastUpdateTime']
            pipe_dict['code_repo_url'] = filtered_result_pipeline_runs['repository']['url']
            pipe_dict['commit_id'] = filtered_result_pipeline_runs['sourceVersion']
            pipe_dict['commit_branch'] = filtered_result_pipeline_runs['sourceBranch']
            pipe_dict['commit_url'] = pipe_dict['code_repo_url'] + "/" + "commit" + "/" +pipe_dict['commit_id']
            pipe_dict['run_status'] = filtered_result_pipeline_runs['status']
            pipe_dict['pipeline_state'] = filtered_result_pipeline_runs['definition']['project']['state']
        
        else:
            pipe_dict['pipeline_name'] = "Created"
            pipe_dict['pipeline_id'] = pipeline_id
            pipe_dict['build_url'] = "NotCreated"
            pipe_dict['pipeline_revision_number'] = 0
            pipe_dict['startTime'] = 0
            pipe_dict['finishTime'] = 0
            pipe_dict['duration'] = 0
            pipe_dict['date_created'] = "NotCreated"
            pipe_dict['last_updated_time'] = "NotCreated"
            pipe_dict['code_repo_url'] = "NotCreated"
            pipe_dict['commit_id'] = "NotCreated"
            pipe_dict['commit_branch'] = "NotCreated"
            pipe_dict['commit_url'] = "NotCreated"
            pipe_dict['run_status'] = "Created"
            pipe_dict['pipeline_state'] = "Created"
        latest_run_of_each_pipeline_list.append(pipe_dict)
    return latest_run_of_each_pipeline_list

def format_dataset(df):
    """
    This function is used to load a dataset from the azure data stores, converts it into a dictioanry 
    format and returns it.

    Args :
        df : convert pandas dataframe object to dictionary
    Returns :
        new_dic : formatted dataset dictionary
    """
    new_dic = df.to_dict('series')
    for key in new_dic.keys():
        new_dic[key] = new_dic[key].tolist()
    return new_dic
    
def return_dataset_from_azure_blobstorage(blob_name, container_name, folder_name, input_params):
    """
    This function is used to load a dataset from the azure data stores, converts it into a dictioanry 
    format and returns it.

    Args :
        input_params : input parameter configurations
        blob_name : name of the blob (name of the dataset.csv file)
        container_name : name of the container
        folder_name : name of the folder inside the container
    Returns :
        dataset_dic : dataset converted to dictionary format from CSV format
    """
    dataset_dic = dict()
    sa_container = container_name + "/" + folder_name
    sa_blob = blob_name
    URL = sas_key_generation(sa_container, sa_blob, input_params)
    
    if(requests.get(URL).status_code == 200):
        df_response = requests.get(URL).content
        df_response = pd.read_csv(io.StringIO(df_response.decode('utf-8')))
        dataset_dic['dataset_present'] = True
        dataset_dic['dataset_values'] = format_dataset(df_response.sample(10))
    else:
        dataset_dic['dataset_present'] = False
        dataset_dic['dataset_values'] = dict()
    return dataset_dic

def list_pipelines_in_dev_ops(input_params):
    """
    This function is used to list all the pipelines that are present in the azure dev ops portal.
    Detailed are displayed in tabular format in the response body.

    Args :
        input_params : input parameter configurations
    Returns :
        dic_whole : dictionary containing all the details of all the pipelines
    """
    login_azure(input_params)
    cmd = "az pipelines build list --organization {} --project {} -o table".format(input_params['ORGNIZATION_URL'] , input_params['PROJECT_NAME'] )
    result = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = result.stdout
    final = result.decode('utf-8').replace("-",'')

    cols = ['Run-ID', 'Version-Number', 'Status', 'Result', 'Pipeline-ID', 'Pipeline-Name', 'Source-Branch', 'Queued-Date', 'Queued-Time', 'Reason']
    
    final_list = final.splitlines()
    final_line_list_all = []
    final_line_list_no_result = []
    for line in final_list:
        if(len(line.split()) == 10):
            final_line_list_all.append(line.split()) 
        elif(len(line.split()) < 10 and len(line.split()) > 1):
            final_line_list_no_result.append(line.split())
            
    df1 = pd.DataFrame(final_line_list_all, columns = cols)
    final_line_list_no_result_new = []
    
    for list_obj in final_line_list_no_result:
        list_obj.insert(3, "awaiting")
        final_line_list_no_result_new.append(list_obj)
        
    df2 = pd.DataFrame(final_line_list_no_result_new, columns = cols)
    df = pd.concat([df1,df2], axis=0)
    df = df.sort_values(['Run-ID'], ascending=[False])

    #df.to_csv('C:/Users/sauga/Desktop/pipelinestatus.csv', index=None)
    dic_whole = format_dataset(df)    
    return dic_whole

def get_endpoint_details(input_params,endpoint_name):
    """
    This function is used to generate the endpoint details based on the name of the endpoint.

    Args :
        input_params : input parameter configurations
        endpoint_name: enpoint name whose detail needs to be fetched
    Returns :
        endpoint_details : endpoint details after the model has been desployed successfully
    """
    login_azure(input_params)
    cmd = "az ml endpoint realtime list --workspace-name {} --resource-group {}".format(input_params['workspace'],input_params['resourceGroup']) 
    result_endpoint = subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result_endpoint_list = json.loads(result_endpoint.stdout)
    
    index = 0
    for i in range(len(result_endpoint_list)):
        if(result_endpoint_list[i]['name'] == endpoint_name):
            index = i
            break
        else:
            index="null"
            
    if(index != "null"):
        endpoint_details = dict()
        endpoint_details = result_endpoint_list[index]
        IP = endpoint_details['scoringUri'].split("/")[2].split(":")[0]
        endpoint_details['swaggerURL'] = "http://{}/api/v1/service/{}/swagger.json".format(IP, endpoint_name)

    else:
        endpoint_details = dict()
        endpoint_details['computeType'] = "null"
        endpoint_details['name'] = endpoint_name
        endpoint_details['properties'] = "null"
        endpoint_details['scoringUri'] = "null"
        endpoint_details['state'] = "null"
        endpoint_details['tags'] = "null"
        endpoint_details['updatedAt'] = "null"
        endpoint_details['swaggerURL'] = "null"
        
    return endpoint_details

def create_deployment_config_json(input_params):
    """
    This function is used to generate the deployment_config file dynamically. It loads the 
    default template for deployment_config, modifies them based on the input and experiment
    type, and saves the files in local. This is later pushed to the azure repo

    Args :
        input_params : input parameter configurations
    Returns :
        None : Deployment config generated.
    """

    blob_name = input_params['train_stage_input_csv']                     #filename
    container_name = input_params['CONTAINER_NAME']                       #bucketname
    folder_name = input_params['train_stage_dataset_container_name']      #upload_id
    
    sa_container = container_name + "/" + folder_name
    sa_blob = blob_name
    URL = sas_key_generation(sa_container, sa_blob, input_params)

    if(requests.get(URL).status_code == 200):
        df_response = requests.get(URL).content
        df_response = pd.read_csv(io.StringIO(df_response.decode('utf-8')))
        df = df_response.sample(n=1)

        train_cols = input_params['train_columns'].split(",")
        target_cols = input_params['target_columns']

        X_df = df[train_cols]
        X_dic = X_df.to_dict(orient='list')

        for key in X_dic.keys():
            X_dic[key] = X_dic[key][0]
            
        y_df = df[target_cols].to_frame()
        y_dic = y_df.to_dict(orient='list')

        for key in y_dic.keys():
            y_dic[key] = y_dic[key][0]

        config_dict = dict()
        config_dict["sample_data"] = X_dic
        config_dict["target_data"] = y_dic
        config_dict["model_name"] = input_params['base_data_name_for_commit'].upper()
        config_dict["dataset_name"] = input_params['base_data_name_for_commit'].lower()

        #Write code to save JSON file in root direcoty of Project Folder.
        with open(input_params['save_config_location'], 'w') as f:
            json.dump(config_dict, f)

    else:
        pass

    print("Deployment config generated")

def generate_deployment_files(input_params):
    """
    This function is used to generate the score.py file and smoke_tests.py files dynamically. It loads the 
    default template for score.py and smoke_tests.py, modifies both of them based on the input and experiment
    type, and saves the files in local. This is later pushed to the azure repo

    Args :
        input_params : input parameter configurations
    Returns :
        None : None
    """

    json_path = input_params['save_config_location']
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    #Generate Score files
    f = open(input_params['score_file_template_location'], "r")
    score_file = f.read()

    config_dict_str = "config = {}\n\n".format(str(config_dict))

    new_score_file = open(input_params['score_file_save_location'],"w+")
    new_score_file.write(config_dict_str)
    new_score_file.write(score_file)
    new_score_file.close()

    #Generate Smoke Test files
    f = open(input_params['smoke_test_file_template_location'], "r")
    smoke_file = f.read()

    config_dict_str = "config = {}\n\n".format(str(config_dict))

    new_smoke_file = open(input_params['smoke_test_file_save_location'],"w+")
    new_smoke_file.write(config_dict_str)
    new_smoke_file.write(smoke_file)
    new_smoke_file.close()