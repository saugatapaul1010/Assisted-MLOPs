
# Assisted MLOps Azure Controller Backend Setup Guide

One Platform to deploy you machine learning models in automated pipeline on any cloud GCP, Azure and AWS.


## Installation in Local

### 1. Create Conda Environment, Glone the GIT Repo, Install Dependencies
You should have anaconda, celery installed as a pre-requisite to run the below steps.

```bash
   conda create --name assisted_mlops_env python=3.8 
   conda activate assisted_mlops_env
   git clone https://git.atosone.com/amlops/azure_assisted_mlops.git
   cd azure_assisted_mlops
   pip install -r requirements.txt
   python manage.py runserver
```


### 2. Start Redis Server in Local

Open another terminal and activate the same conda environment --> "conda activate assisted_mlops_env"

```bash
   nohup celery -A mlops_apis worker -l info -P solo -E -Q azure_async_q --concurrency=500 &        #Using Nohup
   celery -A mlops_apis worker -l info -P solo -E -Q azure_async_q --concurrency=500                #Without using Nohup
```


## Deployment of Backend in VM


### 1. Configure the VM

You should have a signed version of mlops.ppk and mlops.pem files for authentication.

```bash
   ssh -i ./mlops.pem azureuser@20.185.91.194         #Enter Password : mlops@123
```


### 2. Glone the GIT Repo, Setup Docker Containers, Run the Docker Image.
You should have anaconda, celery installed as a pre-requisite to run the below steps.

```bash
   git clone https://git.atosone.com/amlops/azure_assisted_mlops.git
   cd azure_assisted_mlops
   sudo docker build -t azure-amlops . --network host
   sudo docker run --network="host" -it -p 8001:8001 azure-amlops:latest
```


### 3. Start Redis Server Inside Docker Container

```bash
   sudo docker ps # Get Container IDs
   sudo docker exec -it <container_id> bash #Get inside the container

   #Use either of the two below commands.
   nohup celery -A mlops_apis worker -l info -P solo -E -Q azure_async_q --concurrency=500 &        #Using Nohup
   celery -A mlops_apis worker -l info -P solo -E -Q azure_async_q --concurrency=500                #Without using Nohup
```


### 3. Authenticate The Docker Containers with AZURE CLI

```bash
   sudo docker exec -it <container_id> bash #Get inside the container
   echo <PAT> | az devops login --organization <organization_URL>
   chown -R www-data:www-data /root/.azure-devops #This is mandatory.

   #Example - echo l2ckn6zjch4hsfjcjtwczznuoozrqafgyaff7cya5m5ru3nwmfuq | az devops login --organization https://dev.azure.com/SyntbotsAI-RnD/
```


### 4. Creating a New Agent

```bash
   mkdir <agent_name>                                                             #Create an agent with any name, any path in the VM
   ssh -i mlops.pem azureuser@20.185.91.194                                        #Connect to the remote VM using SSH
   scp -i mlops.pem vsts-agent-linux-x64-2.204.0.tar.gz azureuser@20.185.91.194:. #Transfer files to the remote VM using SCP
```


### 5. Sample Requests for each API that you can use from Postman
```bash
   
   Dataset Upload API
   URL: http://localhost:8000/azure_dataset_upload/
   Method: POST
   Payload: FROM UI

   List Datasets API
   URL: http://127.0.0.1:8000/azure/datasets/list/
   Method: GET
   Payload: 
   N/A

   List Buckets API
   URL: http://127.0.0.1:8000/azure/buckets/list/
   Method: GET
   Payload:
   N/A

   List Create Bucket API
   URL: http://127.0.0.1:8000/azure/buckets/create/
   Method: POST
   Payload:
    {
    "name": "test_bucket"
    }

   List Create Pipeline API
   URL: http://127.0.0.1:8001/azure/pipeline/create/
   Method: POST
   Payload:
    {
    "pipeline_name": "GlassPipeline",
    "experiment_name":"GlassExperiment",
    "dataset_name": "azureml-blobstore-fa9d2517-040e-4c2e-be65-676d5c708710/glassdata/glass.csv",
    "train_columns": ["RI", "NA", "MG", "AL", "SI", "K", "CA", "BA", "FE"],
    "target_columns": "TYPE",
    "train_size": 0.85,
    "ml-compute_v_m_size": "STANDARD_DS12_V2",
    "aks-compute_v_m_size" : "STANDARD_DS12_V2",
    "missing_value":true,
    "remove_outlier":true,
    "agent_name": "New_Self_Hosted_Agent_V2",
    "model_selection": {
        "model":"Random Forest",
            "params":{
            "random_state":42,
            "n_estimators":200,
            "criterion":"gini"
            }
        },
    "tasks": {
    "run_traininfrasetup": true,
    "run_preprocess": true,
    "run_train": true,
    "run_deployinfrasetup": true,
    "run_deploytoaks": true,
    "run_publishendpoint": true
        }
    }

   View Dataset API
   URL: http://127.0.0.1:8000/azure/dataset/view/
   Method: POST
   Payload:
    {
    "dataset_name": "azureml-blobstore-fa9d2517-040e-4c2e-be65-676d5c708710/segmentation/segmentation.csv"
    }

   Execute Pipeline API
   URL: http://127.0.0.1:8000/azure/pipeline/create/execute/
   Method: POST
   Payload:
    {
        "definition_id": "187"
    }

   List Pipeline API
   URL: http://127.0.0.1:8000/pipelines/
   Method: GET
   Payload:
   N/A

   Pipeline Status API
   URL: http://127.0.0.1:8000/azure/pipelines/list/
   Method: POST
   Payload:
    {
        "pipeline_definition_id_list": [182]
    }


   Get Endpoint API
   URL: http://127.0.0.1:8000/azure/pipeline/getendpoint/
   Method: POST
   Payload:
    {
        "endpoint_name": "segmentation-endpoint-lny-myk"
    }


   Dataset Upload API
   URL: http://127.0.0.1:8000/azure/dataset/upload/
   Method: POST
   Payload:
    {
        "endpoint_name": "segmentation-endpoint-lny-myk"
    }

```





































































































































SECRET KEY
```
    {
        "PAT for Hosted Agent": "zaayvm43oexnvswefk2q5pe5ut3ejasx7nyazzdbw4qfoexsyavq",
        "PAT for Azure GIT": "glpat-S6GGb_mNe-_7Ekk-XySs"
    }
```