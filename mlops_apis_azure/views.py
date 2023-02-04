# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""

#from asyncio.windows_utils import pipe
from fileinput import filename
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from mlops_apis_azure import api_methods, tasks
from mlops_apis.settings import BASE_DIR, input_params
from django.core.files.storage import FileSystemStorage
import os
import json
from datetime import datetime
import time
import logging
import time
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import generic
from .utils import StandardResponse
from django_fine_uploader.views import FineUploaderView
from django_fine_uploader.fineuploader import ChunkedFineUploader
from distutils.command.upload import upload
from .models import SVCProvider, BucketStorage, FineFile, Dataset
from rest_framework.views import APIView
from .utils import FileUploadStatus, StandardResponse
from django.http import Http404
import os
from django_fine_uploader.views import FineUploaderView
from django.views.decorators.csrf import csrf_exempt
from .serializers import DatasetSerializer
import shutil
from io import StringIO
from six.moves import range
import pickle
import subprocess
import random
import string

from azure.devops.v6_0.pipelines.pipelines_client import PipelinesClient
from azure.devops.v6_0.build.build_client import BuildClient
from msrest.authentication import BasicAuthentication


path = os.path.join(BASE_DIR, "my_pat_token.txt")
with open(path, 'r') as file:
    PAT = file.read().replace('\n', '')

os.environ['AZURE_DEVOPS_EXT_PAT'] = PAT


"""
29 March:

connectedServiceNameARM - Check if it's dynamically created for each individual Service Prinicipals.
Taks execution from 1 to 10, suppose the jobs fails at 3, can we start the job next time from step 4? 
GCP already has it.
"""


class CeleryJobsView(APIView):
    """
    This function is used to track celery tasks. If you add any new tasks,
    you will have to define the same in the "processes" dictionary below. Currently,
    we are tracking three tasks. The function will return a StandardResponse json
    object.
    """
    processes = {
        "upload_data_azure_task": tasks.upload_data_azure_task, 
        "commit_file_in_yaml_to_azure_repo_task" : tasks.commit_file_in_yaml_to_azure_repo_task,
        "execute_current_pipeline_task": tasks.execute_current_pipeline_task
        }

    def post(self, request, format=None):
        task_id = request.data["task_id"]
        process_name = request.data["process_name"]
        print(task_id, process_name)
        task = self.processes[process_name].AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {
                "status": str(task.state),
                "success": "false"
            }

        elif task.state == 'FAILURE':
            response = {
                "status": str(task.state),
                "success": "false"
            }

        elif task.state == 'SUCCESS':
            response = {
                "status": str(task.state),
                "success": "true"
            }

        else:
            response = {
                "status": str(task.state),
                "success": "false",
                "error": str(task.info)
            }
        
        return StandardResponse.Response(True, "Task Status", response)

#Unused
class AzureDataUpload(APIView):
    """
    This is currently not getting used. Keep this for future reference. This 
    was initially used to upload the data from local path to azure blob containers.
    Since, the upload functionality was separately implemented in the UI, we are
    not making use of this block of code. You can use it for future references, if
    there's any requirement for you to add the upload functionality in local
    """
    # @csrf_exempt
    def post(self, request):

        start_time = time.time()

        if 'filename' in request.FILES:
            filename = request.FILES['filename']
            print("File Exist")
            static_folder_location = os.path.join(BASE_DIR, 'static')

            if not os.path.exists(static_folder_location):
                os.mkdir(static_folder_location)

            local_dataset_folder_path = os.path.join(
                BASE_DIR, 'static', filename.name.split('.')[0])
            print(local_dataset_folder_path)
            if not os.path.exists(local_dataset_folder_path):
                os.mkdir(local_dataset_folder_path)
            try:
                if (os.path.exists(os.path.join(local_dataset_folder_path, filename.name))):
                    os.remove(os.path.join(
                        local_dataset_folder_path, filename.name))
                print("File deleted.")
            except:
                print("File doesn't exist.")

            fs = FileSystemStorage(location=local_dataset_folder_path)
            fs.save(filename.name, filename)
            local_dataset_file_path = os.path.join(
                local_dataset_folder_path, filename.name)

            try:
                if (os.path.exists(local_dataset_file_path)):
                    print("Local File Saved Path", local_dataset_file_path)
                else:
                    print("File Doesn't Exist")
            except:
                print("File Doesn't Exist")

            input_params['input_file_path'] = local_dataset_file_path
            azure_dataset_location = api_methods.upload_data_azure(
                input_params)

            try:
                os.remove(os.path.join(
                    local_dataset_folder_path, filename.name))
                os.rmdir(local_dataset_folder_path)
            except:
                print("Failed to delete input folder")

            now = datetime.now()
        return HttpResponse(json.dumps(azure_dataset_location), content_type='application/json')


class AzureCreateBucket(APIView):
    """
    Used to make a function call to "create_bucket_api()",
    which is used to create a new bucket in the given azure
    container.
    """
    # @csrf_exempt
    def post(self, request):
        input_params['name'] = request.data['name']
        try:
            response = api_methods.create_bucket_api(input_params)
            create_status = response[0]
            primary_endpoint_bucket = response[1]

        except Exception as e:
            return StandardResponse.Response(False, "Exception", str(e))

        if(create_status):
            return StandardResponse.Response(True, "Success. ", primary_endpoint_bucket)
        else:
            return StandardResponse.Response(False, "Bucket Already Present. ", primary_endpoint_bucket)



class AzureListBucket(APIView):
    """
    Used to make a function call to "list_bucket_api()",
    which is used to create a list of all the buckets in the given azure
    container.
    """
    def get(self, request):
        try:
            container_names = api_methods.list_bucket_api(input_params)
        except Exception as e:
            return StandardResponse.Response(False, "Bucket list is empty.", str(e))
        return StandardResponse.Response(True, "Success. ", container_names)


class AzureListDataset(APIView):
    """
    Used to make a function call to "list_dataset_api()",
    which is used to create a list all datasets in the given azure
    container.
    """
    def get(self, request):
        try:
            dataset_list = api_methods.list_dataset_api(input_params)
        except Exception as e:
            return StandardResponse.Response(False, "Dataset list is empty.", str(e))

        return StandardResponse.Response(True, "Success. ", dataset_list)


class DataSetFileUploadView(FineUploaderView):
    """
    Uploading file to integrator before creating db entry for dataset.
    """
    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super(DataSetFileUploadView, self).dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        # print(form.cleaned_data)
        file_name = form.cleaned_data["qqfilename"]
        _, file_extension = os.path.splitext(form.cleaned_data["qqfilename"])
        print(file_extension)
        if file_extension not in [".csv", ".xlsx"]:
            return StandardResponse.Response(False, "only csv and xlsx file allowed", None)
        self.process_upload(form)
        data = {'success': True, "message": "success"}
        fineFile = FineFile.objects.filter(upload_id=self.upload.uuid).first()
        # print(self.upload.part_index)
        if not fineFile and self.upload.part_index == 0:
            FineFile.objects.create(fine_file="", upload_id=self.upload.uuid,
                                    filename=file_name, status=FileUploadStatus.InProgress.value)
        if self.upload.finished:
            data['data'] = {'file_url': self.upload.url}
            # Let's save in database?
            fineFile = FineFile.objects.filter(
                upload_id=self.upload.uuid).first()
            if fineFile:
                fineFile.status = FileUploadStatus.Compeleted.value
                fineFile.fine_file = self.upload.real_pathe
                fineFile.save()
            else:
                FineFile.objects.create(fine_file=self.upload.real_path, upload_id=self.upload.uuid,
                                        filename=file_name, status=FileUploadStatus.Compeleted.value)
            # send to azure api in chunks
            # dispatch celery file upload
            # print(os.path.basename(self.upload._full_file_path))
        if "data" in data:
            return StandardResponse.Response(data["success"], data["message"], data["data"])
        else:
            return StandardResponse.Response(data["success"], data["message"], None)


class AzureVizUpload(APIView):
    """
    Used to generate pandas visualization, and upload them to a specific
    location in the azure containers.
    """
    # @csrf_exempt
    def post(self, request):

        start_time = time.time()

        if 'upload_file' in request.FILES:
            filename = request.FILES['upload_file']
            upload_id = request.data['upload_id']
            print("File Exist")
            static_folder_location = os.path.join(BASE_DIR, 'static')

            if not os.path.exists(static_folder_location):
                os.mkdir(static_folder_location)

            local_dataset_folder_path = os.path.join(
                BASE_DIR, 'static', upload_id)

            if not os.path.exists(local_dataset_folder_path):
                os.mkdir(local_dataset_folder_path)
            try:
                if (os.path.exists(os.path.join(local_dataset_folder_path, filename.name))):
                    os.remove(os.path.join(
                        local_dataset_folder_path, filename.name))
                print("File deleted.")
            except:
                print("File doesn't exist.")

            fs = FileSystemStorage(location=local_dataset_folder_path)
            fs.save(filename.name, filename)
            local_dataset_file_path = os.path.join(
                local_dataset_folder_path, filename.name)

            try:
                if (os.path.exists(local_dataset_file_path)):
                    print("Local File Saved Path", local_dataset_file_path)
                else:
                    print("File Doesn't Exist")
            except:
                print("File Doesn't Exist")

            input_params['data_viz_file_path'] = local_dataset_file_path
            input_params['data_viz_upload_id'] = upload_id

            try:
                viz_url = api_methods.upload_data_viz_azure(input_params)
                return StandardResponse.Response(True, "Success. ", viz_url)
                os.remove(os.path.join(
                    local_dataset_folder_path, filename.name))
                os.rmdir(local_dataset_folder_path)
            except Exception as e:
                print("Failed to delete input folder")
                return StandardResponse.Response(True, "Success. ", str(e))


class CustomFineUploaderView(FineUploaderView):
    """
    Let's get the file url and add to the json response, so we can
    get it on the frontend. More info on `onComplete` callback on
    myapp/example.html
    """

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super(CustomFineUploaderView, self).dispatch(request, *args, **kwargs)

    def process_upload(self, form):
        self.upload = ChunkedFineUploader(form.cleaned_data, self.concurrent)
        if self.upload.concurrent and self.chunks_done:
            self.combine_chunks()
        else:
            self.upload.save()

    def combine_chunks(self):
        """Combine a chunked file into a whole file again. Goes through each part,
        in order, and appends that part's bytes to another destination file.
        Discover where the chunks are stored in settings.CHUNKS_DIR
        Discover where the uploads are saved in settings.UPLOAD_DIR
        """
        # So you can see I'm saving a empty file here. That way I'm able to
        # take advantage of django.core.files.storage.Storage.save (and
        # hopefully any other custom Django storage). In a nutshell the
        # ``final_file`` will get a valid name
        # django.core.files.storage.Storage.get_valid_name
        # and I don't need to create some dirs along the way to open / create
        # my ``final_file`` and write my chunks on it.
        # https://docs.djangoproject.com/en/dev/ref/files/storage/#django.core.files.storage.Storage.save
        # In my experience with, for example, custom AmazonS3 storages, they
        # implement the same behaviour.

        print("overidden func")
        self.upload.real_path = self.upload.storage.save(
            self.upload._full_file_path, StringIO())

        #Check file_Path value__ TODO
        with self.upload.storage.open(self.upload.real_path, 'wb') as final_file:
            for i in range(self.upload.total_parts):
                part = os.path.join(self.upload.chunks_path, str(i))
                with self.upload.storage.open(part, 'rb') as source:
                    final_file.write(source.read())
        shutil.rmtree(self.upload._abs_chunks_path)

    def form_valid(self, form):
        self.process_upload(form)
        data = {'success': True, "message": "success"}
        if self.upload.finished:  # Insi
            data['data'] = {'file_url': self.upload.url,
                            'task_id': '', 'process_name': 'upload_data_azure_task'}
            # Let's save in database?
            # FineFile.objects.create(fine_file=self.upload.real_path)
            data['data']['task_id'] = str(tasks.upload_data_azure_task.delay(
                self.upload.real_path, self.upload.uuid))

        if "data" in data:
            return StandardResponse.Response(data["success"], data["message"], data["data"])
        else:
            return StandardResponse.Response(data["success"], data["message"], None)


class AzureCreatePipeline(APIView):
    """
    This is where all the variable initializations takes place. All the request fields 
    are loaded and instantiated, and the entire training pipeline is setup dynamically.
    If there's a need to include any new variable in the request headers, you must access
    that value from within the post() method defined below. Returns a StandardResponse object.
    """
    # @csrf_exempt
    def post(self, request):
        #Variables comming from frontend
        dataset_name = str(request.data['dataset_name']).split("/")[-1]
        base_data_name = str(dataset_name.split(".")[0].split("_")[0]).lower()
        input_params['base_data_name'] = base_data_name
        input_params['base_data_name_for_commit'] = base_data_name
        input_params['repo_name'] = str(request.data['pipeline_name']).capitalize() + "-MLOps-Repository"
        input_params['pipeline_name'] = request.data['pipeline_name']
        input_params['pipeline_description'] = "{} Pipeline Creation".format(request.data['experiment_name'])
        input_params['EXPERIMENT_NAME'] = request.data['experiment_name']
        input_params['train_columns'] = ",".join(request.data['train_columns'])
        input_params['target_columns'] = request.data['target_columns']
        input_params['train_size'] = request.data['train_size']
        input_params['computeVMSize'] = request.data['ml-compute_v_m_size']
        input_params['missing_value'] = request.data['missing_value']
        input_params['remove_outlier'] = request.data['remove_outlier']
        input_params['agent_name'] = request.data['agent_name']
        
        #Model Selection
        input_params['model_selection'] = request.data['model_selection']
        input_params['run_traininfrasetup'] = request.data["tasks"]['run_traininfrasetup']
        input_params['run_preprocess'] = request.data["tasks"]['run_preprocess']
        input_params['run_train'] = request.data["tasks"]['run_train']
        input_params['run_deployinfrasetup'] = request.data["tasks"]['run_deployinfrasetup']
        input_params['run_deploytoaks'] = request.data["tasks"]['run_deploytoaks']
        input_params['run_publishendpoint'] = request.data["tasks"]['run_publishendpoint']
        input_params['run_deltraincluster'] = False
        input_params['run_delinfcluster'] = False

        #Variables which are formed dynamically using inputs from UI
        input_params['aks_vmSize'] = request.data['aks-compute_v_m_size']
        input_params['computeName'] = "trainvm"+base_data_name
        input_params['aks_clusterName'] = ("aksvm"+base_data_name)[:15]
        input_params['aks_service_name'] = "{}-endpoint-{}-{}".format(base_data_name,"".join(random.choices(string.ascii_letters, k=3)),"".join(random.choices(string.ascii_letters, k=3)))
        input_params['aks_service_name'] = input_params['aks_service_name'].lower()
        input_params['aks_Location'] = "centralindia"
        input_params['train_stage_run_configuration_name'] = "{}_training".format(base_data_name)
        #input_params['train_stage_input_csv'] = dataset_name       
        input_params['train_stage_model_path'] = "./models/{}_model.pkl".format(base_data_name)      
        input_params['train_stage_dataset_desc'] = "{}_DataSet_Description".format(base_data_name.upper())         
        input_params['train_stage_dataset_container_name'] = "{}data".format(base_data_name) 
        input_params['train_stage_dataset_name'] = '{}_ds'.format(base_data_name) 
        input_params['register_model_name'] = "{}".format(base_data_name.upper())
        input_params['register_model_algo_description'] = "{}_Decision_Tree_Classifier".format(base_data_name.upper()) 
        input_params['publish_model_artifactName'] = "{}TrainingArtifacts".format(base_data_name.upper())
        input_params['tagname'] = '{}_classification_tag'.format(base_data_name)
        input_params['CONTAINER_NAME'] = str(request.data['dataset_name']).split("/")[0]                       #bucketname
        input_params['train_stage_dataset_container_name'] = str(request.data['dataset_name']).split("/")[1]   #upload_id
        input_params['train_stage_input_csv'] = str(request.data['dataset_name']).split("/")[-1]                #filename


        save_file_location, azure_yaml_file = api_methods.generate_yaml_file(input_params)
        input_params['save_file_location'] = save_file_location
        input_params['azure_yaml_file'] = azure_yaml_file

        #Params needed for generation of training files.
        preprocess_config_dict = dict()
        preprocess_config_dict["missing_value_treatment"] = input_params["missing_value"]
        preprocess_config_dict["remove_outlier_treatment"] = input_params["remove_outlier"]
        input_params['preprocess_config'] = preprocess_config_dict
        input_params['user_input_training'] = request.data["tasks"]['run_traininfrasetup']

        input_params['model_config'] = {'rf_model_training': False, 'lr_model_training': False, 'xtc_model_training': False, 'svc_model_training': False}

        if(request.data["model_selection"]["model"] == "Random Forest"):
            input_params['model_config']['rf_model_training'] = True
        elif(request.data["model_selection"]["model"] == "Logistic Regression"):
            input_params['model_config']['lr_model_training'] = True
        elif(request.data["model_selection"]["model"] == "XTC"):
            input_params['model_config']['xtc_model_training'] = True
        elif(request.data["model_selection"]["model"] == "SVC"):
            input_params['model_config']['svc_model_training'] = True

            

        #Build Training Script
        api_methods.script_builder(input_params)

        #Build Train Config Script
        input_params['train_config_save_location'] = os.path.join(BASE_DIR, 'yaml_outputs/training/')
        save_file_location_config, config_file = api_methods.generate_runconfig_file(input_params)
        input_params['train_config_save_location'] = save_file_location_config
        input_params['train_config_file'] = config_file

        #Generate Conda Deployment Config File for each dataset
        input_params['conda_config_save_location'] = os.path.join(BASE_DIR, 'yaml_outputs/training/')
        conda_deployment_config_save_file_location, conda_config_file = api_methods.generate_conda_config_file(input_params)
        input_params['conda_config_save_location'] = conda_deployment_config_save_file_location
        input_params['conda_config_file'] = conda_config_file  

        #Build TEST & Deployment Config JSON, save it in the templates folder
        api_methods.create_deployment_config_json(input_params)

        #Generate Score.py file and smoke_test.py file dynamically based on amlops_config.json
        api_methods.generate_deployment_files(input_params)

        #Check if pipeline is already present or not.
        pipeline_exists, pipeline_dict = api_methods.check_if_pipeline_exists(input_params)

        # print(input_params)
        # #Save dictionary --> Use this to debu Update YAML function.
        # with open('/home/saugata/Desktop/input_params.pickle', 'wb') as handle:
        #     pickle.dump(input_params, handle)

        if(pipeline_exists):
            """
            If pipeline is present, then we are not creating a new pipeline.
            We will create a new experiment in the existing pipeline. Returns a StandardResponse object.
            """
            data = dict()
            
            task_id = tasks.commit_file_in_yaml_to_azure_repo_task.delay(input_params)
            input_params['pipeline_creation_status'] = False
            data['pipeline_details'] = {'pipeline_name': input_params['pipeline_name'],
                                        'definition_id': pipeline_dict[input_params['pipeline_name']],
                                        'process_name': "commit_file_in_yaml_to_azure_repo_task",
                                        'task_id': str(task_id)}  

            input_params['pipeline_name'] = input_params['pipeline_name']
            input_params['definition_id'] = pipeline_dict[input_params['pipeline_name']]
            input_params['PIPELINE_ID'] = input_params['definition_id']
            data['execution_parameters'] =  input_params  
        else:
            """
            If pipeline doesn't exist, Clone the template repository from previous branch and create a new
            repository. Build the pipeline using the newly created repository. Returns a StandardResponse object.
            """
            data = dict()
            input_params["new_pipeline_commit_repo"] = "https://SyntbotsAI-RnD@dev.azure.com/SyntbotsAI-RnD/{}/_git/{}".format(input_params['PROJECT_NAME'],input_params['repo_name'])
            input_params['created_repo_details'] = api_methods.create_new_repository(input_params)
            input_params['created_repo_import_details'] = api_methods.clone_existing_repo_to_new_repo(input_params)
            task_id = tasks.commit_file_in_yaml_to_azure_repo_task.delay(input_params)
            pipeline_creation_result_dic = api_methods.create_new_pipeline(input_params)
            input_params['pipeline_creation_status'] = pipeline_creation_result_dic['pipeline_creation_status']
            data['pipeline_details'] = {'pipeline_name': pipeline_creation_result_dic['name'],
                                        'definition_id': pipeline_creation_result_dic['id'],
                                        'process_name': "commit_file_in_yaml_to_azure_repo_task",
                                        'task_id': str(task_id)}  

            input_params['pipeline_name'] = pipeline_creation_result_dic['name']
            input_params['definition_id'] = pipeline_creation_result_dic['id']
            input_params['PIPELINE_ID'] = pipeline_creation_result_dic['id']
            data['execution_parameters'] =  input_params          
        if(task_id):
            return StandardResponse.Response(True, "Success. ", data)
        else:
            return StandardResponse.Response(False, "Error. ", "Pipeline is not created.")

class AzureExecutePipeline(APIView): 
    """
    Used to execute any pipeline based on their defintion id (pipeline ID).
    Returns a StandardResponse object.
    """
    @csrf_exempt
    def post(self, request):
        data = {'success': True, "message": "success"}
        data['data'] = {'task_id': '', 
                        'process_name': 'execute_current_pipeline_task'}
        #input_params = request.data['execution_parameters']
        print(request.data)
        input_params['PIPELINE_ID'] = request.data['definition_id']
        try:
            data['data']['task_id'] = str(tasks.execute_current_pipeline_task.delay(input_params))
            
            #data['data']['execution_parameters'] = input_params
        except Exception as e:
            return StandardResponse.Response(False, "Pipeline is not executed.", str(e))
        return StandardResponse.Response(data["success"], data["message"], data["data"])
        
class AzureListPipeline(APIView):
    """
    Used to list all the pipelines that are present in the azure dev ops portal.
    Details are displayed in tabular format in the response body. Returns a StandardResponse object.
    """
    def get(self,request):
        data = {'success': True, "message": "success"}
        data['pipeline_build_lists'] = api_methods.list_pipelines_in_dev_ops(input_params)
        return StandardResponse.Response(True, "Success. ", data)

class AzureViewDataset(APIView):
    """
    Used to to load a dataset from the azure data stores, converts it into a dictioanry 
    format and returns it. Returns a StandardResponse object.
    """
    @csrf_exempt
    def post(self, request):
        data = {'success': True, "message": "success"}
        #Variables comming from frontend
        blob_name = str(request.data['dataset_name']).split("/")[-1]
        container_name = str(request.data['dataset_name']).split("/")[0]
        folder_name = str(request.data['dataset_name']).split("/")[1]
        data['data'] = api_methods.return_dataset_from_azure_blobstorage(blob_name, container_name, folder_name, input_params)
        if(data['data']['dataset_present']):
            return StandardResponse.Response(True, "Success. ", data)
        else:
            return StandardResponse.Response(True, "Error. ", data)

class AzurePipelinesStatus(APIView):
    """
    This function is used to get the details of all the pipelines,
    based on the input definition id lists. Return a StandardResponse object.
    """
    @csrf_exempt
    def post(self, request):
        definition_id_list = request.data['pipeline_definition_id_list']
        data = {'success': True, "message": "success"}
        data['pipeline_status'] = api_methods.pipelines_status(input_params, definition_id_list)
        return StandardResponse.Response(True, "Success. ", data)

class AzureGetEndpoint(APIView):
    """
    Used to generate the endpoint details based on the name of the endpoint.
    Returns a StandardResponse object.
    """
    @csrf_exempt
    def post(self, request):
        endpoint_name = request.data['endpoint_name']
        data = {'success': True, "message": "success"}
        data['endpoint_details'] = api_methods.get_endpoint_details(input_params, endpoint_name)
        return StandardResponse.Response(True, "Success. ", data)