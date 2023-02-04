"""mlops_apis URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from mlops_apis_azure import views

#from mlops_apis_azure.views import AzureDataUpload, AzureDataList, AzureModelDeployment

#Previous URL Mappings, for falling back. (Before Ravi, Anuj, Snehashis had suggested to make the URL uniform across all the 3 clouds)
"""
urlpatterns = [
    path('admin/', admin.site.urls),
    path('azure_dataset_upload/', views.AzureDataUpload.as_view(), name='azure_dataset_upload'),
    path('create_bucket_in_azure/', views.AzureCreateBucket.as_view(), name='create_bucket_in_azure'),
    path('list_buckets_in_azure/', views.AzureListBucket.as_view(), name='list_buckets_in_azure'),
    path('list_datasets_in_azure/', views.AzureListDataset.as_view(), name='list_datasets_in_azure'),
    path('custom_fine_uploader_azure/', views.CustomFineUploaderView.as_view(), name='custom_fine_uploader_azure'),
    path('task/status/', views.CeleryJobsView.as_view(), name='celery_task_status'),
    path('datasets/fileupload/', views.DataSetFileUploadView.as_view(), name='chunkedfileupload'),
    path('vizfile/', views.AzureVizUpload.as_view(), name='data_visualization_upload'),
    path('pipeline/create/azure/', views.AzureCreatePipeline.as_view(), name='create_azure_pipeline'),
    path('pipelines/', views.AzureListPipeline.as_view(), name='list_azure_pipeline'),
    path('pipeline/create/azure/execute', views.AzureExecutePipeline.as_view(), name='execute_azure_pipeline'),
    path('view_dataset/', views.AzureViewDataset.as_view(), name='view_dataset'),
    path('pipeline/status/azure/', views.AzurePipelinesStatus.as_view(), name='pipelines_status'),
    path('pipeline/getendpoint/azure/', views.AzureGetEndpoint.as_view(), name='get_endpoint')
]
"""


urlpatterns = [
    path('admin/', admin.site.urls),
    path('azure_dataset_upload/', views.AzureDataUpload.as_view(), name='azure_dataset_upload'),
    path('azure/buckets/create/', views.AzureCreateBucket.as_view(), name='create_bucket_in_azure'),
    path('azure/buckets/list/', views.AzureListBucket.as_view(), name='list_buckets_in_azure'),
    path('azure/datasets/list/', views.AzureListDataset.as_view(), name='list_datasets_in_azure'),
    path('azure/dataset/upload/', views.CustomFineUploaderView.as_view(), name='custom_fine_uploader_azure'),
    path('task/status/', views.CeleryJobsView.as_view(), name='celery_task_status'),
    path('datasets/fileupload/', views.DataSetFileUploadView.as_view(), name='chunkedfileupload'),
    path('vizfile/', views.AzureVizUpload.as_view(), name='data_visualization_upload'),
    path('azure/pipeline/create/', views.AzureCreatePipeline.as_view(), name='create_azure_pipeline'),
    path('azure/pipelines/list/', views.AzureListPipeline.as_view(), name='list_azure_pipeline'),
    path('azure/pipeline/create/execute/', views.AzureExecutePipeline.as_view(), name='execute_azure_pipeline'),
    path('azure/dataset/view/', views.AzureViewDataset.as_view(), name='view_dataset'),
    path('azure/pipeline/status/', views.AzurePipelinesStatus.as_view(), name='pipelines_status'),
    path('azure/pipeline/getendpoint/', views.AzureGetEndpoint.as_view(), name='get_endpoint')
]





