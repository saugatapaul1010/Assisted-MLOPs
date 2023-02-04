from enum import unique
from django.db import models
from .utils import CloudProviders, FileUploadStatus, StandardResponse

class SVCProvider(models.Model):
    class Meta:
        unique_together = (('name', 'cloud_provider'),)
    name = models.CharField(max_length=200)
    cloud_provider = models.CharField(max_length=200,choices=CloudProviders.choices())
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class BucketStorage(models.Model):
    class Meta:
        unique_together = (('name','cloud_provider'),)
    name = models.CharField(max_length=200)
    cloud_provider = models.CharField(max_length=200,choices=CloudProviders.choices())
    storage_type = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

def file_using_key_path(instance, filename):
    return '{0}/{1}'.format(instance.key, filename)

class FineFile(models.Model):
    key = models.CharField(max_length=20)
    fine_file = models.FileField(upload_to=file_using_key_path)
    upload_id = models.CharField(max_length=200)
    filename = models.CharField(max_length=200)
    status = models.CharField(max_length=20,choices=FileUploadStatus.choices(),default="InProgress")
    def __str__(self):
        return self.fine_file.name


class Dataset(models.Model):
    
    dataset_name = models.CharField(max_length=200)
    filename = models.CharField(max_length=200)
    data_url = models.CharField(max_length=1000,default=None, blank=True, null=True) 
    bucket_name = models.CharField(max_length=200)
    cloud_provider = models.CharField(max_length=200,choices=CloudProviders.choices())
    upload_id = models.CharField(max_length=200,default=None, blank=True, null=True)
    version = models.CharField(max_length=50)
    visualize_url = models.CharField(max_length=1000,default=None, blank=True, null=True)    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)