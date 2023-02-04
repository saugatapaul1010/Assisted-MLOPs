from rest_framework import serializers
from .models import SVCProvider,BucketStorage,Dataset
from rest_framework.reverse import reverse

class SVCProviderSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SVCProvider
        fields = ['id','name', 'cloud_provider', 'created_at', 'updated_at']

class BucketSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = BucketStorage
        fields = ['id','name', 'cloud_provider', 'storage_type' ,'created_at', 'updated_at']

class DatasetSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Dataset
        fields = ['dataset_name','id', 'created_at', 'updated_at', 'url', 'version', 'dataset_size', 'cloud_provider', 'filename']