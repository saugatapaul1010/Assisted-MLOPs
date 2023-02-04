import os
from azureml.core import Workspace, Datastore, Dataset
import argparse
import numpy as np
os.environ['AZURE_DEVOPS_EXT_GITHUB_PAT'] = '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'
os.environ['AZURE_DEVOPS_EXT_PAT'] = 'l2ckn6zjch4hsfjcjtwczznuoozrqafgyaff7cya5m5ru3nwmfuq'


class AzurePreprocessing():

    def __init__(self, args):
        '''
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Load Workspace
        '''
        self.args = args
        self.workspace = Workspace.from_config()
        self.random_state = 1984

    def get_files_from_datastore(self, container_name, file_name, workspace, datastore):
        '''
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
            workspace: name of the workspace
            datastore: name of the datastore
        Returns :
            data_ds : Azure ML Dataset object
        '''

        datastore_paths = [(datastore, os.path.join(container_name,file_name))]
        data_ds = Dataset.Tabular.from_delimited_files(path=datastore_paths)
        dataset_name = self.args.dataset_name
        if dataset_name not in workspace.datasets:
            print("Registering Model to Workspace")
            data_ds = data_ds.register(workspace=workspace,
                        name=dataset_name,
                        description=self.args.dataset_desc,
                        tags={'format': 'CSV'},
                        create_new_version=True)
        else:
            print('Dataset {} already in workspace '.format(dataset_name))
        return data_ds

    def upload_processed_file_to_datastore(self):

        '''
        Stores the processed file in intermediate host directory,
        uploads the file to azure datastorage,
        deleted the file from host directory after upload is complete.
        '''
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)
        processed_file_temp_path = self.args.processed_file_path
        if not os.path.exists(processed_file_temp_path):
            os.makedirs(processed_file_temp_path)
        self.final_df.to_csv(os.path.join(processed_file_temp_path, upload_filename), index=None)
        self.datastore.upload(src_dir=processed_file_temp_path, target_path=self.args.container_name)
        print("Processed file uploaded to Azure Datastorage")

        #Remove Temporary Files
        os.remove(os.path.join(processed_file_temp_path, upload_filename))
        print("Processed file deleted from Host Agent")

    def missing_value_treatment(self):
        """
        Removes any missing value from the data.
        """

        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name,
                                                self.args.input_csv,
                                                self.workspace,
                                                self.datastore)
        self.final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",self.final_df.info())
        print("Input DF Head",self.final_df.head())
        for feature in self.final_df.columns:
            if feature in self.final_df.select_dtypes(include=np.number).columns.tolist():
                self.final_df[feature].fillna(self.final_df[feature].mean(), inplace=True)
            else:
                self.final_df[feature].fillna(self.final_df[feature].mode()[0], inplace=True)

        self.upload_processed_file_to_datastore()

    def remove_outlier_treatment(self):
        """
        Removes outliers from the data.
        """
        from scipy import stats
        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")
        input_ds = self.get_files_from_datastore(self.args.container_name,
                                                    self.args.input_csv,
                                                    self.workspace,
                                                    self.datastore)
        self.final_df = input_ds.to_pandas_dataframe()
        print("Input DF Info",self.final_df.info())
        print("Input DF Head",self.final_df.head())
        num_features = self.final_df.select_dtypes(include=np.number).columns.tolist()
        self.final_df[num_features] = self.final_df[num_features][(np.abs(stats.zscore(self.final_df[num_features])) < 3).all(axis=1)]
        # for feature in self.final_df.columns:
        #     if feature in self.final_df.select_dtypes(include=np.number).columns.tolist():
        #         mean = np.mean(self.final_df[feature])
        #         std = np.std(self.final_df[feature])
        #         threshold = 3
        #         outlier = []
        #         for i in self.final_df[feature]:
        #             z = (i-mean)/std
        #             if z > threshold:
        #                 outlier.append(i)
        #         for i in outlier:
        #             self.final_df[feature] = np.delete(self.final_df[feature], np.where(self.final_df[feature]==i))
        self.upload_processed_file_to_datastore()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')
    parser.add_argument('--parent_bucket_name', default='azureml-blobstore-fa9d2517-040e-4c2e-be65-676d5c708710', type=str, help='Name of container where data is present')
    parser.add_argument('--container_name', default='glassdata', type=str, help='Name of folder where data is present')
    parser.add_argument('--input_csv', default='glass.csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', default = 'glass_ds', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', default = 'GLASS_DataSet_Description', type=str, help='Dataset description')
    parser.add_argument('--training_columns', default = 'RI,NA,MG,AL,SI,K,CA,BA,FE', type=str, help='model training columns comma separated')
    parser.add_argument('--target_column', default = 'TYPE', type=str, help='target_column of model prediction')
    parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')
    args = parser.parse_args()
    preprocessor = AzurePreprocessing(args)
    preprocessor.__init__(args)
    preprocessor.missing_value_treatment()
    preprocessor.remove_outlier_treatment()
