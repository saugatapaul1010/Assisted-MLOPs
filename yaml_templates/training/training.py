# -*- coding: utf-8 -*-
"""
@author: saugata.paul@atos.net
"""

import os
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.run import Run
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import re
import seaborn as sn
import matplotlib.pyplot as plt
import joblib

class AzureClassification():

    def __init__(self, args):
        """
        Initialize Steps
        ----------------
            1. Initalize Azure ML Run Object
            2. Create directories
        """
        self.args = args
        self.run = Run.get_context()
        self.workspace = self.run.experiment.workspace
        os.makedirs('./model_metas', exist_ok=True)
        self.random_state = 1984

    def get_files_from_datastore(self, container_name, file_name):
        """
        Get the input CSV file from workspace's default data store
        Args :
            container_name : name of the container to look for input CSV
            file_name : input CSV file name inside the container
        Returns :
            data_ds : Azure ML Dataset object
        """
        datastore_paths = [(self.datastore, os.path.join(container_name,file_name))]
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

    def create_confusion_matrix(self, y_true, y_pred, name):
        """
        Create the confusion matrix.
        Args :
            y_true : ground truth
            y_pred : predictions
            name : base name of the CSV file
        Returns :
            None :
        """

        try:
            confm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
            print("Shape : ", confm.shape)

            df_cm = pd.DataFrame(confm, columns=np.unique(y_true), index=np.unique(y_true))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            df_cm.to_csv(name+".csv", index=False)
            self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

            plt.figure(figsize = (120,120))
            sn.set(font_scale=1.4)
            c_plot = sn.heatmap(df_cm, fmt="d", linewidths=.2, linecolor='black',cmap="Oranges", annot=True,annot_kws={"size": 16})
            plt.savefig("./outputs/"+name+".png")
            self.run.log_image(name=name, plot=plt)

        except Exception as e:
            #traceback.print_exc()
            print(e)
            logging.error("Create consufion matrix Exception")

    def create_outputs(self, y_true, y_pred, X_test, name):
        """
        Create prediction results as a CSV.
        Args :
            y_true : ground truth
            y_pred : predictions
            name : base name of the CSV file
        Returns :
            None :
        """
        pred_output = {"Actual "+self.args.target_column : y_true[self.args.target_column].values, "Predicted "+self.args.target_column: y_pred[self.args.target_column].values}
        pred_df = pd.DataFrame(pred_output)
        pred_df = pred_df.reset_index()
        X_test = X_test.reset_index()
        final_df = pd.concat([X_test, pred_df], axis=1)
        final_df.to_csv(name+".csv", index=False)
        self.run.upload_file(name="./outputs/"+name+".csv",path_or_stream=name+".csv")

    def validate(self, y_true, y_pred, X_test):
        """
        Validate model performances and log the outputs.
        Args :
            y_true : ground truth
            y_pred : predictions
            X_test : test dataset
        Returns :
            None :
        """
        self.run.log(name="Precision", value=round(precision_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Recall", value=round(recall_score(y_true, y_pred, average='weighted'), 2))
        self.run.log(name="Accuracy", value=round(accuracy_score(y_true, y_pred), 2))
        self.create_confusion_matrix(y_true, y_pred, "confusion_matrix")
        y_pred_df = pd.DataFrame(y_pred, columns = [self.args.target_column])
        self.create_outputs(y_true, y_pred_df,X_test, "predictions")
        self.run.tag(self.args.tag_name)

    def rf_model_training(self):
        """
        Train models using Random Forest
        """

        self.datastore = Datastore.get(self.workspace, self.workspace.get_default_datastore().name)
        print("Received datastore")

        #Read the processed CSV file (Not actual CSV)
        act_filename =  self.args.input_csv
        temp = act_filename.split(".")
        temp[0] = temp[0]+"_PreProcess"
        upload_filename = ".".join(temp)

        input_ds = self.get_files_from_datastore(self.args.container_name, file_name=upload_filename)
        self.final_df = input_ds.to_pandas_dataframe()

        # self.final_df = self.get_processed_files_from_swd()

        self.X = self.final_df[self.args.training_columns.split(",")]
        self.y = self.final_df[[self.args.target_column]]

        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=1-self.args.train_size,random_state=self.random_state)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state,n_jobs=-1)
        model.fit(X_train,y_train)

        # 5 Cross-validation
        CV = 5
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
        acc = np.mean(accuracies)
        print("Cross Validation accuracy mean: ", acc)

        y_pred = model.predict(X_test)
        print("Test Accuracy Score : ", accuracy_score(y_test, y_pred))

        joblib.dump(model, self.args.model_path)

        self.validate(y_test, y_pred, X_test)

        match = re.search('([^\/]*)$', self.args.model_path)
        # Upload Model to Run artifacts
        self.run.upload_file(name=self.args.artifact_loc + match.group(1),
                                path_or_stream=self.args.model_path)

        print("Run Files : ", self.run.get_file_names())
        self.run.complete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Azure DevOps Pipeline')
    parser.add_argument('--container_name', type=str, help='Path to default datastore container')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--dataset_name', type=str, help='Dataset name to store in workspace')
    parser.add_argument('--dataset_desc', type=str, help='Dataset description')
    parser.add_argument('--model_path', type=str, help='Path to store the model')
    parser.add_argument('--artifact_loc', type=str,help='DevOps artifact location to store the model', default='')
    parser.add_argument('--training_columns', type=str, help='model training columns comma separated')
    parser.add_argument('--target_column', type=str, help='target_column of model prediction')
    parser.add_argument('--train_size', type=float, help='train data size percentage. Valid values can be 0.01 to 0.99')
    parser.add_argument('--tag_name', type=str, help='Model Tag name')
    parser.add_argument('--processed_file_path', default = '/', type=str, help='processed dataset storage location path')
    args = parser.parse_args()
    classifier = AzureClassification(args)
    classifier.__init__(args)
    classifier.rf_model_training()
