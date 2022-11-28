import pandas as pd
import mlflow

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_mlflow_uri)

# TODO: Set the experiment 
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_mlflow_uri)
mlflow.projects.run(uri='.',
                experiment_name='RandomForestGridSearch', #'LinearRegressionGridSearch', 
                entry_point='tracking.py') # choose main or serve dep. on use case
