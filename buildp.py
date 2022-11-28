from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

print("\n\n yesssir we here \n\n")

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_mlflow_uri)

mlflow.projects.run(uri='.',
                experiment_name='RFR', #'LinearRegressionGridSearch', 
                entry_point='main') # choose main or serve dep. on use case