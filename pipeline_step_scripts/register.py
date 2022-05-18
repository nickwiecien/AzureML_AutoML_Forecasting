# Import required packages
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
import shutil
import joblib
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from datetime import datetime, timedelta

# #Parse input arguments
parser = argparse.ArgumentParser("Evaluate new model and register if more performant")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--target_column', type=str, required=True)
parser.add_argument('--result_dataset_name', type=str, required=True)

args, _ = parser.parse_known_args()
model_name = args.model_name
target_column = args.target_column
result_dataset_name = args.result_dataset_name

#Get current run
current_run = Run.get_context()

#Get parent run
parent_run = current_run.parent

#Get associated AML workspace
ws = current_run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

#Get testing dataset
forecast_datset = current_run.input_datasets['Forecasting_Data']
forecast_df = forecast_datset.to_pandas_dataframe()

#Separate inputs from outputs (actuals). Create separate dataframes for testing champion and challenger.
try:
    forecast_df = forecast_df.drop(columns=[target_column])
except Exception:
    pass

#Get updated 'challenger' model
for c in parent_run.get_children():
    if 'AutoML' in c.name:
        best_child_run_id = c.tags['automl_best_child_run_id']
        automl_run = ws.get_run(best_child_run_id)
        automl_run.download_files('outputs', output_directory='challenger_outputs', append_prefix=False)
        challenger_model = joblib.load('challenger_outputs/model.pkl')
        print(os.listdir('challenger_outputs'))
        print()
        print(best_child_run_id)
        print()
        challenger_tags = {'Parent Run ID': parent_run.id, 'AutoML Run ID': best_child_run_id}
        
#Calculate challenger metrics
challenger_preds, _trans = challenger_model.forecast(forecast_df)
print(challenger_preds)

forecast_df['Forecasted_Quantity'] = challenger_preds

import os
os.rename('challenger_outputs', 'outputs')
Model.register(model_path="outputs",
                      model_name=model_name,
                      tags=challenger_tags,
                      workspace=ws)

timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

os.makedirs(result_dataset_name, exist_ok=True)
# Save Named Outputs
filename = './result_data.csv'
forecast_df.to_csv(filename, index=False)
ds.upload_files([filename], relative_root='', target_path='forecastdata/{}'.format(timestamp), overwrite=True)
datapath = DataPath(ds, 'forecastdata/{}'.format(timestamp))
new_ds = Dataset.Tabular.from_delimited_files(path=datapath)
new_ds.register(ws, name = result_dataset_name, create_new_version=True)        