# Import required packages
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
from sklearn import preprocessing
import numpy as np
from datetime import datetime, timedelta

#Parse input arguments
parser = argparse.ArgumentParser("Get tabular data from attached datastore, split into separate files, and register as datasets in the AML workspace")
parser.add_argument('--training_data', dest='training_data', required=True)
parser.add_argument('--forecasting_data', dest='forecasting_data', required=True)
parser.add_argument('--datastore_relative_path', type=str, required=True)
parser.add_argument('--raw_dataset_name', type=str, required=True)
parser.add_argument('--train_dataset_name', type=str, required=True)
parser.add_argument('--forecast_dataset_name', type=str, required=True)
parser.add_argument('--timestamp_column', type=str, required=True)
parser.add_argument('--cutoff_date', type=str, required=True)

args, _ = parser.parse_known_args()
training_data = args.training_data
forecasting_data = args.forecasting_data
datastore_relative_path = args.datastore_relative_path
raw_dataset_name = args.raw_dataset_name
train_dataset_name = args.train_dataset_name
forecast_dataset_name = args.forecast_dataset_name
timestamp_column = args.timestamp_column
cutoff_date = args.cutoff_date

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Connect to expected Blob Store
ds = Datastore.get(ws, 'timeseriesdatastore')

#Read dataset from AML Datastore
dataset = Dataset.Tabular.from_delimited_files(path=(ds, f'{datastore_relative_path}/*'))
source_df = dataset.to_pandas_dataframe()

# Make directory on mounted storage for output dataset
os.makedirs(training_data, exist_ok=True)
os.makedirs(forecasting_data, exist_ok=True)

# Save Outputs for AutoML
train_filename = "train.csv"
forecast_filename = "forecast.csv"
before_split_date = source_df[timestamp_column] < cutoff_date
train_df, forecast_df = source_df[before_split_date], source_df[~before_split_date]

train_df.to_csv(os.path.join(training_data, train_filename), index=False)
forecast_df.to_csv(os.path.join(forecasting_data, forecast_filename), index=False)

timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

# Save Named Outputs
filename = './rawdata.csv'
source_df.to_csv(filename, index=False)
ds.upload_files([filename], relative_root='', target_path='rawdata/{}'.format(timestamp), overwrite=True)
datapath = DataPath(ds, 'rawdata/{}'.format(timestamp))
new_ds = Dataset.Tabular.from_delimited_files(path=datapath)
new_ds.register(ws, name = raw_dataset_name, create_new_version=True)

filename = './traindata.csv'
train_df.to_csv(filename, index=False)
ds.upload_files([filename], relative_root='', target_path='traindata/{}'.format(timestamp), overwrite=True)
datapath = DataPath(ds, 'traindata/{}'.format(timestamp))
new_ds = Dataset.Tabular.from_delimited_files(path=datapath)
new_ds.register(ws, name = train_dataset_name, create_new_version=True)

filename = './forecastdata.csv'
forecast_df.to_csv(filename, index=False)
ds.upload_files([filename], relative_root='', target_path='forecastdata/{}'.format(timestamp), overwrite=True)
datapath = DataPath(ds, 'forecastdata/{}'.format(timestamp))
new_ds = Dataset.Tabular.from_delimited_files(path=datapath)
new_ds.register(ws, name = forecast_dataset_name, create_new_version=True)