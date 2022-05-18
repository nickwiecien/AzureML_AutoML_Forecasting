# Azure Machine Learning - AutoML Forecasting Pipeline Sample

![Background](img/01.png?raw=true "Background")

## Overview

This repo contains sample notebooks for constructing and Azure Machine Learning pipeline to:

- Retrieve data from an AML-linked blob datastore
- Format and organize retrieved data into train/forecast time-series 
- Run a AutoML forecasting training job using the extracted training time-series
- Register the best performing model derived from AutoML
- Generate a forward-looking forecast using the trained model 

The notebooks here will create a published pipeline endpoint in your Azure ML workspace that can be used for future model training/forecasting activities. Additionally, this pipeline is parameterized so you can optionally point at different data sources to train models over different sets of input data. 

As part of a quickstart we have included a separate environment setup notebook `01_Setup_AML_Env.ipynb` which creates a standalone datastore in the AML workspace and uploads a sample dataset containing information on sales of orange juice. The data used here was taken from the [OJ Sales Simulated Dataset](https://docs.microsoft.com/en-us/azure/open-datasets/dataset-oj-sales-simulated?tabs=azureml-opendatasets) made available as part of [Azure's Open Datasets](https://docs.microsoft.com/en-us/azure/open-datasets/overview-what-are-open-datasets).

## Prerequisites

To run the code included in this repo you must have access to an [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/) workspace. Details on provisioning an AML workspace can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources).

## Getting Started

We recommend running the notebooks contained in this repo from an Azure ML Compute instance (these are standalone VMs designed to be used for ML development). You can provision a new Compute Instance by following the guide below - we recommend selecting a `Standard_DS3_v2` SKU VM for this task.

[Create and Manage an Azure Machine Learning Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance)

Once your compute instance is provisioned, launch the terminal and execute the following commands:

```
cd Users/<YOUR-USERNAME>
git clone https://github.com/nickwiecien/AzureML_AutoML_Forecasting
```

This will surface all code from this repo inside your compute instance environment. Open both the `01_Setup_AML_Env.ipynb` and `02_Create_Model_Training_Pipeline.ipynb` notebooks and execute in sequence.