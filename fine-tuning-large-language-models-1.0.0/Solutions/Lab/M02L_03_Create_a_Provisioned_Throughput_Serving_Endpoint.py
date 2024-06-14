# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 3: Create a Provisioned Throughput Serving Endpoint 
# MAGIC
# MAGIC In this notebook we create a [Provisioned Throughput Foundation Model API](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html#provisioned-throughput-foundation-model-apis). Provisioned Throughput provides optimized inference for Foundation Models with performance guarantees for production workloads. 
# MAGIC
# MAGIC See [Provisioned throughput Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis) for a list of supported model architectures.
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC 1. Defines the model to deploy. This will be the fine-tuned model registered Unity Catalog
# MAGIC 1. Get the optimization info for the registered model
# MAGIC 1. Configure and create the endpoint
# MAGIC 1. Query the endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC # *!!DO NOT RUN THIS NOTEBOOK AT SUMMIT UNLESS YOU ARE AN INSTRUCTOR DUE TO CONCURRENT GPU CAPACITY CONSTRAINTS!!*

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.24.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

import json
import requests

import mlflow
import mlflow.deployments
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput,
    EndpointCoreConfigInput,
    AutoCaptureConfigInput,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Set up the configuration variables for Unity Catalog. We define the registered model name, and the model version to deploy. We also define the name of the endpoint, and define the name of the [inference table](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html) which will be created to log the endpoint request-response payloads.
# MAGIC

# COMMAND ----------

# Model Serving Endpoint variables
ENDPOINT_NAME = "adv_genai_ift_model_lab"
INFERENCE_TABLE_PREFIX = "ift_request_response"

# COMMAND ----------

# MAGIC %md
# MAGIC # Question 1: UC model name

# COMMAND ----------


# UC_MODEL_NAME = "<FILL_IN>"  # Name of model registered to Unity Catalog
# MODEL_VERSION = 1

# COMMAND ----------

UC_MODEL_NAME = (
    "ift-mistral-7b-instruct-v0-2-8rl6xp"  # Name of model registered to Unity Catalog
)
MODEL_VERSION = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get optimization information for the model
# MAGIC
# MAGIC Given the model name and model version we can retrieve the optimization info for the model.  This is the number of tokens/second that corresponds to 1 throughput unit for your specific model.

# COMMAND ----------

API_TOKEN, API_ROOT = get_api_credentials()

def get_model_optimization_info(full_model_name: str, model_version: int):
    """Get the model optimization information for the specified registered model and version."""
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}
    url = f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{full_model_name}/{model_version}"
    response = requests.get(url=url, headers=headers)
    return response.json()


# Get optimization info given the registered model
model_optimization_info = get_model_optimization_info(
    full_model_name=f"{CATALOG}.{SCHEMA}.{UC_MODEL_NAME}", model_version=MODEL_VERSION
)
print("model_optimization_info: ", model_optimization_info)
min_provisioned_throughput = model_optimization_info["throughput_chunk_size"]
# We set the max to be equal to minimum for cost savings. You could choose a higher number based on expected incoming request load.
max_provisioned_throughput = model_optimization_info["throughput_chunk_size"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged model is automatically deployed with optimized LLM serving.

# COMMAND ----------

w = WorkspaceClient()

_ = spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload")

print("Creating endpoint..")
w.serving_endpoints.create_and_wait(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=f"{CATALOG}.{SCHEMA}.{UC_MODEL_NAME}",
                entity_version=MODEL_VERSION,
                max_provisioned_throughput=max_provisioned_throughput,
                min_provisioned_throughput=min_provisioned_throughput,
            )
        ],
        auto_capture_config=AutoCaptureConfigInput(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            enabled=True,
            table_name_prefix=INFERENCE_TABLE_PREFIX,
        ),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint
# MAGIC To see your more information about your endpoint, go to the **Serving** on the left navigation bar and search for your endpoint name.
# MAGIC
# MAGIC Depending on the model size and complexity, it can take 30 minutes or more for the endpoint to get ready.