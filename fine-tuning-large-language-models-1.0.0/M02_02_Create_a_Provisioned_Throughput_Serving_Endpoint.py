# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a Provisioned Throughput Serving Endpoint 
# MAGIC
# MAGIC In this notebook, we create a [Provisioned Throughput Foundation Model API](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html#provisioned-throughput-foundation-model-apis). Provisioned Throughput provides optimized inference for Foundation Models with performance guarantees for production workloads. 
# MAGIC
# MAGIC See [Provisioned throughput Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis) for a list of supported model architectures.
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC 1. Defines the model to deploy. This will be the fine-tuned model registered Unity Catalog
# MAGIC 1. Get the optimization info for the registered model
# MAGIC 1. Configure and create the endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC Install `databricks-sdk` to set up serving endpoint.

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.24.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

isInstructor = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Set up the configuration variables for Unity Catalog. We define the registered model name and the model version to deploy. We also define the name of the endpoint, and define the name of the [inference table](https://docs.databricks.com/en/machine-learning/model-serving/inference-tables.html) which will be created to log the endpoint request-response payloads.
# MAGIC

# COMMAND ----------

import json
import requests
import mlflow
import mlflow.deployments
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceAlreadyExists
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

# Model Serving Endpoint Variables
ENDPOINT_NAME = "adv_genai_ift_model"
INFERENCE_TABLE_PREFIX = "ift_request_response"

print(f"Using endpoint: {ENDPOINT_NAME}")
print(f"Using inference table prefix: {INFERENCE_TABLE_PREFIX}")

# COMMAND ----------

# Model info
UC_MODEL_NAME = "<mistralai/Mistral-7B-Instruct-v0.2>"  # Name of model registered to Unity Catalog by instructor from M02_01
MODEL_VERSION = 1
if UC_MODEL_NAME=="<FILL_IN>":
    raise Exception("UC_MODEL_NAME must be set to the name of the model registered to Unity Catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get optimization information for the model
# MAGIC
# MAGIC Given the model name and model version, we can retrieve the model optimization info, including throughput chunk size. This is the number of tokens/second that corresponds to 1 throughput unit for your specific model.

# COMMAND ----------

if isInstructor:
    # Get the API endpoint and token for the current notebook context
    API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
    API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

# COMMAND ----------

if isInstructor:
    def get_model_optimization_info(full_model_name: str, model_version: int):
        """Get the model optimization information for the specified registered model and version."""
        headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}
        url = f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{full_model_name}/{model_version}"
        response = requests.get(url=url, headers=headers)
        return response.json()


    # Get optimization info given the registered model
    model_optimization_info = get_model_optimization_info(full_model_name=f"{CATALOG}.{SCHEMA}.{UC_MODEL_NAME}", 
                                                        model_version=MODEL_VERSION)
    print("model_optimization_info: ", model_optimization_info)
    min_provisioned_throughput = model_optimization_info["throughput_chunk_size"]
    # We set the max to be equal to minimum for cost savings. You could choose a higher number based on expected incoming request load. 
    max_provisioned_throughput = model_optimization_info["throughput_chunk_size"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged model is automatically deployed with optimized LLM serving.
# MAGIC
# MAGIC This can take ~15 mins.

# COMMAND ----------

if isInstructor:
    w = WorkspaceClient()
    try:
        print("Attempting to create endpoint...")
        w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=[
                    ServedEntityInput(
                        entity_name=f"{CATALOG}.{SCHEMA}.{UC_MODEL_NAME}",
                        entity_version=MODEL_VERSION,
                        max_provisioned_throughput=max_provisioned_throughput,
                        min_provisioned_throughput=min_provisioned_throughput
                    )
                ],
                auto_capture_config = AutoCaptureConfigInput(
                    catalog_name=CATALOG,
                    schema_name=SCHEMA,
                    enabled=True,
                    table_name_prefix=INFERENCE_TABLE_PREFIX
                )
            )
        )
    except ResourceAlreadyExists:
        print("Endpoint already found...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint
# MAGIC To see your more information about your endpoint, go to the **Serving** on the left navigation bar and search for your endpoint name.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>