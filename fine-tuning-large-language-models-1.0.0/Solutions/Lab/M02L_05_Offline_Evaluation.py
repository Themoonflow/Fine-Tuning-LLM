# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 5: Offline Evaluation 
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Queries model serving endpoint with a single request
# MAGIC 1. Queries the endpoint with a batch of requests
# MAGIC     - In the demo, you used pandas UDFs to submit a batch of requests. In this lab, you will simplify modify the request structure to accept a list of prompts that can be submitted against the model serving endpoint

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.24.0 textstat==0.7.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json
import mlflow
from mlflow import MlflowClient
from mlflow.metrics.genai.metric_definitions import answer_similarity
import pandas as pd

# COMMAND ----------

INSTRUCTION_DATASET_TABLE = "blog_title_generation_eval_ift_data"
table_name = f"{CATALOG}.{SCHEMA}.{INSTRUCTION_DATASET_TABLE}"

API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().host
WORKSPACE_URL = mlflow.utils.databricks_utils.get_databricks_host_creds().token
ENDPOINT_NAME = "adv_genai_ift_model_lab"
max_tokens = 128
temperature = 0.9

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1: Format the dataframe appropriately
# MAGIC
# MAGIC `mlflow.evaluate` requires the eval DataFrame to be formatted as a pandas df with "inputs" and "ground_truth"

# COMMAND ----------

eval_df = (spark.table(table_name)
           .select("prompt", "response")
           .withColumnRenamed("prompt", "inputs")
           .withColumnRenamed("response", "ground_truth")
           .toPandas() 
           )
    
print(eval_df.count())
display(eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2: Submit a batch of prompts against the endpoint
# MAGIC
# MAGIC Rather than using pandas UDFs as in the demo, here, we are going to directly submit a batch of prompts against the endpoint. To start with, we are going to test with just a single prompt.

# COMMAND ----------

def get_predictions(prompts, max_tokens, temperature, model_serving_endpoint):
    from mlflow.utils.databricks_utils import get_databricks_env_vars
    import requests

    mlflow_db_creds = get_databricks_env_vars("databricks")
    API_TOKEN = mlflow_db_creds["DATABRICKS_TOKEN"]
    WORKSPACE_URL = mlflow_db_creds["_DATABRICKS_WORKSPACE_HOST"]

    payload = {"inputs": {"prompt": prompts},
               "params": {"max_tokens": max_tokens, "temperature": temperature}
               }
    
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_TOKEN}"
               }
    
    response = requests.post(url=f"{WORKSPACE_URL}/serving-endpoints/{model_serving_endpoint}/invocations",
                             json=payload,
                             headers=headers
                             )
    predictions = response.json().get("predictions")
    print(predictions)
    return predictions

# COMMAND ----------

get_predictions(prompts=[eval_df["inputs"][0]], 
                max_tokens=max_tokens, 
                temperature=temperature,
                model_serving_endpoint=ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC Above, you see that you can query the endpoint with a single prompt. Now it's your turn to modify the prompt structure for the endpoint to accept a batch of prompts in `eval_df`. Note that since `eval_df` contains 187 rows, it can take ~5 mins for the responses to return. Optionally, you can try with submitting just 2 prompts to speed things up.

# COMMAND ----------

predictions = get_predictions(prompts=eval_df["inputs"].tolist()[:2], 
                              max_tokens=max_tokens, 
                              temperature=temperature,
                              model_serving_endpoint=ENDPOINT_NAME)

# COMMAND ----------

results = [pred["candidates"][0]["text"] for pred in predictions]
eval_df["generated_title"] = pd.Series(results, index=eval_df.index[:2])
display(eval_df)

# COMMAND ----------

# save the evaluation dataframe
spark.createDataFrame(eval_df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.eval_blog_df")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 3: Evaluate the LLM-generated titles
# MAGIC
# MAGIC Now that we have all the LLM-generated titles, let's evaluate their quality!

# COMMAND ----------

w = WorkspaceClient()
model_name = w.serving_endpoints.get(name=ENDPOINT_NAME).config.served_entities[0].entity_name
model_version = 1
mlflow_client = MlflowClient(registry_uri="databricks-uc")

# Retrieve model version object for registered model
# Note this will fail if you do not have the right UC permissions
mv = mlflow_client.get_model_version(name=model_name, version=model_version)
training_run_id = mv.run_id

# COMMAND ----------


with mlflow.start_run(run_id=training_run_id) as run: 
    results = mlflow.evaluate(data=eval_df, 
                              targets="ground_truth",
                              predictions="generated_title",
                              model_type="text"
                            )
    
    print(json.dumps(results.metrics, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 4: Use LLM as a judge
# MAGIC
# MAGIC In addition to the metrics above, let's use LLM as a judge to generate more metrics. Since we already generated the out-of-the-box metrics, we are going to turn off the default metrics by removing the `model_type` argument and generate only LLM-judge metrics.

# COMMAND ----------

# ANSWER 

llm_judge = "endpoints:/databricks-dbrx-instruct"
answer_similarity_metric = answer_similarity(model=llm_judge)

with mlflow.start_run(run_id=training_run_id) as run: 
    results = mlflow.evaluate(data=eval_df, 
                              targets="ground_truth",
                              predictions="generated_title",
                              extra_metrics=[answer_similarity_metric]
                            )
    print(json.dumps(results.metrics, indent=2))