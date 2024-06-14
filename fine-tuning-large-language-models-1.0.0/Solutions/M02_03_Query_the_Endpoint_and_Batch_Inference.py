# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Query the Endpoint and Batch Inference
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Queries model serving endpoint with a single request
# MAGIC 1. Applies pandas user-defined functions (pandas UDFs) to scale out inference

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Install `databricks-genai`.

# COMMAND ----------

# MAGIC %pip install openai==1.2.0 databricks-sdk==0.24.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

import requests
import json
import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient

import pandas as pd 
from typing import Iterator
import pyspark.sql.functions as F
from openai import OpenAI

# COMMAND ----------

endpoint_name = "adv_genai_ift_model"
eval_data_path = f"{CATALOG}.{SCHEMA}.ift_eval"

print(f"Using endpoint: {endpoint_name}")
print(f"Using eval data path: {eval_data_path}")

eval_df = spark.table(f"{eval_data_path}")
display(eval_df)

# COMMAND ----------

prompt = eval_df.select("prompt").first()[0]
prompt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query model serving endpoint with a single prompt example

# COMMAND ----------

w = WorkspaceClient()

temperature = 1.0 
max_tokens = 100
w_response = w.serving_endpoints.query(name=endpoint_name, 
                                       prompt=prompt, 
                                       temperature=temperature, 
                                       max_tokens=max_tokens)
print(w_response.choices[0].text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to scalar iterator pandas UDFs for batch inference
# MAGIC
# MAGIC Starting Spark 2.3, there are pandas user-defined functions, also known as vectorized UDFs, available in Python to improve the efficiency of UDFs. Pandas UDFs utilize Apache Arrow to speed up computation and accept an iterator of `pandas.Series` or `pandas.DataFrame`.
# MAGIC
# MAGIC When the number of records you’re working with is greater than **`spark.conf.get('spark.sql.execution.arrow.maxRecordsPerBatch')`**, which is 10,000 by default, you should see speed ups from using a scalar iterator Pandas UDF compared to using a pandas scalar UDF because the scalar iterator pandas UDF iterates through batches of `pd.Series`.
# MAGIC
# MAGIC It has the general syntax of: 
# MAGIC <br>
# MAGIC ```
# MAGIC @pandas_udf(...)
# MAGIC def predict(iterator):
# MAGIC     model = ... # load model
# MAGIC     for features in iterator:
# MAGIC         yield model.predict(features)
# MAGIC ```
# MAGIC
# MAGIC Refer to [this page for documentation](https://docs.databricks.com/en/udf/pandas.html#iterator-of-series-to-iterator-of-series-udf).

# COMMAND ----------

# MAGIC %md
# MAGIC Since Databricks Python SDK's Workspace Client only works on the driver node, when we use pandas UDFs that leverage workers, we will not be able to use Workspace Client. As such, we pivot to using the `requests` library.

# COMMAND ----------

@F.pandas_udf("string")
def get_prediction_udf(batch_prompt: Iterator[pd.Series]) -> Iterator[pd.Series]:

    import mlflow

    max_tokens = 100 
    temperature = 1.0
    api_root = mlflow.utils.databricks_utils.get_databricks_host_creds().host
    api_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }
    
    for batch in batch_prompt:
        result = []
        for prompt in batch:
            data = {"inputs": {"prompt": [prompt]},
                    "params": {"max_tokens": max_tokens, 
                               "temperature": temperature}
                    }

            response = requests.post(
                url=f"{api_root}/serving-endpoints/{endpoint_name}/invocations",
                json=data,
                headers=headers
            )
    
            if response.status_code == 200:
                endpoint_output = json.dumps(response.json())
                data = json.loads(endpoint_output)
                prediction = data.get("predictions")
                predicted_docs = prediction[0]["candidates"][0]["text"].split('"""')[1]
                result.append(predicted_docs)
            else:
                result.append(response.raise_for_status())
    yield pd.Series(result)

# COMMAND ----------

output_df = eval_df.withColumn("llm_response", get_prediction_udf("prompt"))

# COMMAND ----------

output_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.llm_output_df")

# COMMAND ----------

display(spark.table(f"{CATALOG}.{SCHEMA}.llm_output_df"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>