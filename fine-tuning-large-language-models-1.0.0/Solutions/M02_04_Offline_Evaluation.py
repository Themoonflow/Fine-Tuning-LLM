# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Offline Evaluation
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Evaluates the LLM outputs using built-in MLflow metrics and LLM-as-a-judge

# COMMAND ----------

# MAGIC %md
# MAGIC Install `databricks-genai` and other libraries.

# COMMAND ----------

# MAGIC %pip install mlflow==2.11.3 textstat==0.7.3 databricks-sdk==0.24.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

from databricks.sdk import WorkspaceClient

import mlflow
from mlflow import MlflowClient
from mlflow.metrics.genai.metric_definitions import answer_correctness, answer_similarity

# COMMAND ----------

training_run_id = "FILL IN" # fill this in with the relevant run id

if training_run_id == "FILL IN":
    raise Exception("Fill in the variable above with the run id you want to evaluate.")

# COMMAND ----------

serving_endpoint = "adv_genai_ift_model"

output_df = (spark.table(f"{CATALOG}.{SCHEMA}.llm_output_df")
             .withColumnRenamed("prompt", "inputs")
             .withColumnRenamed("response", "ground_truth"))
display(output_df)

# COMMAND ----------

# Depending on how you executed the run, you could use this code as well
# w = WorkspaceClient()
# model_name = w.serving_endpoints.get(name=serving_endpoint).config.served_entities[0].entity_name
# model_version = 1
# mlflow_client = MlflowClient(registry_uri="databricks-uc")

# # Retrieve model version object for registered model
# mv = mlflow_client.get_model_version(name=model_name, version=model_version)
# training_run_id = mv.run_id
# training_run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate using MLflow and LLM-as-a-judge
# MAGIC
# MAGIC In this section, we will use MLflow to generate evaluation metrics on the batch inference dataframe. We will also specify DBRX-instruct as our LLM judge to evaluate answer similarity and correctness. 
# MAGIC
# MAGIC Resources:
# MAGIC - [MLflow docs](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
# MAGIC   - [LLM evaluation metrics](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#llm-evaluation-metrics)
# MAGIC   - [Custom metrics](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#llm-evaluation-metrics)
# MAGIC - [Databricks blog post part 1](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
# MAGIC - [Databricks blog post part 2](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part)

# COMMAND ----------

llm_judge = "endpoints:/databricks-dbrx-instruct"
answer_correctness_metric = answer_correctness(model=llm_judge)
answer_similarity_metric = answer_similarity(model=llm_judge)

# We need to convert this Spark DataFrame to pandas DataFrame for mlflow.evaluate() 
output_pdf = output_df.toPandas()

with mlflow.start_run(run_id=training_run_id) as run: 
    results = mlflow.evaluate(data=output_pdf, 
                              targets="ground_truth",
                              predictions="llm_response",
                              model_type="text",
                              extra_metrics=[answer_correctness_metric, answer_similarity_metric]
                            )
    print(results.metrics)

# COMMAND ----------

results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC You can also go to MLflow Experiments run page to view the same metrics logged to the fine-tuning run.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>