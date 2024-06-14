# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2: Instruction Fine-tuning
# MAGIC
# MAGIC This lab demonstrates how to perform instruction fine-tuning (IFT) on a pre-trained language model. 
# MAGIC
# MAGIC Objectives:
# MAGIC
# MAGIC 1. Trigger a single IFT run with specified hyperparameters

# COMMAND ----------

# MAGIC %md
# MAGIC Install `databricks-genai`.

# COMMAND ----------

# MAGIC %pip install databricks-genai==1.0.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

import pandas as pd

# COMMAND ----------

TABLE = "blogs_bronze"
VOLUME = "blog_ift_data" # Volume containing the instruction dataset

TRAIN_TABLE = "blog_title_generation_train_ift_data"
EVAL_TABLE = "blog_title_generation_eval_ift_data"

UC_MODEL_NAME = "blog_title_generation_llm"  # Name of model registered to Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1: Create a fine-tuning run

# COMMAND ----------

from databricks.model_training import foundation_model as fm

model =  "mistralai/Mistral-7B-Instruct-v0.2"
register_to = f"{CATALOG}.{SCHEMA}"
training_duration = "3ep"
learning_rate = "3e-06"
data_prep_cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

run = fm.create(
  model=model,
  train_data_path=f"{CATALOG}.{SCHEMA}.{TRAIN_TABLE}",
  eval_data_path=f"{CATALOG}.{SCHEMA}.{EVAL_TABLE}",
  data_prep_cluster_id=data_prep_cluster_id,
  register_to=register_to,
  training_duration=training_duration,
  learning_rate=learning_rate,
)
run

# COMMAND ----------

info = fm.get(run.name)

def display_run_html(ft_run_info):
    run_info_dict = vars(ft_run_info)

    keys = ["name", "status", "details", "model", "task_type", "learning_rate", "training_duration", "train_data_path",
            "eval_data_path", "register_to", "experiment_path", "eval_prompts", "custom_weights_path", "data_prep_cluster_id"]

    dict_summary = {key: str(run_info_dict[key]) for key in keys}
    info_html = pd.DataFrame.from_dict(dict_summary, orient="index", columns=["value"]).to_html()
    displayHTML(info_html)
    
display_run_html(info)

# COMMAND ----------

run.get_events()