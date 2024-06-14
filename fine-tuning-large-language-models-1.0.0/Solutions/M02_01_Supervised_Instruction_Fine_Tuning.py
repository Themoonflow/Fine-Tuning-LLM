# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Supervised Instruction Fine-Tuning (IFT)
# MAGIC
# MAGIC In this demo, we are solving a documentation problem for many typical technical projects. In order to hit deadlines, writing documentation often is not the highest priority. This results in a lot of undocumented code and tribal knowledge. To help with code readability, we would like to use a LLM to generate documentation based on existing code. The project is written using Apache Spark; therefore, we will fine-tune a foundation model on [Apache Spark's documentation](https://github.com/apache/spark). 
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this module, you will be able to:
# MAGIC
# MAGIC 1. Understand the data preparation needed for supervised instruction fine-tuning
# MAGIC 1. Fine-tune a foundation model that results in a model adapted to generate documentation from code 
# MAGIC 1. Deploy a model using Model Serving
# MAGIC 1. Generate predictions at scale 
# MAGIC 1. Evaluate a model using MLflow 
# MAGIC
# MAGIC ## Demo 01: Prepare IFT Data and Fine-Tune Model
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC 1. Shows how to format data in two ways:
# MAGIC     - prompt-response pairs in a Delta table
# MAGIC     - convert the data into .jsonl format to save to a Unity Catalog Volume
# MAGIC 1. Conducts instruction fine-tuning (IFT) to adapt a foundation model to generate documentation from code 
# MAGIC
# MAGIC Note as of May 17, 2024: We are using a public preview feature, `from databricks.model_training import foundation_model as fm`, for fine-tuning. 
# MAGIC
# MAGIC **Compute Requirements**
# MAGIC - 14.3 LTS Machine Learning Runtime 
# MAGIC - Single Node 
# MAGIC - i3.xlarge

# COMMAND ----------

# MAGIC %pip install databricks-genai==1.0.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
from databricks.model_training import foundation_model as fm
import os
import glob

gold_df = spark.read.table(f"{CATALOG}.{SCHEMA}.pyspark_code_gold")

display(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Filter out any irrelevant data

# COMMAND ----------

@F.udf("string")
def parse_fn_def(fn_string: str) -> str:
    NUMPY_DOCSTRING_START = '"""'
    return fn_string[: fn_string.find(NUMPY_DOCSTRING_START)]

# COMMAND ----------

fn_with_docstrings = (
    gold_df
    .filter(F.col("doc_string").isNotNull())  # We want only functions with docstrings
    .filter(F.col("element_type") == "function")  # We only want functions, not classes or class methods
    .filter(F.col("args").isNotNull())  # remove null argument functions
    .filter(F.length("args") > 0)  # Remove functions with no arguments
    .filter(F.col("doc_string").contains("Parameters"))  # Get numpy style doc strings
    .withColumn("fn_def", parse_fn_def("source"))
)
display(fn_with_docstrings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Format 1: Create prompt-response pairs for the instruction fine-tuning dataset

# COMMAND ----------

INSTRUCTION = """The following is a pyspark function definition. Write a numpy style docstring for the provided function definition."""
DEF_KEY = "### Function definition"
RESPONSE_KEY = "### Numpy style docstring"

# COMMAND ----------

@F.udf("string")
def format_instructions(fn_def: str) -> str:
    return f"""{INSTRUCTION}\n{DEF_KEY}\n{fn_def}\n{RESPONSE_KEY}\n"""

# COMMAND ----------

# Function Documentation Formatter
ift_df = fn_with_docstrings.select(
    format_instructions("fn_def").alias("prompt"), F.col("doc_string").alias("response")
)
display(ift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Randomly split data into train and eval sets

# COMMAND ----------

train_df, eval_df = ift_df.randomSplit([0.9, 0.1], seed=42)

# COMMAND ----------

train_data_path = f"{CATALOG}.{SCHEMA}.ift_train"
eval_data_path = f"{CATALOG}.{SCHEMA}.ift_eval"

train_df.write.mode("overwrite").saveAsTable(train_data_path)
eval_df.write.mode("overwrite").saveAsTable(eval_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Starting a fine-tuning run
# MAGIC
# MAGIC Note that you run will be submitted and queued. The wait duration depends on the number of position you are in the queue.

# COMMAND ----------

data_prep_cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")

model = "mistralai/Mistral-7B-Instruct-v0.2"
register_to = f"{CATALOG}.{SCHEMA}"
training_duration = "3ep"
learning_rate = "5e-7"

run = fm.create(
    model=model,
    train_data_path=train_data_path,
    eval_data_path=eval_data_path,
    data_prep_cluster_id=data_prep_cluster_id, # Required for data in Delta Table
    register_to=register_to,
    training_duration=training_duration,
    learning_rate=learning_rate,
)
run

# COMMAND ----------

def display_run_html(ft_run_info):
    run_info_dict = vars(ft_run_info)

    keys = ["name", "status", "details", "model", "task_type", "learning_rate", "training_duration", "train_data_path",
            "eval_data_path", "register_to", "experiment_path", "eval_prompts", "custom_weights_path", "data_prep_cluster_id"]

    dict_summary = {key: str(run_info_dict[key]) for key in keys}
    info_html = pd.DataFrame.from_dict(dict_summary, orient="index", columns=["value"]).to_html()
    displayHTML(info_html)

display_run_html(fm.get(run.name))

# COMMAND ----------

# MAGIC %md
# MAGIC You can refresh the cell below to get the latest status of the fine-tuning run.

# COMMAND ----------

run.get_events()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Format 2: Create .jsonl file and save within a Unity Catalog Volume

# COMMAND ----------

# Pyspark Part File Renamer
TRAIN_VOLUME_NAME = "finetuning_train_data"
EVAL_VOLUME_NAME = "finetuning_eval_data"

TRAIN_VOLUME_DIRECTORY = f"/Volumes/{CATALOG}/{SCHEMA}/{TRAIN_VOLUME_NAME}"
EVAL_VOLUME_DIRECTORY = f"/Volumes/{CATALOG}/{SCHEMA}/{EVAL_VOLUME_NAME}"


def find_and_rename_pyspark_part_file(directory: str, new_name: str) -> None:
    """
    Pyspark will by default save with names that begin with `part-00000-tid-*`. Before saving, we repartition to one partition so there will only be 1 part file that we can easily find and rename.

    This function will rename the part-X.json file to the new_name
    """
    json_file_paths = glob.glob(f"{directory}/part-*.json", recursive=False)
    if len(json_file_paths) != 1:
        raise Exception("There should be only one json part file in the folder")
    os.rename(json_file_paths[0], f"{directory}/{new_name}")

# COMMAND ----------

_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{TRAIN_VOLUME_NAME}")
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{EVAL_VOLUME_NAME}")

# COMMAND ----------

# Pyspark Data Preparation and Export
(
    train_df.repartition(1)  # We just want one jsonl file
    .write.mode("overwrite")
    .json(TRAIN_VOLUME_DIRECTORY)
)
find_and_rename_pyspark_part_file(TRAIN_VOLUME_DIRECTORY, "train.jsonl")

# COMMAND ----------

# MAGIC %md
# MAGIC Repeat the same thing for evaluation data

# COMMAND ----------

(
    eval_df.repartition(1)  # We just want one jsonl file
    .write.mode("overwrite")
    .json(EVAL_VOLUME_DIRECTORY)
)

find_and_rename_pyspark_part_file(EVAL_VOLUME_DIRECTORY, "eval.jsonl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start a fine-tuning run 
# MAGIC
# MAGIC Mistral-7B fine-tuning setup

# COMMAND ----------

train_data_path = f"{TRAIN_VOLUME_DIRECTORY}/train.jsonl"
eval_data_path = f"{EVAL_VOLUME_DIRECTORY}/eval.jsonl"

model = "mistralai/Mistral-7B-Instruct-v0.2"
register_to = f"{CATALOG}.{SCHEMA}"
training_duration = "5ep"
learning_rate = "5e-7"

run2 = fm.create(
    model=model,
    train_data_path=train_data_path,
    eval_data_path=eval_data_path,
    register_to=register_to,
    training_duration=training_duration,
    learning_rate=learning_rate,
)
run2

# COMMAND ----------

display_run_html(fm.get(run2.name))

# COMMAND ----------

# MAGIC %md
# MAGIC You can refresh the cell below to get the latest status of the fine-tuning run.

# COMMAND ----------

run2.get_events()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can navigate to the MLflow UI to view the run and the run metrics.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>