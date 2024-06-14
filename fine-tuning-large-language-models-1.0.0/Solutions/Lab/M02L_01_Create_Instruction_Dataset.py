# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1: Create Instruction Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC In this lab, we create an instruction dataset to be used in instruction fine-tuning. The use case we are solving is generating blog post titles in the style of historic Databricks blog posts. For this we will prepare historic blog posts and their titles into an instruction dataset.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Load the raw Databricks articles data
# MAGIC 1. Filter out empty rows, and deduplicate the data
# MAGIC 1. Structure the blog text into a prompt
# MAGIC 1. Create a table with columns `prompt`, `response`, where the `response` column is the blog post title
# MAGIC 1. Use this table to write a JSONL file to a [Unity Catalog Volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

from typing import Iterator, List
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unity Catalog Configuration
# MAGIC
# MAGIC Define the Unity Catalog configuration variables, including the catalog, schema, input table, output table, and output volume/JSONL file names.

# COMMAND ----------

# UC Instruction dataset volume
INPUT_TABLE = "blogs_bronze"
OUTPUT_VOLUME = "blog_ift_data"
OUTPUT_TRAIN_JSONL = "blog_ift_train.jsonl"
OUTPUT_EVAL_JSONL = "blog_ift_eval.jsonl"
OUTPUT_TRAIN_TABLE = "blog_title_generation_train_ift_data"
OUTPUT_EVAL_TABLE = "blog_title_generation_eval_ift_data"

# COMMAND ----------

# MAGIC %md
# MAGIC If the defined Unity Catalog Volume does not exist, create it.

# COMMAND ----------

# Create the specified output volume if it doesn't already exist
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{OUTPUT_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Loading and Filtering
# MAGIC
# MAGIC Load the raw blog data from the specified Unity Catalog table and filter out rows with null or empty values in the 'text' or 'title' columns.
# MAGIC

# COMMAND ----------

def load_and_filter(table_name: str, response_col: str = "title") -> DataFrame:
    """
    Load table and filter null or empty strings in 'text' or `response_col`.

    Args:
        table_name: The name of the table to load.
        response_col: The column to filter for null or empty strings.

    Returns:
        Filtered DataFrame.
    """
    print(f"Loading table: {table_name}")
    df = spark.table(table_name)
    original_count = df.count()
    print(f"Row count: {original_count}")

    print(f"\nFilter null or empty strings in 'text' or '{response_col}'")
    filtered_df = filter_null_or_empty(df, ["text", response_col])
    filtered_count = filtered_df.count()
    print(f"Number of rows dropped: {original_count - filtered_count}")
    print(f"Filtered count: {filtered_count}")

    return filtered_df
  

def filter_null_or_empty(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Filter rows where any of the specified columns are null or empty.

    Args:
        df: The DataFrame to filter.
        columns: The list of columns to check for null or empty values.

    Returns:
        Filtered DataFrame.
    """
    print("Filter rows where any of the specified columns are null or empty...")
    for col in columns:
        print(f"\tColumn: {col}")
        df = df.filter((F.col(col).isNotNull()) & (F.col(col) != ""))
    return df

# COMMAND ----------

filtered_df = load_and_filter(table_name=f"{CATALOG}.{SCHEMA}.{INPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deduplication
# MAGIC
# MAGIC Deduplicate the filtered dataset based on the 'text' and 'title' columns to ensure unique blog posts.

# COMMAND ----------

filtered_deduped_df = filtered_df.drop_duplicates(subset=["text", "title"])
filtered_deduped_count = filtered_deduped_df.count()
print(f"Final deduplicated count: {filtered_deduped_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1: Add prompt column
# MAGIC
# MAGIC
# MAGIC Fill in the blanks to have the `PromptTemplate` class to return:
# MAGIC - instruction
# MAGIC - blog key
# MAGIC - blog text
# MAGIC - response

# COMMAND ----------

# # TODO

# class PromptTemplate:
#     """Class to represent a prompt template for instruction dataset generation."""

#     def __init__(self, instruction: str, blog_key: str, response_key: str) -> None:
#         self.instruction = <FILL_IN>
#         self.blog_key = <FILL_IN>
#         self.response_key = <FILL_IN>

#     def generate_prompt(self, blog_text: str) -> str:
#         """
#         Generate a prompt using the template and the given blog text.

#         Args:
#             blog_text: The text of the blog.

#         Returns:
#             Prompt template.
#         """
#         return f"""{<FILL_IN>}
# {<FILL_IN>}
# {<FILL_IN>}
# {<FILL_IN>}
# """

# COMMAND ----------


class PromptTemplate:
    """Class to represent a prompt template for instruction dataset generation."""

    def __init__(self, instruction: str, blog_key: str, response_key: str) -> None:
        self.instruction = instruction
        self.blog_key = blog_key
        self.response_key = response_key

    def generate_prompt(self, blog_text: str) -> str:
        """
        Generate a prompt using the template and the given blog text.

        Args:
            blog_text: The text of the blog.

        Returns:
            Prompt template.
        """
        return f"""{self.instruction}
{self.blog_key}
{blog_text}
{self.response_key}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2: Construct a Prompt Template 
# MAGIC
# MAGIC Hint: 
# MAGIC - `instruction` should include pointers to the LLM on generating title based on the provided blog post 
# MAGIC

# COMMAND ----------


blog_title_generation_template = PromptTemplate(
    instruction="The following is the text of a Databricks blog post. Create a title for the provided blog post.",
    blog_key="### Blog:",
    response_key="### Title:"
)

# COMMAND ----------

def add_instruction_prompt_column(df: DataFrame, prompt_template: PromptTemplate) -> DataFrame:
    """
    Add 'prompt' column to the DataFrame using the specified template.

    Args:
        df: Input DataFrame.
        prompt_template: Prompt template to use for generating prompts.

    Returns:
        DataFrame with 'prompt' column.
    """
    @F.pandas_udf(StringType())
    def generate_prompt(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for texts in batch_iter:
            prompts = texts.apply(prompt_template.generate_prompt)
            yield prompts

    return df.withColumn("prompt", generate_prompt(df["text"]))

# COMMAND ----------

# Add prompt column 
instruction_df = add_instruction_prompt_column(filtered_deduped_df, blog_title_generation_template)

# subset to prompt col, and rename title to response
instruction_df = instruction_df.selectExpr("prompt", "title as response")
display(instruction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Example prompt

# COMMAND ----------

print(instruction_df.select("prompt").limit(1).collect()[0]["prompt"])

# COMMAND ----------

# MAGIC %md
# MAGIC Split data randomly into train and test

# COMMAND ----------

train_df, eval_df = instruction_df.randomSplit([0.9,0.1], seed=42)
print(train_df.count(), eval_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 3: Write train/eval to separate tables

# COMMAND ----------


OUTPUT_TRAIN_TABLE = "blog_title_generation_train_ift_data"
OUTPUT_EVAL_TABLE = "blog_title_generation_eval_ift_data"

train_data_path = f"{CATALOG}.{SCHEMA}.{OUTPUT_TRAIN_TABLE}"
eval_data_path = f"{CATALOG}.{SCHEMA}.{OUTPUT_EVAL_TABLE}"

train_df.write.mode("overwrite").saveAsTable(train_data_path)
eval_df.write.mode("overwrite").saveAsTable(eval_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 4 (Optional): Convert Unity Catalog table to single JSONL file

# COMMAND ----------

def write_table_to_volume_as_jsonl(
    catalog: str,
    schema: str,
    table_name: str,
    volume: str,
    output_file_name: str) -> None:
    """
    Loads table from Unity Catalog and writes to a Unity Catalog volume as a single JSONL file.
    Intended for use on instruction dataset tables, consisting of "prompt" and "response" columns.

    Args:
        catalog: UC Catalog
        schema: UC Schema
        table_name: Name of the table to load and write as JSONL.
        volume: Name of the UC volume to write the JSONL file to.
        output_file_name: Name of the JSONL output file.
        limit: Optional limit for the number of rows to load from the table. If None, fetch all rows.
    """
    volume_path = f"{catalog}/{schema}/{volume}"
    temp_path = "/tmp/temp_json"

    # Load table from UC
    print("Loading table: ", table_name)
    instruct_dataset_df = spark.table(table_name)

    # Write out as JSON, coalescing to a single partition
    print("Coalesce into a single partition and write to a temporary JSON file...")
    print(f"\tWriting to: {temp_path}")
    (instruct_dataset_df.coalesce(1)
     .write
     .mode("overwrite")
     .option("header", "false")
     .json(temp_path))

    # Get the name of the single .part files
    part_files = dbutils.fs.ls(temp_path)
    json_part_file = next((file.path for file in part_files if file.name.startswith("part-")), None)
    if json_part_file:
        print("Moving JSON part file to Unity Catalog volume as JSONL file.")
        final_path = f"/Volumes/{volume_path}/{output_file_name}"
        print("Final path: ", final_path)
        dbutils.fs.mv(json_part_file, final_path)
        dbutils.fs.rm(temp_path, recurse=True)  # Clean up temp dir

        print(f"Successfully wrote Spark DataFrame as JSONL to {final_path}")
    else:
        print("No part file found. Check the temp path.")

# COMMAND ----------


_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{OUTPUT_VOLUME}")

# Convert the UC train table to single JSONL file
write_table_to_volume_as_jsonl(
    catalog=CATALOG,
    schema=SCHEMA,
    table_name=f"{CATALOG}.{SCHEMA}.{OUTPUT_TRAIN_TABLE}",
    volume=OUTPUT_VOLUME,
    output_file_name=OUTPUT_EVAL_JSONL
)

# Convert the UC eval table to single JSONL file'
write_table_to_volume_as_jsonl(
    catalog=CATALOG,
    schema=SCHEMA,
    table_name=f"{CATALOG}.{SCHEMA}.{OUTPUT_EVAL_TABLE}",
    volume=OUTPUT_VOLUME,
    output_file_name=OUTPUT_EVAL_JSONL
)