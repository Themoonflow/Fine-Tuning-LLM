# Databricks notebook source
CATALOG = spark.sql("SELECT current_catalog()").first()[0]
SCHEMA = "DBACADEMY_ADV_GENAI_COURSE"

# Schema creation
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
_ = spark.sql(f"USE SCHEMA {SCHEMA}")

datasets_path = DA.paths.datasets.replace("/dbfs", "dbfs:")

print(f"This lesson will use {CATALOG}.{SCHEMA}. Use variables `CATALOG` and `SCHEMA` as needed for a user-specific schema to write your data to. If you're running this outside of the classroom environment, replace the variables `CATALOG` and `SCHEMA` with your own locations (note that this must be Unity Catalog and not the hive metastore).")

# COMMAND ----------

def create_tables(table_name: str, relative_path: str, datasets_path: str = datasets_path, schema: str = SCHEMA) -> None:
    """
    Create a Delta table from a Delta file at the specified path.

    Parameters:
    - table_name (str): The name of the table to be created.
    - relative_path (str): The relative path to the Delta file.
    - datasets_path (str): The base path where datasets are stored.
    - schema (str): The schema to use for the table.

    Returns:
    - None
    """
    path = f"{datasets_path}/{relative_path}"
    table_name = f"{schema}.{table_name}"

    df = spark.read.format("delta").load(path)
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    df.write.saveAsTable(table_name)

    print(f"Created table {table_name}")

# Uncomment tables as needed
create_tables("blogs_bronze", "blogs/bronze")
# create_tables("pyspark_code_bronze", "pyspark-code/bronze")
create_tables("pyspark_code_gold", "pyspark-code/gold") 
# create_tables("pyspark_code_gold_flat", "pyspark-code/gold-flat") 
# create_tables("spark_docs_gold", "spark-docs/gold")
# create_tables("spark_docs_bronze", "spark-docs/txt-raw")

print()
print("Completed table creation")