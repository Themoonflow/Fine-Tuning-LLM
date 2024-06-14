# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
# DA.reset_lesson()                                 # Skipping to persist user-generated files in a given lesson
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.paths.working_dir = DA.paths.to_vm_path(DA.paths.working_dir)
DA.paths.datasets = DA.paths.to_vm_path(DA.paths.datasets)
DA.paths.user_db = DA.paths.to_vm_path(DA.paths.user_db)

# COMMAND ----------

# MAGIC %run ./Create-Tables

# COMMAND ----------

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe models developed or used in this course are for demonstration and learning purposes only.\nModels may occasionally output offensive, inaccurate, biased information, or harmful instructions.")