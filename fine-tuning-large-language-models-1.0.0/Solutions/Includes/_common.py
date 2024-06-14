# Databricks notebook source
# INSTALL_LIBRARIES
version = "v4.0.9"
if not version.startswith("v"): library_url = f"git+https://github.com/databricks-academy/dbacademy@{version}"
else: library_url = f"https://github.com/databricks-academy/dbacademy/releases/download/{version}/dbacademy-{version[1:]}-py3-none-any.whl"
pip_command = f"install --quiet --disable-pip-version-check {library_url}"

# COMMAND ----------

# MAGIC %pip $pip_command

# COMMAND ----------

from dbacademy import dbgems
from dbacademy.dbhelper import DBAcademyHelper, Paths, CourseConfig, LessonConfig

course_config = CourseConfig(course_code = "gen",
                             course_name = "adv-genai",
                             data_source_version = "v01",
                             install_min_time = "15 min",
                             install_max_time = "60 min",
                             supported_dbrs = ["14.3.x-cpu-ml-scala2.12", "14.3.x-gpu-ml-scala2.12"],
                             expected_dbrs = "{{supported_dbrs}}")


lesson_config = LessonConfig(name = None,
                             create_schema = False,
                             create_catalog = True,
                             requires_uc = True,
                             installing_datasets = True,
                             enable_streaming_support = False,
                             enable_ml_support = True)