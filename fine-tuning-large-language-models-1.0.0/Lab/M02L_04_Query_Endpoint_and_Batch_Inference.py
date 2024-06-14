# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4: Query Endpoint and Batch Inference
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Queries model serving endpoint with a single request
# MAGIC 1. Applies pandas user-defined functions (pandas UDFs) to scale out inference

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

from bs4 import BeautifulSoup
import json
import requests

import mlflow.deployments
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1: Supply a blog post URL 
# MAGIC First let's get the blog text for a single blog post, given a blog URL. Go to https://www.databricks.com/blog and select any of the URLs for a blog post.

# COMMAND ----------

def get_single_blog_post(url: str) -> str:
    """
    Retrieve the text of a single blog post given its URL.

    Args:
        url (str): URL of the blog post.

    Returns:
        str: cleaned text of the blog post.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # find the blog post text container
    blog_text_container = soup.find("div", class_="rich-text-blog")
    
    if blog_text_container:
        # remove HTML tags and extract the text
        blog_text = " ".join(blog_text_container.stripped_strings)
        
        # clean text
        blog_text = blog_text.replace("\\'", "'")
        blog_text = blog_text.replace(" ,", ",")
        blog_text = blog_text.replace(" .", ".")
        
        return blog_text
    else:
        print(f"Blog text not found for URL: {url}")
        return ""

url = "https://www.databricks.com/blog/illuminating-future-unveiling-databricks-power-analyzing-electrical-grid-assets-using-computer"
blog_post_text = get_single_blog_post(url)

blog_post_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## Construct a prompt using the template you have worked on in Lab 1.

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

blog_title_generation_template = PromptTemplate(
    instruction="The following is the text of a Databricks blog post. Create a title for the provided blog post.",
    blog_key="### Blog:",
    response_key="### Title:"
)

prompt = blog_title_generation_template.generate_prompt(blog_post_text)
print(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2: Query the endpoint

# COMMAND ----------

from mlflow.utils.databricks_utils import get_databricks_env_vars

mlflow_db_creds = get_databricks_env_vars("databricks")
API_TOKEN = mlflow_db_creds["DATABRICKS_TOKEN"]
WORKSPACE_URL = mlflow_db_creds["_DATABRICKS_WORKSPACE_HOST"]

ENDPOINT_NAME = "<FILL_IN>" # use the adv_genai_ift_model_lab endpoint we built for you
max_tokens = "<FILL_IN>"
temperature = "<FILL_IN>"

payload = {
    "inputs": {"prompt": [prompt]},
    "params": {"max_tokens": max_tokens, "temperature": temperature}
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    json=payload,
    headers=headers
)

predictions = response.json().get("predictions")
print(predictions[0]["candidates"][0]["text"])