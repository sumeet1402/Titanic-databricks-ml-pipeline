# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC # Data Collection
# MAGIC
# MAGIC ## Objective
# MAGIC Learn how to collect data from CSV and Excel files and load them into Pandas DataFrames using Databricks.
# MAGIC
# MAGIC ## Dataset
# MAGIC - Dataset: Titanic Dataset
# MAGIC - Source: Kaggle
# MAGIC - File format: CSV
# MAGIC - Additional practice file: Excel

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## 1. Machine Learning Development Life Cycle
# MAGIC
# MAGIC A machine learning project generally follows these stages:
# MAGIC
# MAGIC 1. Problem Definition
# MAGIC 2. Data Collection
# MAGIC 3. Data Preprocessing
# MAGIC 4. Model Training
# MAGIC 5. Model Evaluation
# MAGIC 6. Deployment
# MAGIC 7. Monitoring
# MAGIC
# MAGIC In this notebook, I am focusing on the **Data Collection** stage.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Source
# MAGIC
# MAGIC I am using the Titanic Dataset from Kaggle for this practice project.
# MAGIC
# MAGIC The dataset contains information about a passenger, such as:
# MAGIC - Passenger class
# MAGIC - Age
# MAGIC - Sex
# MAGIC - Number of siblings or spouses
# MAGIC - Number of parents or children
# MAGIC - Fare
# MAGIC - Port of embarkation
# MAGIC - Survival status
# MAGIC
# MAGIC I downloaded the dataset in CSV format.

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.read_csv("/Volumes/workspace/default/ml_data/Titanic-Dataset.csv")

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Collect the Titanic dataset from Kaggle and load CSV and Excel files into Pandas DataFrames.
# MAGIC
# MAGIC - Load CSV data using `pd.read_csv()`.
# MAGIC - Load Excel data using `pd.read_excel()`.
# MAGIC - Store datasets in separate DataFrame variables.
# MAGIC - Verify the loaded data using `head()` and `shape`.